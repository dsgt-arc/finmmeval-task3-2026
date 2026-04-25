"""Orchestration bridge between the API and the DeepFund workflow.

Separation of concerns:
- Receives the raw payload from the FastAPI layer.
- Runs the existing workflow in a controlled subprocess with a timeout.
- Normalizes the result into a valid `BUY` / `HOLD` / `SELL` action.
- Keeps process management and request isolation out of the HTTP file.
"""

from __future__ import annotations

import json
import os
import logging
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict

VALID_ACTIONS = {"BUY", "HOLD", "SELL"}
logger = logging.getLogger(__name__)


def _validate_payload(payload: Dict[str, Any]) -> str:
    symbols = payload.get("symbol") or []
    if not symbols:
        raise ValueError("Payload did not include any symbols")
    return symbols[0]


def _build_worker_env(db_path: Path) -> dict[str, str]:
    # The worker runs in a subprocess, so we have to rebuild the import path
    # and point it at a request-specific SQLite file.
    env = os.environ.copy()
    repo_root = Path(__file__).resolve().parent.parent
    decision_pkg = repo_root / "decision_making"
    pythonpath_parts = [str(decision_pkg), str(repo_root)]
    if env.get("PYTHONPATH"):
        pythonpath_parts.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_parts)
    env["DB_PATH"] = str(db_path)
    env.setdefault("DECISION_BRIDGE_CONFIG", str(repo_root / "decision_making" / "config" / "dev.yaml"))
    env.setdefault("DECISION_BRIDGE_EXP_NAME", "api_endpoint")
    return env


def _normalize_action(action: str) -> str:
    normalized = action.strip().upper()
    if normalized not in VALID_ACTIONS:
        raise RuntimeError(f"Decision worker returned invalid action: {action!r}")
    return normalized


def recommend_action(payload: Dict[str, Any]) -> str:
    """Run the existing DeepFund workflow and return the latest action string."""

    # The API already validated the request shape. Here we only enforce the
    # minimum fields the workflow needs to run.
    _validate_payload(payload)
    if not payload.get("date"):
        raise ValueError("Payload did not include a trading date")

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "api.sqlite3"
        payload_path = Path(tmpdir) / "payload.json"
        payload_path.write_text(json.dumps(payload), encoding="utf-8")

        env = _build_worker_env(db_path)
        # Run the worker as a separate Python process so the API stays thin and
        # the workflow can use its normal imports and DB initialization.
        cmd = [sys.executable, "-m", "api.decision_bridge_worker", str(payload_path)]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=170)
        except subprocess.TimeoutExpired:
            logger.warning("Decision worker timed out; defaulting to HOLD")
            return "HOLD"
        except Exception as exc:
            logger.warning("Decision worker could not be started: %s", exc)
            return "HOLD"

        if result.returncode != 0:
            logger.warning("Decision worker failed; defaulting to HOLD: %s", result.stderr.strip() or result.stdout.strip())
            return "HOLD"

        try:
            data = json.loads(result.stdout)
        except json.JSONDecodeError:
            logger.warning("Decision worker returned invalid JSON; defaulting to HOLD")
            return "HOLD"

        action = data.get("recommended_action")
        if not isinstance(action, str):
            logger.warning("Decision worker returned invalid payload; defaulting to HOLD: %r", data)
            return "HOLD"

        return _normalize_action(action)
