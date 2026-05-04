"""Workflow runner executed by the bridge in a subprocess.

Separation of concerns:
- Boots the DS@GT StockTron workflow with the right config and DB state.
- Uses the decision-making code exactly as it already exists.
- Writes the final action to stdout as JSON for the bridge to consume.
- Avoids any HTTP or request-validation logic.
"""

from __future__ import annotations

from datetime import datetime
from collections import Counter
import json
import os
from pathlib import Path
import sys

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DECISION_MAKING_DIR = REPO_ROOT / "decision_making"
for path in (str(DECISION_MAKING_DIR), str(REPO_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from graph.workflow import AgentWorkflow  # noqa: E402
from util.db_helper import db_initialize, get_db  # noqa: E402
from util.logger import logger  # noqa: E402

from decision_making.run_decision_making import load_portfolio_config  # noqa: E402


def _load_payload(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _normalize_tickers(payload: dict) -> list[str]:
    symbols = payload.get("symbol") or list((payload.get("price") or {}).keys())
    if not symbols:
        raise ValueError("Payload did not include any symbols")

    # Keep the organizer order, drop duplicates, and only keep symbols that
    # still have a price entry in the request.
    prices = payload.get("price") or {}
    normalized: list[str] = []
    seen: set[str] = set()
    for symbol in symbols:
        if symbol in seen:
            continue
        if symbol not in prices:
            continue
        seen.add(symbol)
        normalized.append(symbol)

    if not normalized:
        raise ValueError("Payload did not include any symbols with prices")

    return normalized


def _load_base_config() -> dict:
    config_path = Path(os.getenv("DECISION_BRIDGE_CONFIG", REPO_ROOT / "decision_making" / "config" / "api.yaml"))
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _build_config(payload: dict) -> dict:
    # Reuse the normal workflow config and override only the fields that come
    # from the competition request.
    cfg = _load_base_config()
    # Preserve the YAML config name by default so Cloud Run serves the exact
    # requested workflow configuration. An explicit env var can still override
    # it when we intentionally want a different database namespace.
    cfg["exp_name"] = os.getenv("DECISION_BRIDGE_EXP_NAME") or cfg.get("exp_name", "api_endpoint")
    cfg["tickers"] = _normalize_tickers(payload)
    cfg["trading_date"] = datetime.strptime(payload["date"], "%Y-%m-%d")
    cfg["planner_mode"] = cfg.get("planner_mode", False)
    # Pass the full payload so the news pipeline can extract news/10k/10q
    cfg["api_payload"] = payload
    return cfg


def _latest_action(exp_name: str, tickers: list[str]) -> str:
    db = get_db()
    actions: list[str] = []

    for ticker in tickers:
        memory = db.get_decision_memory(exp_name, ticker, 1)
        if not memory:
            logger.warning("No decision found for %s; skipping it in action aggregation", ticker)
            continue

        action = str(memory[0].get("action", "")).strip().upper()
        if action:
            actions.append(action)

    if not actions:
        raise RuntimeError("No decision found for any ticker")

    # Prefer the majority action. If there is a tie, keep HOLD as the safest
    # fallback and otherwise use the first action we observed.
    counts = Counter(actions)
    top_count = max(counts.values())
    candidates = [action for action, count in counts.items() if count == top_count]
    if "HOLD" in candidates:
        return "HOLD"
    if "BUY" in candidates:
        return "BUY"
    if "SELL" in candidates:
        return "SELL"
    return actions[0]


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: python -m api.decision_bridge_worker <payload.json>", file=sys.stderr)
        return 2

    try:
        payload_path = Path(sys.argv[1])
        payload = _load_payload(payload_path)
        request_id = os.getenv("DECISION_BRIDGE_REQUEST_ID", "unknown")
        logger.info(
            "Worker start request_id=%s date=%s symbols=%s",
            request_id,
            payload.get("date"),
            payload.get("symbol") or list((payload.get("price") or {}).keys()),
        )
        cfg = _build_config(payload)
        logger.info(
            "Worker config built request_id=%s exp_name=%s tickers=%s config=%s",
            request_id,
            cfg.get("exp_name"),
            cfg.get("tickers"),
            os.getenv("DECISION_BRIDGE_CONFIG", str(REPO_ROOT / "decision_making" / "config" / "api.yaml")),
        )

        # Initialize the isolated SQLite database for this request.
        logger.info("Worker initializing isolated SQLite database request_id=%s", request_id)
        db_initialize(use_local_db=True)
        db = get_db()
        config_id = load_portfolio_config(cfg, db)
        logger.info("Worker config ready request_id=%s config_id=%s", request_id, config_id)

        # DS@GT StockTron workflow.
        logger.info("Worker starting workflow request_id=%s config_id=%s", request_id, config_id)
        workflow = AgentWorkflow(cfg, config_id)
        workflow.run(config_id)
        logger.info("Worker workflow completed request_id=%s config_id=%s", request_id, config_id)

        # Read the latest action back from the workflow DB and print it as JSON.
        action = _latest_action(cfg["exp_name"], cfg["tickers"])
        logger.info("Worker selected action request_id=%s action=%s", request_id, action)
        print(json.dumps({"recommended_action": action}))
        return 0
    except Exception:
        logger.exception("Worker failed request_id=%s", os.getenv("DECISION_BRIDGE_REQUEST_ID", "unknown"))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
