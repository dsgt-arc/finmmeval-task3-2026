"""Workflow runner executed by the bridge in a subprocess.

Separation of concerns:
- Boots the existing DeepFund workflow with the right config and DB state.
- Uses the decision-making code exactly as it already exists.
- Writes the final action to stdout as JSON for the bridge to consume.
- Avoids any HTTP or request-validation logic.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DECISION_MAKING_DIR = REPO_ROOT / "decision_making"
for path in (str(DECISION_MAKING_DIR), str(REPO_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from decision_making.run_decision_making import load_portfolio_config  # noqa: E402
from graph.workflow import AgentWorkflow  # noqa: E402
from util.db_helper import db_initialize, get_db  # noqa: E402


def _load_payload(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_base_config() -> dict:
    config_path = Path(os.getenv("DECISION_BRIDGE_CONFIG", REPO_ROOT / "decision_making" / "config" / "dev.yaml"))
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _build_config(payload: dict) -> dict:
    # Reuse the normal workflow config and override only the fields that come
    # from the competition request.
    cfg = _load_base_config()
    cfg["exp_name"] = os.getenv("DECISION_BRIDGE_EXP_NAME", "api_endpoint")
    cfg["tickers"] = payload.get("symbol") or []
    cfg["trading_date"] = datetime.strptime(payload["date"], "%Y-%m-%d")
    cfg["planner_mode"] = cfg.get("planner_mode", False)
    return cfg


def _latest_action(exp_name: str, ticker: str) -> str:
    db = get_db()
    memory = db.get_decision_memory(exp_name, ticker, 1)
    if memory:
        action = str(memory[0].get("action", "")).strip().upper()
        if action:
            return action
    raise RuntimeError(f"No decision found for {ticker}")


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: python -m api.decision_bridge_worker <payload.json>", file=sys.stderr)
        return 2

    payload_path = Path(sys.argv[1])
    payload = _load_payload(payload_path)
    cfg = _build_config(payload)

    # Initialize the isolated SQLite database for this request.
    db_initialize(use_local_db=True)
    db = get_db()
    config_id = load_portfolio_config(cfg, db)

    # This is the actual DeepFund workflow the other team owns.
    workflow = AgentWorkflow(cfg, config_id)
    workflow.run(config_id)

    ticker = cfg["tickers"][0]
    # Read the latest action back from the workflow DB and print it as JSON.
    action = _latest_action(cfg["exp_name"], ticker)
    print(json.dumps({"recommended_action": action}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
