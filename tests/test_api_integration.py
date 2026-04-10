"""End-to-end integration tests for the competition API."""

from __future__ import annotations

from pathlib import Path
import sys

from fastapi.testclient import TestClient
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from api.simple_trading_api import app  # noqa: E402
from decision_making.ama_data import load_data  # noqa: E402

pytestmark = [pytest.mark.integration, pytest.mark.slow]


def _real_payload(symbol: str, trading_date: str) -> dict:
    # Build the request from real downloaded market data so this test exercises
    # the same payload shape the organizers use.
    df = load_data(symbol, download_if_missing=False, competition_data=True)
    row = df.filter(df["date"] == trading_date).to_dicts()[0]
    history = [
        {"date": item["date"], "price": item["prices"]}
        for item in df.filter(df["date"] <= trading_date).select(["date", "prices"]).tail(10).to_dicts()
    ]

    return {
        "date": trading_date,
        "price": {symbol: row["prices"]},
        "news": {symbol: row["news"]},
        "symbol": [symbol],
        "momentum": {symbol: row["momentum"]},
        "10k": {symbol: row["10k"]},
        "10q": {symbol: row["10q"]},
        "history_price": {symbol: history},
    }


@pytest.mark.parametrize("symbol", ["TSLA", "BTC"])
def test_competition_action_runs_real_workflow(monkeypatch, symbol):
    # Keep the integration test deterministic by telling the bridge which
    # config to use, then hit the API for real.
    monkeypatch.setenv("DECISION_BRIDGE_CONFIG", str(ROOT / "decision_making" / "config" / "debug_pm.yaml"))
    monkeypatch.setenv("DECISION_BRIDGE_EXP_NAME", f"api_integration_test_{symbol.lower()}")

    client = TestClient(app)
    payload = _real_payload(symbol, "2026-03-19")

    response = client.post("/competition_action/", json=payload)

    assert response.status_code == 200
    assert response.json()["recommended_action"] in {"BUY", "HOLD", "SELL"}
