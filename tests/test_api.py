"""Tests for the competition API wrapper."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from api import decision_bridge
from api import decision_bridge_worker
from api.simple_trading_api import app


@pytest.fixture
def client():
    # FastAPI's TestClient lets us call the app like an HTTP server without
    # actually starting uvicorn.
    return TestClient(app)


@pytest.fixture
def sample_payload():
    # A representative organizer payload, using the same shape that the
    # competition sends to the endpoint.
    return {
        "date": "2026-03-19",
        "price": {"TSLA": 380.29998779296875},
        "news": {"TSLA": ["Tesla-centred news on 2026-03-19"]},
        "symbol": ["TSLA"],
        "momentum": {"TSLA": "bearish"},
        "10k": {"TSLA": []},
        "10q": {"TSLA": []},
        "history_price": {
            "TSLA": [
                {"date": "2026-03-10", "price": 399.2349853515625},
                {"date": "2026-03-11", "price": 407.82000732421875},
                {"date": "2026-03-12", "price": 395.010009765625},
                {"date": "2026-03-13", "price": 391.20001220703125},
                {"date": "2026-03-14", "price": 391.20001220703125},
                {"date": "2026-03-15", "price": 391.20001220703125},
                {"date": "2026-03-16", "price": 395.55999755859375},
                {"date": "2026-03-17", "price": 399.2699890136719},
                {"date": "2026-03-18", "price": 392.7799987792969},
                {"date": "2026-03-19", "price": 380.29998779296875},
            ]
        },
    }


@pytest.fixture
def btc_payload():
    # A BTC payload with sparse filing context to mirror the organizer note
    # that 10-K / 10-Q may be null on some days.
    return {
        "date": "2026-03-19",
        "price": {"BTC": 83000.0},
        "news": {"BTC": ["Bitcoin news on 2026-03-19"]},
        "symbol": ["BTC"],
        "momentum": {"BTC": "neutral"},
        "10k": None,
        "10q": None,
        "history_price": {
            "BTC": [
                {"date": "2026-03-10", "price": 81500.0},
                {"date": "2026-03-11", "price": 82000.0},
                {"date": "2026-03-12", "price": 82500.0},
                {"date": "2026-03-13", "price": 81800.0},
                {"date": "2026-03-14", "price": 82100.0},
                {"date": "2026-03-15", "price": 82600.0},
                {"date": "2026-03-16", "price": 82800.0},
                {"date": "2026-03-17", "price": 82900.0},
                {"date": "2026-03-18", "price": 82750.0},
                {"date": "2026-03-19", "price": 83000.0},
            ]
        },
    }


@pytest.fixture
def multi_symbol_payload(sample_payload, btc_payload):
    return {
        "date": "2026-03-19",
        "price": {"TSLA": 380.29998779296875, "BTC": 83000.0},
        "news": {
            "TSLA": ["Tesla-centred news on 2026-03-19"],
            "BTC": ["Bitcoin news on 2026-03-19"],
        },
        "symbol": ["TSLA", "BTC"],
        "momentum": {"TSLA": "bearish", "BTC": "neutral"},
        "10k": {"TSLA": [], "BTC": None},
        "10q": {"TSLA": [], "BTC": None},
        "history_price": {
            "TSLA": sample_payload["history_price"]["TSLA"],
            "BTC": btc_payload["history_price"]["BTC"],
        },
    }


def test_health_endpoint(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_competition_action_returns_exact_response_shape(client, sample_payload, monkeypatch):
    # monkeypatch temporarily replaces the bridge so this test only checks the
    # API wrapper behavior, not the full workflow.
    monkeypatch.setattr("api.simple_trading_api.recommend_action", lambda payload: "buy")

    response = client.post("/competition_action/", json=sample_payload)

    assert response.status_code == 200
    assert response.json() == {"recommended_action": "BUY"}


def test_competition_action_passes_request_id_when_bridge_accepts_it(client, sample_payload, monkeypatch):
    observed = {}

    def fake_recommend_action(payload, request_id=None):
        observed["payload"] = payload
        observed["request_id"] = request_id
        return "hold"

    monkeypatch.setattr("api.simple_trading_api.recommend_action", fake_recommend_action)

    response = client.post("/competition_action/", json=sample_payload)

    assert response.status_code == 200
    assert response.json() == {"recommended_action": "HOLD"}
    assert observed["payload"]["symbol"] == ["TSLA"]
    assert isinstance(observed["request_id"], str)
    assert observed["request_id"]


def test_competition_action_falls_back_to_price_symbol_when_symbol_missing(client, sample_payload, monkeypatch):
    sample_payload = dict(sample_payload)
    sample_payload.pop("symbol")

    observed = {}

    def fake_recommend_action(payload):
        observed["payload"] = payload
        return "SELL"

    # Capture the payload that the API passes into the bridge so we can verify
    # the symbol fallback logic.
    monkeypatch.setattr("api.simple_trading_api.recommend_action", fake_recommend_action)

    response = client.post("/competition_action/", json=sample_payload)

    assert response.status_code == 200
    assert response.json() == {"recommended_action": "SELL"}
    assert observed["payload"]["symbol"] == ["TSLA"]


def test_competition_action_falls_back_to_all_price_symbols_when_symbol_missing(client, multi_symbol_payload, monkeypatch):
    sample_payload = dict(multi_symbol_payload)
    sample_payload.pop("symbol")

    observed = {}

    def fake_recommend_action(payload):
        observed["payload"] = payload
        return "BUY"

    monkeypatch.setattr("api.simple_trading_api.recommend_action", fake_recommend_action)

    response = client.post("/competition_action/", json=sample_payload)

    assert response.status_code == 200
    assert response.json() == {"recommended_action": "BUY"}
    assert observed["payload"]["symbol"] == ["TSLA", "BTC"]


def test_competition_action_rejects_missing_price(client, sample_payload):
    sample_payload = dict(sample_payload)
    sample_payload.pop("price")

    response = client.post("/competition_action/", json=sample_payload)

    assert response.status_code == 422


def test_competition_action_rejects_missing_date(client, sample_payload):
    sample_payload = dict(sample_payload)
    sample_payload.pop("date")

    response = client.post("/competition_action/", json=sample_payload)

    assert response.status_code == 422


def test_competition_action_accepts_null_optional_context(client, sample_payload, monkeypatch):
    sample_payload = dict(sample_payload)
    sample_payload["news"] = {"TSLA": None}
    sample_payload["momentum"] = {"TSLA": None}
    sample_payload["10k"] = {"TSLA": None}
    sample_payload["10q"] = {"TSLA": None}
    sample_payload["history_price"] = {"TSLA": None}

    observed = {}

    def fake_recommend_action(payload):
        observed["payload"] = payload
        return "BUY"

    monkeypatch.setattr("api.simple_trading_api.recommend_action", fake_recommend_action)

    response = client.post("/competition_action/", json=sample_payload)

    assert response.status_code == 200
    assert response.json() == {"recommended_action": "BUY"}
    assert observed["payload"]["symbol"] == ["TSLA"]
    assert observed["payload"]["news"] == {"TSLA": None}
    assert observed["payload"]["10k"] == {"TSLA": None}
    assert observed["payload"]["10q"] == {"TSLA": None}


def test_competition_action_accepts_btc_with_null_filings(client, btc_payload, monkeypatch):
    observed = {}

    def fake_recommend_action(payload):
        observed["payload"] = payload
        return "HOLD"

    monkeypatch.setattr("api.simple_trading_api.recommend_action", fake_recommend_action)

    response = client.post("/competition_action/", json=btc_payload)

    assert response.status_code == 200
    assert response.json() == {"recommended_action": "HOLD"}
    assert observed["payload"]["symbol"] == ["BTC"]
    assert observed["payload"]["10k"] is None
    assert observed["payload"]["10q"] is None


def test_competition_action_preserves_multiple_symbols(client, multi_symbol_payload, monkeypatch):
    observed = {}

    def fake_recommend_action(payload):
        observed["payload"] = payload
        return "HOLD"

    monkeypatch.setattr("api.simple_trading_api.recommend_action", fake_recommend_action)

    response = client.post("/competition_action/", json=multi_symbol_payload)

    assert response.status_code == 200
    assert response.json() == {"recommended_action": "HOLD"}
    assert observed["payload"]["symbol"] == ["TSLA", "BTC"]


def test_worker_aggregates_actions_across_multiple_tickers(monkeypatch):
    class FakeDB:
        def get_decision_memory(self, exp_name, ticker, limit):
            return [{"action": {"TSLA": "SELL", "BTC": "HOLD"}[ticker]}]

    monkeypatch.setattr(decision_bridge_worker, "get_db", lambda: FakeDB())

    assert decision_bridge_worker._latest_action("api_endpoint", ["TSLA", "BTC"]) == "HOLD"


def test_worker_preserves_exp_name_from_config(monkeypatch):
    monkeypatch.delenv("DECISION_BRIDGE_EXP_NAME", raising=False)

    cfg = decision_bridge_worker._build_config(
        {
            "date": "2026-03-19",
            "price": {"TSLA": 380.29998779296875},
            "symbol": ["TSLA"],
            "news": {"TSLA": ["Tesla-centred news on 2026-03-19"]},
            "momentum": {"TSLA": "bearish"},
            "10k": {"TSLA": []},
            "10q": {"TSLA": []},
            "history_price": {
                "TSLA": [
                    {"date": "2026-03-18", "price": 392.7799987792969},
                    {"date": "2026-03-19", "price": 380.29998779296875},
                ]
            },
        }
    )

    assert cfg["exp_name"] == "api"


def test_bridge_normalizes_worker_output(monkeypatch, tmp_path):
    # This test checks the orchestration layer in isolation by pretending the
    # worker returned a lowercase action string.
    payload = {"date": "2026-03-19", "symbol": ["TSLA"], "price": {"TSLA": 380.3}}

    class DummyResult:
        returncode = 0
        stdout = json.dumps({"recommended_action": "sell"})
        stderr = ""

    def fake_run(*args, **kwargs):
        return DummyResult()

    monkeypatch.setattr(decision_bridge.subprocess, "run", fake_run)
    monkeypatch.setattr(
        decision_bridge.tempfile,
        "TemporaryDirectory",
        lambda: type(
            "TmpDir",
            (),
            {
                "__enter__": lambda self: str(tmp_path),
                "__exit__": lambda self, exc_type, exc, tb: None,
            },
        )(),
    )

    assert decision_bridge.recommend_action(payload) == "SELL"


def test_bridge_defaults_to_hold_on_worker_failure(monkeypatch, tmp_path):
    payload = {"date": "2026-03-19", "symbol": ["TSLA"], "price": {"TSLA": 380.3}}

    class DummyResult:
        returncode = 1
        stdout = ""
        stderr = "boom"

    def fake_run(*args, **kwargs):
        return DummyResult()

    monkeypatch.setattr(decision_bridge.subprocess, "run", fake_run)
    monkeypatch.setattr(
        decision_bridge.tempfile,
        "TemporaryDirectory",
        lambda: type(
            "TmpDir",
            (),
            {
                "__enter__": lambda self: str(tmp_path),
                "__exit__": lambda self, exc_type, exc, tb: None,
            },
        )(),
    )

    assert decision_bridge.recommend_action(payload) == "HOLD"


def test_bridge_defaults_to_hold_on_timeout(monkeypatch, tmp_path):
    payload = {"date": "2026-03-19", "symbol": ["TSLA"], "price": {"TSLA": 380.3}}

    def fake_run(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd=kwargs.get("args", "cmd"), timeout=170)

    monkeypatch.setattr(decision_bridge.subprocess, "run", fake_run)
    monkeypatch.setattr(
        decision_bridge.tempfile,
        "TemporaryDirectory",
        lambda: type(
            "TmpDir",
            (),
            {
                "__enter__": lambda self: str(tmp_path),
                "__exit__": lambda self, exc_type, exc, tb: None,
            },
        )(),
    )

    assert decision_bridge.recommend_action(payload) == "HOLD"
