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
