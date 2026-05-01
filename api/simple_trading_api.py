"""Competition-facing FastAPI wrapper.

Separation of concerns:
- Validates the organizer payload and exposes the HTTP endpoints.
- Does not contain trading logic.
- Calls `api.decision_bridge.recommend_action(payload)` and returns the
  exact competition response shape for the Task 3 signal-only endpoint.
- Keeps the transport layer small so the deployment is easy to reason about.
"""

from __future__ import annotations

import inspect
import os
import uuid
import logging
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from api.decision_bridge import recommend_action

load_dotenv()

app = FastAPI(title="Simple Trading API", version="5.0.0")
logger = logging.getLogger(__name__)

VALID_ACTIONS = {"BUY", "HOLD", "SELL"}


class HistoricalPrice(BaseModel):
    date: str
    price: float


class TradingRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    # This is the organizer payload. Some fields are optional because the
    # competition text says news and filings may be absent for some assets.
    # We also allow per-symbol null values so sparse days do not break parsing.
    date: str
    price: Dict[str, float]
    news: Optional[Dict[str, List[str] | None]] = None
    symbol: Optional[List[str]] = None
    momentum: Optional[Dict[str, str | None]] = None
    ten_k: Optional[Dict[str, List[str] | None]] = Field(default=None, alias="10k")
    ten_q: Optional[Dict[str, List[str] | None]] = Field(default=None, alias="10q")
    history_price: Dict[str, List[HistoricalPrice] | None] = Field(default_factory=dict)


class TradingResponse(BaseModel):
    recommended_action: str


def _normalize_action(action: str) -> str:
    normalized = action.strip().upper()
    if normalized not in VALID_ACTIONS:
        raise HTTPException(status_code=500, detail=f"Invalid action returned by decision bridge: {action!r}")
    return normalized


def _payload_for_bridge(request: TradingRequest) -> Dict[str, Any]:
    # Convert the validated Pydantic model back into a plain dict so the bridge
    # can pass it into the existing decision-making code without API-specific
    # types leaking across the boundary.
    payload = request.model_dump(by_alias=True)
    if not payload.get("symbol"):
        # If the organizers omit symbol, infer it from the price map.
        payload["symbol"] = list(payload["price"].keys())
    return payload


def _request_summary(request: TradingRequest) -> dict[str, Any]:
    symbols = request.symbol or list(request.price.keys())
    return {
        "date": request.date,
        "symbols": symbols,
        "price_count": len(request.price),
        "news_count": 0 if request.news is None else len(request.news),
        "ten_k_count": 0 if request.ten_k is None else len(request.ten_k),
        "ten_q_count": 0 if request.ten_q is None else len(request.ten_q),
    }


def _call_recommend_action(payload: Dict[str, Any], request_id: str) -> str:
    """Call the bridge while preserving compatibility with monkeypatched tests."""
    signature = inspect.signature(recommend_action)
    accepts_request_id = "request_id" in signature.parameters
    accepts_var_keyword = any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values())

    if accepts_request_id or accepts_var_keyword:
        return recommend_action(payload, request_id=request_id)
    return recommend_action(payload)


@app.get("/")
async def home():
    return {"message": "Simple Trading API is running"}


@app.get("/health")
async def health():
    return {"status": "healthy"}


def _recommend_action_payload(request: TradingRequest) -> dict:
    request_id = uuid.uuid4().hex[:10]
    payload = _payload_for_bridge(request)
    logger.info("Request received request_id=%s summary=%s", request_id, _request_summary(request))
    # The bridge returns the final 3-way signal from the existing workflow.
    action = _call_recommend_action(payload, request_id)
    normalized = _normalize_action(action)
    logger.info("Request completed request_id=%s action=%s", request_id, normalized)
    return {"recommended_action": normalized}


@app.post("/competition_action/", response_model=TradingResponse)
async def competition_action(request: TradingRequest):
    return _recommend_action_payload(request)


@app.post("/trading_action/", response_model=TradingResponse)
async def trading_action(request: TradingRequest):
    return _recommend_action_payload(request)


if __name__ == "__main__":
    import uvicorn

    print("Starting Simple Trading API...")
    print("Endpoints:")
    print("  POST /competition_action/")
    print("  POST /trading_action/ (legacy alias)")
    print("Health:")
    print("  GET /health")
    port = int(os.getenv("PORT", "62237"))
    print(f"\nAPI will be available at: http://0.0.0.0:{port}")

    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
