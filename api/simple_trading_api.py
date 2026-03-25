"""Competition-facing FastAPI wrapper.

Separation of concerns:
- Validates the organizer payload and exposes the HTTP endpoints.
- Does not contain trading logic.
- Calls `api.decision_bridge.recommend_action(payload)` and returns the
  exact competition response shape.
- Keeps the transport layer small so the deployment is easy to reason about.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from api.decision_bridge import recommend_action

load_dotenv()

app = FastAPI(title="Simple Trading API", version="5.0.0")

VALID_ACTIONS = {"BUY", "HOLD", "SELL"}


class HistoricalPrice(BaseModel):
    date: str
    price: float


class TradingRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    # This is the organizer payload. Some fields are optional because the
    # competition text says news and filings may be absent for some assets.
    date: str
    price: Dict[str, float]
    news: Optional[Dict[str, List[str]]] = None
    symbol: Optional[List[str]] = None
    momentum: Dict[str, str] = Field(default_factory=dict)
    ten_k: Optional[Dict[str, List[str]]] = Field(default=None, alias="10k")
    ten_q: Optional[Dict[str, List[str]]] = Field(default=None, alias="10q")
    history_price: Dict[str, List[HistoricalPrice]] = Field(default_factory=dict)


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
    payload = request.model_dump(by_alias=True, exclude_none=True)
    if not payload.get("symbol"):
        # If the organizers omit symbol, infer it from the price map.
        payload["symbol"] = [next(iter(payload["price"]))]
    return payload


@app.get("/")
async def home():
    return {"message": "Simple Trading API is running"}


@app.get("/health")
async def health():
    return {"status": "healthy"}


def _recommend_action_payload(request: TradingRequest) -> dict:
    payload = _payload_for_bridge(request)
    # The bridge returns the action from the existing workflow.
    action = recommend_action(payload)
    return {"recommended_action": _normalize_action(action)}


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
