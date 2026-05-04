from datetime import datetime
import operator
from typing import Annotated, Any

from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from graph.constants import Action, Signal


class AnalystSignal(BaseModel):
    """Signal from analyst"""

    signal: Signal = Field(
        description=f"Choose from {Signal.BULLISH}, {Signal.BEARISH}, or {Signal.NEUTRAL}", default=Signal.NEUTRAL
    )
    justification: str = Field(description="Brief explanation for the signal", default="No justification provided due to error")

    # Per-section fields populated only by the section_news analyst.
    section: str | None = Field(description="News section this signal covers (section_news analyst only)", default=None)
    confidence: float | None = Field(
        description="Model confidence in [0, 1] (section_news analyst only)", default=None, ge=0.0, le=1.0
    )
    horizon: str | None = Field(description="Investment horizon: short / medium / long (section_news analyst only)", default=None)

    # Enhanced sentiment fields (optional for backward compatibility)
    signal_strength: float | None = Field(
        description="Sentiment strength from -1.0 (strong bearish) to +1.0 (strong bullish)", default=None, ge=-1.0, le=1.0
    )


class RelevanceCheck(BaseModel):
    """Check if content is relevant for next-day price prediction"""

    is_relevant: bool = Field(description="True if relevant for next-day price movement")
    reasoning: str = Field(description="Brief explanation", default="")


class Decision(BaseModel):
    """Decision made by portfolio manager"""

    action: Action = Field(description=f"Choose from {Action.BUY}, {Action.SELL}, or {Action.HOLD}", default=Action.HOLD)
    shares: int = Field(description="Number of shares to buy or sell, set 0 for hold", default=0)
    price: float = Field(description="Current price for the ticker", default=0)
    justification: str = Field(description="Brief explanation for the decision", default="Just hold due to error")


class Position(BaseModel):
    """Position for a single ticker"""

    value: float = Field(default=0.0, description="Monetary value for the position.")
    shares: int = Field(default=0, description="Shares for the position.")


class Portfolio(BaseModel):
    """Portfolio state — kept for DB stub creation in workflow."""

    id: str = Field(description="Portfolio id.")
    cashflow: float = Field(description="Cashflow for the fund.")
    positions: dict[str, Position] = Field(description="Positions for each ticker.")


class FundState(TypedDict):
    """Fund state when running the workflow."""

    # from environment
    exp_name: str = Field(description="Experiment name.")
    trading_date: datetime = Field(description="Trading date.")
    ticker: str = Field(description="Ticker in-the-flow.")
    llm_config: dict[str, Any] = Field(description="LLM configuration.")
    portfolio_id: str = Field(description="Portfolio stub id (FK anchor for DB logging).")

    # raw competition API payload (or None during backtests); analysts pull what they need
    api_payload: dict | None

    # updated by workflow
    # ticker -> signal of all analysts
    analyst_signals: Annotated[list[AnalystSignal], operator.add]
    # portfolio manager output
    decision: Decision
