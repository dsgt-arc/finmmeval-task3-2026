"""Section-aware news analyst.

Replaces the monolithic ``company_news_agent`` with a multi-section pipeline:

1. Fetches its own news/filing items via ``news_pipeline.ingest_news`` using
   the API payload (or AMA fallback) stored in ``FundState``.
2. Classifies each item into a ``NewsSection`` (rules first, LLM fallback).
3. Scores each non-empty section via a dedicated LLM prompt, producing an
   ``AnalystSignal`` with the extra ``section``, ``confidence``, and
   ``horizon`` fields populated.
4. Fuses the per-section signals into one final ``AnalystSignal`` using
   weighted aggregation and an LLM-generated justification.

The fused signal is returned in ``analyst_signals`` so the PM sees exactly
one signal per analyst, consistent with all other analysts.  Per-section
signals are persisted to the DB under namespaced analyst keys
(``section_news:<section>``) but are not threaded through graph state.
"""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING, Any

from graph.constants import AgentKey, NewsSection, Signal
from graph.schema import AnalystSignal, FundState
from llm.cost_estimation import estimate_cost
from llm.inference import agent_call
from llm.prompt import SECTION_NEWS_AGGREGATE_PROMPT, SECTION_SCORE_PROMPT
from news_classifier import classify_items
from news_pipeline import ingest_news
from pydantic import BaseModel, Field
from util.db_helper import get_db
from util.logger import logger

if TYPE_CHECKING:
    from news_pipeline import NewsItem


class _SectionScore(BaseModel):
    """LLM-parsed output for a single news section (local to this analyst)."""

    direction: Signal = Field(default=Signal.NEUTRAL, description="Bullish / Bearish / Neutral")
    confidence: float = Field(default=0.5, ge=0.0, le=1.0, description="Model confidence in [0, 1]")
    horizon: str = Field(default="short", description="Investment horizon: short / medium / long")
    rationale: str = Field(default="No rationale provided", description="Brief explanation")


# Source-level weights for the fusion step.
_SOURCE_WEIGHT: dict[str, float] = {
    "api_news": 1.0,
    "api_10k": 0.8,
    "api_10q": 0.8,
    "ama": 0.9,
}

# Direction → numeric sign for aggregation.
_DIRECTION_SIGN: dict[Signal, float] = {
    Signal.BULLISH: 1.0,
    Signal.BEARISH: -1.0,
    Signal.NEUTRAL: 0.0,
}


def section_news_agent(state: FundState):
    """News orchestrator: classify → score per section → fuse → return."""
    agent_name = AgentKey.SECTION_NEWS
    ticker = state["ticker"]
    llm_config = state["llm_config"]
    portfolio_id = state["portfolio"].id

    db = get_db()

    logger.log_agent_status(agent_name, ticker, "Classifying news into sections")

    news_items: list[NewsItem] = ingest_news(ticker, state["trading_date"], state.get("api_payload"))

    if not news_items:
        logger.warning(f"[{agent_name}] No news items for {ticker}, returning neutral")
        neutral = AnalystSignal(signal=Signal.NEUTRAL, justification="No news available for analysis")
        return {"analyst_signals": [neutral]}

    # --- 1. Classify ---
    classified = classify_items(news_items, llm_config)

    # Group items by section
    section_items: dict[NewsSection, list[NewsItem]] = defaultdict(list)
    for item, section in classified:
        section_items[section].append(item)

    # --- 2. Score each section ---
    section_signals: list[AnalystSignal] = []
    for section, items in section_items.items():
        news_text = "\n---\n".join(item.text[:1000] for item in items)
        prompt = SECTION_SCORE_PROMPT.format(section=section.value, news_text=news_text)

        costs = estimate_cost(prompt, llm_config)
        logger.info(f"[{agent_name}] Section {section.value} ({len(items)} items) est. cost: ${costs:.6f}")

        score = agent_call(prompt=prompt, llm_config=llm_config, pydantic_model=_SectionScore)
        sig = AnalystSignal(
            signal=score.direction,
            justification=score.rationale,
            section=section.value,
            confidence=score.confidence,
            horizon=score.horizon,
        )
        section_signals.append(sig)

        logger.info(f"[{agent_name}] {section.value}: {sig.signal} (conf={sig.confidence:.2f})")

    # --- 3. Fuse ---
    fused_signal = _fuse_section_signals(section_signals, news_items, llm_config)

    # --- 4. Persist ---
    logger.log_signal(agent_name, ticker, fused_signal)
    db.save_signal(portfolio_id, agent_name, ticker, "section_news_fused", fused_signal)

    # Save per-section signals into the shared signal table.
    # analyst is namespaced as "section_news:<section>" to distinguish from the
    # fused signal (analyst="section_news_fused") and other analysts.
    for ss in section_signals:
        db.save_signal(portfolio_id, f"section_news:{ss.section}", ticker, "", ss)

    return {"analyst_signals": [fused_signal]}


def _fuse_section_signals(
    section_signals: list[AnalystSignal],
    news_items: list[NewsItem],
    llm_config: dict[str, Any],
) -> AnalystSignal:
    """Weighted fusion of per-section signals into one AnalystSignal.

    score = sum( sign(direction) * confidence * source_weight ) / N
    Then use the LLM to produce the final justification.
    """
    if not section_signals:
        return AnalystSignal(signal=Signal.NEUTRAL, justification="No section signals to fuse")

    # Compute average source weight across all items for a rough multiplier
    avg_source_weight = 1.0
    if news_items:
        weights = [_SOURCE_WEIGHT.get(item.source, 0.9) for item in news_items]
        avg_source_weight = sum(weights) / len(weights)

    total_score = 0.0
    total_weight = 0.0
    for sig in section_signals:
        w = (sig.confidence or 0.5) * avg_source_weight
        total_score += _DIRECTION_SIGN.get(sig.signal, 0.0) * w
        total_weight += w

    normalised = total_score / total_weight if total_weight > 0 else 0.0

    if normalised > 0.15:
        fused_direction = Signal.BULLISH
    elif normalised < -0.15:
        fused_direction = Signal.BEARISH
    else:
        fused_direction = Signal.NEUTRAL

    # Build section breakdown for the aggregation prompt
    breakdown_lines = []
    for sig in section_signals:
        breakdown_lines.append(
            f"- {sig.section}: {sig.signal} (confidence={sig.confidence:.2f}, horizon={sig.horizon}) — {sig.justification}"
        )
    breakdown_text = "\n".join(breakdown_lines)

    prompt = SECTION_NEWS_AGGREGATE_PROMPT.format(section_breakdown=breakdown_text)
    agg_signal = agent_call(prompt=prompt, llm_config=llm_config, pydantic_model=AnalystSignal)

    # Override direction with the numeric fusion to keep consistency
    agg_signal.signal = fused_direction
    return agg_signal
