"""Section-aware news analyst.

Replaces the monolithic ``company_news_agent`` with a multi-section pipeline:

1. Reads ``news_items`` that were ingested earlier (by the workflow) from the
   API payload or AMA fallback.
2. Classifies each item into a ``NewsSection`` (rules first, LLM fallback).
3. Scores each non-empty section via a dedicated LLM prompt.
4. Fuses the per-section ``SectionSignal`` objects into one final
   ``AnalystSignal`` using a weighted aggregation.

The analyst outputs **both** the fused ``AnalystSignal`` (so the PM sees a
single signal, same as ``company_news``) **and** the per-section breakdown
(``section_signals``) for richer risk/PM prompts.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from graph.constants import AgentKey, NewsSection, Signal
from graph.schema import AnalystSignal, FundState, NewsItem, SectionSignal
from llm.cost_estimation import estimate_cost
from llm.inference import agent_call
from llm.prompt import SECTION_NEWS_AGGREGATE_PROMPT, SECTION_SCORE_PROMPT
from news_classifier import classify_items
from util.db_helper import get_db
from util.logger import logger

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
    news_items: list[NewsItem] = state.get("news_items") or []

    db = get_db()

    logger.log_agent_status(agent_name, ticker, "Classifying news into sections")

    if not news_items:
        logger.warning(f"[{agent_name}] No news items for {ticker}, returning neutral")
        neutral = AnalystSignal(signal=Signal.NEUTRAL, justification="No news available for analysis")
        return {"analyst_signals": [neutral], "section_signals": []}

    # --- 1. Classify ---
    classified = classify_items(news_items, llm_config)

    # Group items by section
    section_items: dict[NewsSection, list[NewsItem]] = defaultdict(list)
    for item, section in classified:
        section_items[section].append(item)

    # --- 2. Score each section ---
    section_signals: list[SectionSignal] = []
    for section, items in section_items.items():
        news_text = "\n---\n".join(item.text[:1000] for item in items)
        prompt = SECTION_SCORE_PROMPT.format(section=section.value, news_text=news_text)

        costs = estimate_cost(prompt, llm_config)
        logger.info(f"[{agent_name}] Section {section.value} ({len(items)} items) est. cost: ${costs:.6f}")

        sig = agent_call(prompt=prompt, llm_config=llm_config, pydantic_model=SectionSignal)
        sig.section = section.value
        section_signals.append(sig)

        logger.info(f"[{agent_name}] {section.value}: {sig.direction} (conf={sig.confidence:.2f})")

    # --- 3. Fuse ---
    fused_signal = _fuse_section_signals(section_signals, news_items, llm_config)

    # --- 4. Persist ---
    logger.log_signal(agent_name, ticker, fused_signal)
    db.save_signal(portfolio_id, agent_name, ticker, "section_news_fused", fused_signal)

    # Save section-level signals if the DB supports it
    for ss in section_signals:
        _save_section_signal(db, portfolio_id, ticker, ss)

    return {"analyst_signals": [fused_signal], "section_signals": section_signals}


def _fuse_section_signals(
    section_signals: list[SectionSignal],
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
        w = sig.confidence * avg_source_weight
        total_score += _DIRECTION_SIGN.get(sig.direction, 0.0) * w
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
            f"- {sig.section}: {sig.direction} (confidence={sig.confidence:.2f}, horizon={sig.horizon}) — {sig.rationale}"
        )
    breakdown_text = "\n".join(breakdown_lines)

    prompt = SECTION_NEWS_AGGREGATE_PROMPT.format(section_breakdown=breakdown_text)
    agg_signal = agent_call(prompt=prompt, llm_config=llm_config, pydantic_model=AnalystSignal)

    # Override direction with the numeric fusion to keep consistency
    agg_signal.signal = fused_direction
    return agg_signal


def _save_section_signal(db, portfolio_id: str, ticker: str, sig: SectionSignal):
    """Persist a section-level signal using the extended signal table."""
    try:
        db.save_section_signal(portfolio_id, ticker, sig)
    except AttributeError:
        # DB implementation doesn't support section signals yet — silently skip
        pass
    except Exception as e:
        logger.warning(f"Failed to persist section signal: {e}")
