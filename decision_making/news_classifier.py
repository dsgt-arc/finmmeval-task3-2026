"""Hybrid news section classifier.

Two-stage approach:
1. **Rule-based pre-filter** — fast keyword matching assigns an item to a
   section when the match is unambiguous.  This avoids burning LLM tokens on
   obvious cases (e.g. ``"10-K"`` → ``filings_10k_10q``).
2. **LLM fallback** — items that survive the rule stage are batched into a
   single LLM call that returns section labels.

The classifier always returns a ``NewsSection`` value for every input item.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

from graph.constants import NewsSection
from util.logger import logger

if TYPE_CHECKING:
    from graph.schema import NewsItem

# ---------------------------------------------------------------------------
# Keyword rules — order matters: first match wins.
# Each entry is (compiled regex, target section).
# ---------------------------------------------------------------------------
_RULES: list[tuple[re.Pattern, NewsSection]] = [
    # Filings
    (re.compile(r"\b10[\-\s]?[KkQq]\b", re.IGNORECASE), NewsSection.FILINGS_10K_10Q),
    (re.compile(r"\bannual report\b", re.IGNORECASE), NewsSection.FILINGS_10K_10Q),
    (re.compile(r"\bquarterly (filing|report)\b", re.IGNORECASE), NewsSection.FILINGS_10K_10Q),
    (re.compile(r"\bSEC filing\b", re.IGNORECASE), NewsSection.FILINGS_10K_10Q),
    # Regulatory / policy
    (re.compile(r"\b(regulat|antitrust|compliance|sanction|tariff|subsid)\w*\b", re.IGNORECASE), NewsSection.REGULATORY_POLICY),
    (re.compile(r"\b(FDA|EPA|FTC|SEC|DOJ)\b"), NewsSection.REGULATORY_POLICY),
    # Macro / rates
    (
        re.compile(r"\b(federal reserve|fed funds|interest rate|inflation|CPI|GDP|unemployment rate)\b", re.IGNORECASE),
        NewsSection.MACRO_RATES,
    ),
    (re.compile(r"\b(treasury yield|bond market|monetary policy|quantitative)\b", re.IGNORECASE), NewsSection.MACRO_RATES),
    # Product / demand
    (
        re.compile(r"\b(product launch|deliveries|sales growth|demand|order backlog|shipment)\b", re.IGNORECASE),
        NewsSection.PRODUCT_DEMAND,
    ),
    (re.compile(r"\b(new model|vehicle|production ramp|recall)\b", re.IGNORECASE), NewsSection.PRODUCT_DEMAND),
    # Industry / competition
    (re.compile(r"\b(competitor|market share|rivalry|industry outlook|peer)\b", re.IGNORECASE), NewsSection.INDUSTRY_COMPETITION),
    # Company fundamentals (broad catch for earnings / revenue)
    (
        re.compile(r"\b(earnings|revenue|profit|EPS|guidance|margin|dividend|buyback)\b", re.IGNORECASE),
        NewsSection.COMPANY_FUNDAMENTALS,
    ),
    (re.compile(r"\b(balance sheet|cash flow|debt|leverage)\b", re.IGNORECASE), NewsSection.COMPANY_FUNDAMENTALS),
]


def classify_by_source(item: NewsItem) -> NewsSection | None:
    """Return a definitive section when the *source* alone is enough."""
    if item.source in ("api_10k", "api_10q"):
        return NewsSection.FILINGS_10K_10Q
    return None


def classify_by_rules(text: str) -> NewsSection | None:
    """Return the first keyword-matched section, or ``None``."""
    for pattern, section in _RULES:
        if pattern.search(text):
            return section
    return None


def classify_items(
    items: list[NewsItem],
    llm_config: dict[str, Any] | None = None,
) -> list[tuple[NewsItem, NewsSection]]:
    """Classify every *NewsItem* into a ``NewsSection``.

    1. Source-based shortcut (10k/10q payloads).
    2. Keyword rules.
    3. LLM classification for remaining items.
    4. Anything still unclassified → ``OTHER``.
    """
    results: list[tuple[NewsItem, NewsSection]] = []
    pending_llm: list[tuple[int, NewsItem]] = []  # (index, item)

    for item in items:
        # source shortcut
        section = classify_by_source(item)
        if section:
            results.append((item, section))
            continue

        # keyword rules
        section = classify_by_rules(item.text)
        if section:
            results.append((item, section))
            continue

        # queue for LLM
        pending_llm.append((len(results), item))
        results.append((item, NewsSection.OTHER))  # placeholder

    # --- LLM classification for ambiguous items ---
    if pending_llm and llm_config:
        llm_labels = _classify_via_llm([item for _, item in pending_llm], llm_config)
        for (idx, _item), label in zip(pending_llm, llm_labels):
            results[idx] = (results[idx][0], label)

    section_counts = {}
    for _, sec in results:
        section_counts[sec.value] = section_counts.get(sec.value, 0) + 1
    logger.info(f"[news_classifier] Section distribution: {section_counts}")

    return results


def _classify_via_llm(items: list[NewsItem], llm_config: dict[str, Any]) -> list[NewsSection]:
    """Use the LLM to classify items that the rule engine could not resolve."""
    from llm.inference import agent_call
    from llm.prompt import NEWS_CLASSIFY_PROMPT

    valid_sections = [s.value for s in NewsSection if s != NewsSection.OTHER]
    numbered = "\n".join(f"{i + 1}. {item.text[:500]}" for i, item in enumerate(items))

    prompt = NEWS_CLASSIFY_PROMPT.format(
        news_items=numbered,
        valid_sections=", ".join(valid_sections),
        count=len(items),
    )

    from pydantic import BaseModel, Field

    class ClassifyResult(BaseModel):
        sections: list[str] = Field(description="One section label per news item, in order")

    result = agent_call(prompt=prompt, llm_config=llm_config, pydantic_model=ClassifyResult)

    labels: list[NewsSection] = []
    section_set = {s.value for s in NewsSection}
    for raw in result.sections:
        cleaned = raw.strip().lower()
        if cleaned in section_set:
            labels.append(NewsSection(cleaned))
        else:
            labels.append(NewsSection.OTHER)

    # pad / truncate to match input length
    while len(labels) < len(items):
        labels.append(NewsSection.OTHER)
    labels = labels[: len(items)]

    return labels
