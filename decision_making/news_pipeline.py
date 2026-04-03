"""News ingestion layer.

Merges news from three sources in priority order:
1. API payload fields (news, 10k, 10q) — primary during live competition.
2. AMA parquet data — fallback for backtests when no payload is available.

Each source is normalised into a list of ``NewsItem`` objects so downstream
section classification works identically regardless of origin.
"""

from __future__ import annotations

import datetime
from typing import Any

from graph.schema import NewsItem
from util.logger import logger


def ingest_news(
    ticker: str,
    trading_date: datetime.datetime,
    payload: dict[str, Any] | None = None,
) -> list[NewsItem]:
    """Return a unified list of ``NewsItem`` objects for *ticker* on *trading_date*.

    Parameters
    ----------
    ticker:
        The asset symbol (e.g. ``"TSLA"``).
    trading_date:
        The current trading date.  AMA fallback fetches t-1 data.
    payload:
        The raw competition API payload (may be ``None`` during backtests).
    """
    items: list[NewsItem] = []

    # --- 1. API payload (primary) -------------------------------------------
    if payload:
        items.extend(_extract_payload_news(ticker, payload))

    # --- 2. AMA parquet fallback ---------------------------------------------
    if not items:
        items.extend(_extract_ama_news(ticker, trading_date))

    logger.info(f"[news_pipeline] Ingested {len(items)} news items for {ticker}")
    return items


def _extract_payload_news(ticker: str, payload: dict[str, Any]) -> list[NewsItem]:
    """Extract news items from the competition API payload."""
    items: list[NewsItem] = []
    date_str = payload.get("date")

    # news field: dict[symbol, list[str]]
    news_map = payload.get("news") or {}
    for text in news_map.get(ticker, []):
        if text and text.strip():
            items.append(NewsItem(text=text.strip(), source="api_news", date=date_str))

    # 10-K filings
    tenk_map = payload.get("10k") or {}
    for text in tenk_map.get(ticker, []):
        if text and text.strip():
            items.append(NewsItem(text=text.strip(), source="api_10k", date=date_str))

    # 10-Q filings
    tenq_map = payload.get("10q") or {}
    for text in tenq_map.get(ticker, []):
        if text and text.strip():
            items.append(NewsItem(text=text.strip(), source="api_10q", date=date_str))

    return items


def _extract_ama_news(ticker: str, trading_date: datetime.datetime) -> list[NewsItem]:
    """Fallback: load t-1 news from the AMA parquet dataset."""
    try:
        from decision_making.ama_data import load_specific_data

        historic_date = trading_date - datetime.timedelta(days=1)
        raw_news = load_specific_data(symbol=ticker, date=historic_date, type="news")
        if raw_news and str(raw_news).strip():
            return [
                NewsItem(
                    text=str(raw_news).strip(),
                    source="ama",
                    date=historic_date.isoformat() if hasattr(historic_date, "isoformat") else str(historic_date),
                )
            ]
    except Exception as e:
        logger.warning(f"[news_pipeline] AMA fallback failed for {ticker}: {e}")

    return []
