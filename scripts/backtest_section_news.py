"""End-to-end backtest comparing company_news (baseline) vs section_news analyst.

This script runs a deterministic backtest over the competition parquet data
without requiring LLM API keys.  It tests the full pipeline:

    news ingestion -> classification -> section scoring -> signal fusion -> decision -> P&L

The "LLM" calls are replaced by deterministic keyword-sentiment scoring so the
pipeline logic, classifier, aggregator and portfolio math are all exercised on
real data.

Usage:
    uv run python scripts/backtest_section_news.py
    uv run python scripts/backtest_section_news.py --ticker TSLA --days 60
    uv run python scripts/backtest_section_news.py --save results/
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime
import json
import math
from pathlib import Path
import sys

import numpy as np
import polars as pl  # noqa: TC002

# ---------------------------------------------------------------------------
# Bootstrap imports
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
DECISION_MAKING_DIR = REPO_ROOT / "decision_making"
for p in (str(DECISION_MAKING_DIR), str(REPO_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from decision_making.ama_data import load_data  # noqa: E402
from decision_making.graph.constants import NewsSection, Signal  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BULLISH_KEYWORDS = [
    "surge",
    "soar",
    "rally",
    "record high",
    "beat expectations",
    "strong demand",
    "growth",
    "upgrade",
    "outperform",
    "profit",
    "positive",
    "bullish",
    "buy",
    "breakthrough",
    "innovation",
    "expansion",
    "deliveries up",
    "sales growth",
    "beat estimates",
    "raised guidance",
    "exceeded",
    "all-time high",
]

BEARISH_KEYWORDS = [
    "crash",
    "plunge",
    "plummet",
    "miss",
    "downgrade",
    "sell-off",
    "loss",
    "decline",
    "bearish",
    "sell",
    "recall",
    "investigation",
    "lawsuit",
    "fine",
    "penalty",
    "weak demand",
    "layoff",
    "restructuring",
    "bankruptcy",
    "default",
    "warning",
    "cut guidance",
    "missed estimates",
    "liability",
    "verdict",
    "damages",
    "negative",
    "risk",
    "concern",
    "fell",
    "drop",
]


def keyword_sentiment(text: str) -> tuple[Signal, float]:
    """Score text sentiment using keyword matching. Returns (signal, confidence)."""
    text_lower = text.lower()
    bull_count = sum(1 for kw in BULLISH_KEYWORDS if kw in text_lower)
    bear_count = sum(1 for kw in BEARISH_KEYWORDS if kw in text_lower)
    total = bull_count + bear_count
    if total == 0:
        return Signal.NEUTRAL, 0.3

    score = (bull_count - bear_count) / total
    confidence = min(0.9, 0.3 + 0.4 * (total / 10))

    if score > 0.15:
        return Signal.BULLISH, confidence
    elif score < -0.15:
        return Signal.BEARISH, confidence
    else:
        return Signal.NEUTRAL, confidence


# ---------------------------------------------------------------------------
# News classifier (reuses the rules from news_classifier.py)
# ---------------------------------------------------------------------------

from graph.schema import NewsItem  # noqa: E402
from news_classifier import classify_by_rules, classify_by_source  # noqa: E402


def classify_news_item(item: NewsItem) -> NewsSection:
    """Classify a single news item into a section (rules only, no LLM)."""
    section = classify_by_source(item)
    if section:
        return section
    section = classify_by_rules(item.text)
    if section:
        return section
    return NewsSection.OTHER


# ---------------------------------------------------------------------------
# Baseline: single-stream company_news analyst (deterministic)
# ---------------------------------------------------------------------------


def baseline_company_news_signal(news_text: str) -> tuple[Signal, str]:
    """Mimic company_news_agent: one signal from all news text."""
    signal, confidence = keyword_sentiment(news_text)
    justification = f"Keyword sentiment: {signal.value} (conf={confidence:.2f})"
    return signal, justification


# ---------------------------------------------------------------------------
# Section news analyst (deterministic)
# ---------------------------------------------------------------------------


def section_news_signal(items: list[NewsItem]) -> tuple[Signal, str, list[dict]]:
    """Mimic section_news_agent: classify, score per section, fuse."""
    if not items:
        return Signal.NEUTRAL, "No news available", []

    # classify
    section_map: dict[NewsSection, list[NewsItem]] = defaultdict(list)
    for item in items:
        sec = classify_news_item(item)
        section_map[sec].append(item)

    # score each section
    section_signals = []
    for sec, sec_items in section_map.items():
        combined_text = " ".join(it.text[:1000] for it in sec_items)
        direction, confidence = keyword_sentiment(combined_text)
        section_signals.append({
            "section": sec.value,
            "direction": direction,
            "confidence": confidence,
            "item_count": len(sec_items),
        })

    # Source weights
    source_weight = {"api_news": 1.0, "api_10k": 0.8, "api_10q": 0.8, "ama": 0.9}
    avg_sw = np.mean([source_weight.get(it.source, 0.9) for it in items])

    # Fuse
    dir_sign = {Signal.BULLISH: 1.0, Signal.BEARISH: -1.0, Signal.NEUTRAL: 0.0}
    total_score = 0.0
    total_weight = 0.0
    for ss in section_signals:
        w = ss["confidence"] * avg_sw
        total_score += dir_sign[ss["direction"]] * w
        total_weight += w

    normalised = total_score / total_weight if total_weight > 0 else 0.0

    if normalised > 0.10:
        fused = Signal.BULLISH
    elif normalised < -0.10:
        fused = Signal.BEARISH
    else:
        fused = Signal.NEUTRAL

    breakdown = ", ".join(f"{s['section']}={s['direction'].value}" for s in section_signals)
    justification = f"Section fusion: {fused.value} (norm={normalised:.3f}) | {breakdown}"
    return fused, justification, section_signals


# ---------------------------------------------------------------------------
# Technical signal (simplified, reuses actual functions)
# ---------------------------------------------------------------------------


def simple_technical_signal(prices: list[float]) -> Signal:
    """Minimal EMA-based trend signal."""
    if len(prices) < 21:
        return Signal.NEUTRAL

    def ema(data, span):
        alpha = 2 / (span + 1)
        result = [data[0]]
        for val in data[1:]:
            result.append(alpha * val + (1 - alpha) * result[-1])
        return result

    ema_short = ema(prices, 8)
    ema_med = ema(prices, 21)

    if ema_short[-1] > ema_med[-1]:
        return Signal.BULLISH
    elif ema_short[-1] < ema_med[-1]:
        return Signal.BEARISH
    return Signal.NEUTRAL


# ---------------------------------------------------------------------------
# Trading decision
# ---------------------------------------------------------------------------


def make_decision(
    technical_signal: Signal,
    news_signal: Signal,
    current_price: float,
    current_shares: int,
    cashflow: float,
    position_limit_ratio: float = 0.5,
    total_value: float = 100000.0,
) -> tuple[str, int]:
    """Simple decision: majority vote of technical + news signals, buy/sell up to limit."""
    sig_map = {Signal.BULLISH: 1, Signal.BEARISH: -1, Signal.NEUTRAL: 0}
    combined = sig_map[technical_signal] + sig_map[news_signal]

    position_limit = total_value * position_limit_ratio
    current_value = current_shares * current_price

    if combined > 0:
        room = min(position_limit - current_value, cashflow)
        shares_to_buy = max(0, int(room // current_price))
        if shares_to_buy > 0:
            return "BUY", shares_to_buy
    elif combined < 0:
        shares_to_sell = min(current_shares, max(1, current_shares // 2))
        if shares_to_sell > 0:
            return "SELL", shares_to_sell

    return "HOLD", 0


# ---------------------------------------------------------------------------
# Portfolio tracker
# ---------------------------------------------------------------------------


class PortfolioTracker:
    def __init__(self, initial_cash: float = 100000.0):
        self.cashflow = initial_cash
        self.initial_cash = initial_cash
        self.positions: dict[str, int] = {}  # ticker -> shares
        self.history: list[dict] = []
        self.decisions: list[dict] = []

    def execute(self, ticker: str, action: str, shares: int, price: float, date: str, justification: str = ""):
        if action == "BUY" and shares > 0:
            cost = shares * price
            if cost <= self.cashflow:
                self.cashflow -= cost
                self.positions[ticker] = self.positions.get(ticker, 0) + shares
        elif action == "SELL" and shares > 0:
            held = self.positions.get(ticker, 0)
            actual_sell = min(shares, held)
            if actual_sell > 0:
                self.cashflow += actual_sell * price
                self.positions[ticker] = held - actual_sell

        self.decisions.append({
            "date": date,
            "ticker": ticker,
            "action": action,
            "shares": shares,
            "price": price,
            "justification": justification,
        })

    def snapshot(self, prices: dict[str, float], date: str):
        total = self.cashflow
        for ticker, shares in self.positions.items():
            total += shares * prices.get(ticker, 0)
        self.history.append({
            "date": date,
            "total_assets": total,
            "cashflow": self.cashflow,
            "positions": dict(self.positions),
        })
        return total

    def total_return_pct(self):
        if not self.history:
            return 0.0
        return (self.history[-1]["total_assets"] / self.initial_cash - 1) * 100

    def daily_returns(self):
        values = [h["total_assets"] for h in self.history]
        if len(values) < 2:
            return []
        return [(values[i] / values[i - 1] - 1) for i in range(1, len(values))]

    def sharpe_ratio(self, annualisation=252):
        rets = self.daily_returns()
        if len(rets) < 2:
            return 0.0
        mean_r = np.mean(rets)
        std_r = np.std(rets)
        if std_r == 0:
            return 0.0
        return mean_r / std_r * math.sqrt(annualisation)

    def max_drawdown_pct(self):
        values = [h["total_assets"] for h in self.history]
        if not values:
            return 0.0
        running_max = np.maximum.accumulate(values)
        drawdowns = (np.array(values) - running_max) / running_max * 100
        return float(np.min(drawdowns))

    def hit_rate(self):
        """Fraction of BUY/SELL decisions followed by favourable next-day move."""
        trades = [d for d in self.decisions if d["action"] in ("BUY", "SELL")]
        if not trades:
            return 0.0
        correct = 0
        for trade in trades:
            # find next day price for same ticker
            future = [d for d in self.decisions[self.decisions.index(trade) + 1 :] if d["ticker"] == trade["ticker"]]
            if not future:
                continue
            next_price = future[0]["price"]
            if (trade["action"] == "BUY" and next_price > trade["price"]) or (
                trade["action"] == "SELL" and next_price < trade["price"]
            ):
                correct += 1
        return correct / len(trades) if trades else 0.0

    def action_counts(self):
        counts = {"BUY": 0, "SELL": 0, "HOLD": 0}
        for d in self.decisions:
            counts[d["action"]] = counts.get(d["action"], 0) + 1
        return counts


# ---------------------------------------------------------------------------
# Backtest runner
# ---------------------------------------------------------------------------


def load_competition_data(ticker: str) -> pl.DataFrame:
    """Load competition parquet and return sorted by date."""
    df = load_data(ticker, competition_data=True)
    df = df.sort("date")
    return df


def build_payload_from_row(row: dict, ticker: str) -> dict:
    """Build a competition-style API payload from a parquet row."""
    payload = {
        "date": str(row["date"]),
        "price": {ticker: row["prices"]},
        "symbol": [ticker],
        "momentum": {ticker: row.get("momentum", "")},
    }

    # news
    news_val = row.get("news")
    if news_val:
        if isinstance(news_val, list):
            payload["news"] = {ticker: news_val}
        elif isinstance(news_val, str) and news_val.strip():
            payload["news"] = {ticker: [news_val]}

    # 10k
    tenk_val = row.get("10k")
    if tenk_val and isinstance(tenk_val, list) and any(tenk_val):
        payload["10k"] = {ticker: tenk_val}

    # 10q
    tenq_val = row.get("10q")
    if tenq_val and isinstance(tenq_val, list) and any(tenq_val):
        payload["10q"] = {ticker: tenq_val}

    return payload


def items_from_payload(payload: dict, ticker: str) -> list[NewsItem]:
    """Extract NewsItem objects from a competition payload."""
    items = []
    date_str = payload.get("date")

    for text in (payload.get("news") or {}).get(ticker, []):
        if text and str(text).strip():
            items.append(NewsItem(text=str(text).strip()[:3000], source="api_news", date=date_str))

    for text in (payload.get("10k") or {}).get(ticker, []):
        if text and str(text).strip():
            items.append(NewsItem(text=str(text).strip()[:3000], source="api_10k", date=date_str))

    for text in (payload.get("10q") or {}).get(ticker, []):
        if text and str(text).strip():
            items.append(NewsItem(text=str(text).strip()[:3000], source="api_10q", date=date_str))

    return items


def run_backtest(ticker: str, max_days: int | None = None) -> dict:
    """Run one ticker backtest, returning metrics for both strategies."""
    df = load_competition_data(ticker)
    rows = df.to_dicts()
    if max_days:
        rows = rows[:max_days]

    # Two portfolios — identical starting conditions
    pf_baseline = PortfolioTracker(100000.0)
    pf_section = PortfolioTracker(100000.0)

    price_history: list[float] = []
    section_distribution: dict[str, int] = defaultdict(int)

    print(f"\n{'=' * 70}")
    print(f"  BACKTEST: {ticker}  |  {len(rows)} trading days")
    print(f"  Date range: {rows[0]['date']} to {rows[-1]['date']}")
    print(f"{'=' * 70}")

    for i, row in enumerate(rows):
        date_str = str(row["date"])
        price = row["prices"]
        price_history.append(price)

        # Build payload
        payload = build_payload_from_row(row, ticker)
        news_items = items_from_payload(payload, ticker)

        # Technical signal (shared)
        tech_signal = simple_technical_signal(price_history)

        # --- Baseline: single-stream company_news ---
        all_text = " ".join(it.text for it in news_items) if news_items else ""
        baseline_signal, baseline_just = baseline_company_news_signal(all_text)

        # --- Section news ---
        section_signal, section_just, sec_details = section_news_signal(news_items)
        for sd in sec_details:
            section_distribution[sd["section"]] += sd["item_count"]

        # Shared portfolio info
        prices_map = {ticker: price}

        # Baseline decision
        total_b = pf_baseline.cashflow + pf_baseline.positions.get(ticker, 0) * price
        action_b, shares_b = make_decision(
            tech_signal,
            baseline_signal,
            price,
            pf_baseline.positions.get(ticker, 0),
            pf_baseline.cashflow,
            total_value=total_b,
        )
        pf_baseline.execute(ticker, action_b, shares_b, price, date_str, baseline_just)
        pf_baseline.snapshot(prices_map, date_str)

        # Section decision
        total_s = pf_section.cashflow + pf_section.positions.get(ticker, 0) * price
        action_s, shares_s = make_decision(
            tech_signal,
            section_signal,
            price,
            pf_section.positions.get(ticker, 0),
            pf_section.cashflow,
            total_value=total_s,
        )
        pf_section.execute(ticker, action_s, shares_s, price, date_str, section_just)
        pf_section.snapshot(prices_map, date_str)

        # Progress
        if (i + 1) % 50 == 0 or i == len(rows) - 1:
            print(
                f"  Day {i + 1:3d}/{len(rows)} | Price: ${price:,.2f} | "
                f"Baseline: ${pf_baseline.history[-1]['total_assets']:,.2f} | "
                f"Section: ${pf_section.history[-1]['total_assets']:,.2f}"
            )

    # Ground truth evaluation (if future_price_diff is available)
    direction_accuracy_baseline = _direction_accuracy(pf_baseline.decisions, rows, ticker)
    direction_accuracy_section = _direction_accuracy(pf_section.decisions, rows, ticker)

    return {
        "ticker": ticker,
        "days": len(rows),
        "date_start": str(rows[0]["date"]),
        "date_end": str(rows[-1]["date"]),
        "baseline": _portfolio_metrics(pf_baseline, "company_news", direction_accuracy_baseline),
        "section": _portfolio_metrics(pf_section, "section_news", direction_accuracy_section),
        "section_distribution": dict(section_distribution),
    }


def _direction_accuracy(decisions: list[dict], rows: list[dict], _ticker: str) -> float:
    """Check BUY/SELL decisions against actual next-day direction from parquet data."""
    date_to_idx = {}
    for i, row in enumerate(rows):
        date_to_idx[str(row["date"])] = i

    correct = 0
    total = 0
    for d in decisions:
        if d["action"] == "HOLD":
            continue
        idx = date_to_idx.get(d["date"])
        if idx is None or idx + 1 >= len(rows):
            continue
        future_diff = rows[idx].get("future_price_diff", None)
        if future_diff is None:
            next_price = rows[idx + 1]["prices"]
            future_diff = next_price - d["price"]

        total += 1
        if (d["action"] == "BUY" and future_diff > 0) or (d["action"] == "SELL" and future_diff < 0):
            correct += 1

    return correct / total if total > 0 else 0.0


def _portfolio_metrics(pf: PortfolioTracker, name: str, direction_accuracy: float) -> dict:
    return {
        "strategy": name,
        "total_return_pct": round(pf.total_return_pct(), 4),
        "sharpe_ratio": round(pf.sharpe_ratio(), 4),
        "max_drawdown_pct": round(pf.max_drawdown_pct(), 4),
        "direction_accuracy": round(direction_accuracy, 4),
        "action_counts": pf.action_counts(),
        "final_value": round(pf.history[-1]["total_assets"], 2) if pf.history else 0.0,
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def print_comparison(results: list[dict]):
    """Print a side-by-side comparison table."""
    print("\n")
    print("=" * 80)
    print("  PERFORMANCE COMPARISON: company_news (baseline) vs section_news")
    print("=" * 80)

    for r in results:
        b = r["baseline"]
        s = r["section"]
        print(f"\n  Ticker: {r['ticker']}  |  {r['days']} days  |  {r['date_start']} to {r['date_end']}")
        print(f"  {'':30s} {'Baseline':>15s} {'Section News':>15s} {'Delta':>12s}")
        print(f"  {'-' * 72}")

        metrics = [
            ("Total Return (%)", "total_return_pct"),
            ("Sharpe Ratio", "sharpe_ratio"),
            ("Max Drawdown (%)", "max_drawdown_pct"),
            ("Direction Accuracy", "direction_accuracy"),
            ("Final Portfolio ($)", "final_value"),
        ]

        for label, key in metrics:
            bv = b[key]
            sv = s[key]
            delta = sv - bv
            sign = "+" if delta >= 0 else ""
            print(f"  {label:30s} {bv:>15.4f} {sv:>15.4f} {sign}{delta:>11.4f}")

        print("\n  Action distribution:")
        print(f"  {'':30s} {'Baseline':>15s} {'Section News':>15s}")
        for act in ["BUY", "SELL", "HOLD"]:
            print(f"  {act:30s} {b['action_counts'].get(act, 0):>15d} {s['action_counts'].get(act, 0):>15d}")

        if r.get("section_distribution"):
            print("\n  Section classification distribution:")
            for sec, count in sorted(r["section_distribution"].items(), key=lambda x: -x[1]):
                print(f"    {sec:35s} {count:>5d} items")

    print("\n" + "=" * 80)


def save_results(results: list[dict], output_dir: str):
    """Save results as JSON."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = Path(output_dir) / f"backtest_results_{timestamp}.json"
    with filepath.open("w", encoding="utf-8") as f:
        # Convert Signal enums in section distribution etc
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved to: {filepath}")
    return filepath


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Backtest section_news vs company_news")
    parser.add_argument("--ticker", nargs="+", default=["TSLA", "BTC"], help="Ticker(s) to backtest (default: TSLA BTC)")
    parser.add_argument("--days", type=int, default=None, help="Limit to first N days (default: all)")
    parser.add_argument("--save", type=str, default=None, help="Directory to save JSON results (e.g. results/)")
    args = parser.parse_args()

    all_results = []
    for ticker in args.ticker:
        result = run_backtest(ticker, max_days=args.days)
        all_results.append(result)

    print_comparison(all_results)

    if args.save:
        save_results(all_results, args.save)

    # Always save to default location too
    default_dir = REPO_ROOT / "results"
    save_results(all_results, str(default_dir))


if __name__ == "__main__":
    main()
