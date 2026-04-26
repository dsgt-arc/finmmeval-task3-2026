"""Compare portfolio performance between two backtest runs.

Each run is identified by a SQLite DB file and an experiment name (exp_name).
Typical usage: compare two branches (e.g. boyang-enhanced vs boyang) that were
run over the same date range with the same tickers.

Usage examples
--------------
# Compare two separate DB files (different branches saved to different files)
uv run python scripts/compare_backtest.py \
    --db1 results/enhanced.sqlite3 --exp1 dev_gemini \
    --db2 results/baseline.sqlite3 --exp2 dev_gemini

# Compare two exp_names inside the same DB (e.g. two configs run back-to-back)
uv run python scripts/compare_backtest.py \
    --db1 decision_making/database/deepfund.sqlite3 --exp1 dev_gemini_memory \
    --db2 decision_making/database/deepfund.sqlite3 --exp2 dev_gemini_baseline

# Save a JSON report
uv run python scripts/compare_backtest.py \
    --db1 results/enhanced.sqlite3 --exp1 dev_gemini \
    --db2 results/baseline.sqlite3 --exp2 dev_gemini \
    --save results/
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import math
from pathlib import Path
import sqlite3

import numpy as np

# ---------------------------------------------------------------------------
# Raw data helpers (no dependency on the package, works on any DB file)
# ---------------------------------------------------------------------------


def _connect(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def _get_config_id(db_path: str, exp_name: str) -> str:
    conn = _connect(db_path)
    try:
        row = conn.execute("SELECT id FROM config WHERE exp_name = ?", (exp_name,)).fetchone()
        if not row:
            raise ValueError(f"Experiment '{exp_name}' not found in {db_path}")
        return row["id"]
    finally:
        conn.close()


def _load_portfolio_history(db_path: str, config_id: str) -> list[dict]:
    """Return portfolio rows ordered by trading_date."""
    conn = _connect(db_path)
    try:
        rows = conn.execute(
            """
            SELECT trading_date, cashflow, total_assets
            FROM portfolio
            WHERE config_id = ?
              AND ABS(cashflow) < 1000000
              AND total_assets > 0
            ORDER BY trading_date
            """,
            (config_id,),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def _load_decisions(db_path: str, config_id: str) -> list[dict]:
    """Return all decisions joined with their portfolio row."""
    conn = _connect(db_path)
    try:
        rows = conn.execute(
            """
            SELECT d.trading_date, d.ticker, d.action, d.shares, d.price
            FROM decision d
            JOIN portfolio p ON d.portfolio_id = p.id
            WHERE p.config_id = ?
            ORDER BY d.trading_date, d.ticker
            """,
            (config_id,),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def _load_signals(db_path: str, config_id: str) -> list[dict]:
    """Return all analyst signals."""
    conn = _connect(db_path)
    try:
        rows = conn.execute(
            """
            SELECT p.trading_date, s.ticker, s.analyst, s.signal
            FROM signal s
            JOIN portfolio p ON s.portfolio_id = p.id
            WHERE p.config_id = ?
            ORDER BY p.trading_date
            """,
            (config_id,),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Metric calculations
# ---------------------------------------------------------------------------


def _daily_returns(portfolio_history: list[dict]) -> list[float]:
    values = [r["total_assets"] for r in portfolio_history]
    if len(values) < 2:
        return []
    return [(values[i] / values[i - 1] - 1) for i in range(1, len(values))]


def _sharpe(daily_returns: list[float], annualisation: int = 252) -> float:
    if len(daily_returns) < 2:
        return 0.0
    arr = np.array(daily_returns)
    std = arr.std()
    if std == 0:
        return 0.0
    return float(arr.mean() / std * math.sqrt(annualisation))


def _max_drawdown(portfolio_history: list[dict]) -> float:
    values = np.array([r["total_assets"] for r in portfolio_history])
    if len(values) == 0:
        return 0.0
    running_max = np.maximum.accumulate(values)
    drawdowns = (values - running_max) / running_max * 100
    return float(np.min(drawdowns))


def _total_return(portfolio_history: list[dict]) -> float:
    if len(portfolio_history) < 2:
        return 0.0
    start = portfolio_history[0]["total_assets"]
    end = portfolio_history[-1]["total_assets"]
    return (end / start - 1) * 100


def _annualized_return(total_return_pct: float, num_days: int) -> float:
    if num_days <= 0:
        return 0.0
    return ((1 + total_return_pct / 100) ** (365 / num_days) - 1) * 100


def _action_counts(decisions: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {"BUY": 0, "SELL": 0, "HOLD": 0}
    for d in decisions:
        action = d["action"].upper()
        counts[action] = counts.get(action, 0) + 1
    return counts


def _direction_accuracy(decisions: list[dict]) -> float:
    """Fraction of BUY/SELL decisions where the next recorded price moved correctly."""
    trades = [d for d in decisions if d["action"].upper() in ("BUY", "SELL")]
    if not trades:
        return 0.0

    # Group by ticker for next-price lookup
    by_ticker: dict[str, list[dict]] = {}
    for d in decisions:
        by_ticker.setdefault(d["ticker"], []).append(d)

    correct = 0
    evaluated = 0
    for trade in trades:
        ticker_decisions = by_ticker.get(trade["ticker"], [])
        idx = next((i for i, d in enumerate(ticker_decisions) if d is trade), None)
        if idx is None or idx + 1 >= len(ticker_decisions):
            continue
        next_price = ticker_decisions[idx + 1]["price"]
        evaluated += 1
        if (trade["action"].upper() == "BUY" and next_price > trade["price"]) or (
            trade["action"].upper() == "SELL" and next_price < trade["price"]
        ):
            correct += 1

    return correct / evaluated if evaluated else 0.0


def _signal_distribution(signals: list[dict]) -> dict[str, dict[str, int]]:
    """Return {analyst: {signal: count}} breakdown."""
    result: dict[str, dict[str, int]] = {}
    for s in signals:
        analyst = s["analyst"]
        signal = s["signal"]
        result.setdefault(analyst, {})
        result[analyst][signal] = result[analyst].get(signal, 0) + 1
    return result


# ---------------------------------------------------------------------------
# Main comparison logic
# ---------------------------------------------------------------------------


def compute_metrics(db_path: str, exp_name: str, label: str) -> dict:
    """Load one experiment from a DB and compute all metrics."""
    config_id = _get_config_id(db_path, exp_name)
    history = _load_portfolio_history(db_path, config_id)
    decisions = _load_decisions(db_path, config_id)
    signals = _load_signals(db_path, config_id)

    if not history:
        raise ValueError(f"No portfolio data found for exp='{exp_name}' in {db_path}")

    rets = _daily_returns(history)
    total_ret = _total_return(history)

    first_date = datetime.fromisoformat(history[0]["trading_date"])
    last_date = datetime.fromisoformat(history[-1]["trading_date"])
    num_days = (last_date - first_date).days or 1

    return {
        "label": label,
        "exp_name": exp_name,
        "db_path": db_path,
        "date_start": str(first_date.date()),
        "date_end": str(last_date.date()),
        "trading_days": len(history),
        "num_calendar_days": num_days,
        "initial_value": history[0]["total_assets"],
        "final_value": history[-1]["total_assets"],
        "total_return_pct": round(total_ret, 4),
        "annualized_return_pct": round(_annualized_return(total_ret, num_days), 4),
        "sharpe_ratio": round(_sharpe(rets), 4),
        "max_drawdown_pct": round(_max_drawdown(history), 4),
        "direction_accuracy": round(_direction_accuracy(decisions), 4),
        "action_counts": _action_counts(decisions),
        "signal_distribution": _signal_distribution(signals),
        "portfolio_history": [{"date": r["trading_date"], "total_assets": r["total_assets"]} for r in history],
    }


def print_report(m1: dict, m2: dict) -> None:
    w = 80
    print("\n" + "=" * w)
    print("  BACKTEST COMPARISON")
    print(f"  {m1['label']:<35s}  vs  {m2['label']}")
    print("=" * w)

    # Date range check
    if m1["date_start"] != m2["date_start"] or m1["date_end"] != m2["date_end"]:
        print("  ⚠  Date ranges differ:")
        print(f"     {m1['label']}: {m1['date_start']} → {m1['date_end']}  ({m1['trading_days']} days)")
        print(f"     {m2['label']}: {m2['date_start']} → {m2['date_end']}  ({m2['trading_days']} days)")
    else:
        print(f"  Period : {m1['date_start']} → {m1['date_end']}  ({m1['trading_days']} trading days)")
    print()

    # Core metrics table
    metrics = [
        ("Initial Portfolio ($)", "initial_value", ",.2f"),
        ("Final Portfolio ($)", "final_value", ",.2f"),
        ("Total Return (%)", "total_return_pct", ".4f"),
        ("Annualized Return (%)", "annualized_return_pct", ".4f"),
        ("Sharpe Ratio", "sharpe_ratio", ".4f"),
        ("Max Drawdown (%)", "max_drawdown_pct", ".4f"),
        ("Direction Accuracy", "direction_accuracy", ".4f"),
    ]

    col_w = 22
    print(f"  {'Metric':<30s} {m1['label']:>{col_w}s} {m2['label']:>{col_w}s} {'Delta':>12s}")
    print(f"  {'-' * (30 + col_w * 2 + 14)}")
    for label, key, fmt in metrics:
        v1 = m1[key]
        v2 = m2[key]
        delta = v2 - v1
        sign = "+" if delta >= 0 else ""
        fmt_str = f"{{:{fmt}}}"
        print(f"  {label:<30s} {fmt_str.format(v1):>{col_w}s} {fmt_str.format(v2):>{col_w}s} {sign}{delta:>+11.4f}")

    # Action distribution
    print(f"\n  {'Action distribution':}")
    print(f"  {'':30s} {m1['label']:>{col_w}s} {m2['label']:>{col_w}s}")
    for act in ["BUY", "SELL", "HOLD"]:
        c1 = m1["action_counts"].get(act, 0)
        c2 = m2["action_counts"].get(act, 0)
        print(f"  {act:<30s} {c1:>{col_w}d} {c2:>{col_w}d}")

    # Signal distribution per analyst
    all_analysts = sorted(set(list(m1["signal_distribution"]) + list(m2["signal_distribution"])))
    if all_analysts:
        print("\n  Analyst signal breakdown:")
        for analyst in all_analysts:
            d1 = m1["signal_distribution"].get(analyst, {})
            d2 = m2["signal_distribution"].get(analyst, {})
            print(f"  {analyst}:")
            for sig in ["Bullish", "Bearish", "Neutral"]:
                c1 = d1.get(sig, 0)
                c2 = d2.get(sig, 0)
                print(f"    {sig:<28s} {c1:>{col_w}d} {c2:>{col_w}d}")

    print("\n" + "=" * w)

    # Winner summary
    winner_metrics = {
        "Total Return": m2["total_return_pct"] > m1["total_return_pct"],
        "Sharpe Ratio": m2["sharpe_ratio"] > m1["sharpe_ratio"],
        "Max Drawdown": m2["max_drawdown_pct"] > m1["max_drawdown_pct"],  # less negative = better
        "Direction Acc.": m2["direction_accuracy"] > m1["direction_accuracy"],
    }
    m2_wins = sum(1 for v in winner_metrics.values() if v)
    m1_wins = len(winner_metrics) - m2_wins

    print(f"\n  Summary: {m1['label']} wins {m1_wins}/4 metrics, {m2['label']} wins {m2_wins}/4 metrics")
    for metric, m2_better in winner_metrics.items():
        winner = m2["label"] if m2_better else m1["label"]
        print(f"    {metric:<25s} → {winner}")
    print()


def save_report(m1: dict, m2: dict, output_dir: str) -> Path:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    fpath = Path(output_dir) / f"comparison_{ts}.json"
    report = {
        "generated_at": ts,
        "run_1": {k: v for k, v in m1.items() if k != "portfolio_history"},
        "run_2": {k: v for k, v in m2.items() if k != "portfolio_history"},
    }
    with fpath.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"  Report saved to: {fpath}")
    return fpath


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two backtest runs by portfolio performance")
    parser.add_argument("--db1", required=True, help="Path to SQLite DB for run 1")
    parser.add_argument("--exp1", required=True, help="exp_name for run 1")
    parser.add_argument("--label1", default=None, help="Display label for run 1 (default: exp1)")
    parser.add_argument("--db2", required=True, help="Path to SQLite DB for run 2")
    parser.add_argument("--exp2", required=True, help="exp_name for run 2")
    parser.add_argument("--label2", default=None, help="Display label for run 2 (default: exp2)")
    parser.add_argument("--save", default="results/", help="Directory to save JSON report (default: results/)")
    args = parser.parse_args()

    label1 = args.label1 or args.exp1
    label2 = args.label2 or args.exp2

    print(f"\nLoading run 1: {label1}  ({args.db1})")
    m1 = compute_metrics(args.db1, args.exp1, label1)

    print(f"Loading run 2: {label2}  ({args.db2})")
    m2 = compute_metrics(args.db2, args.exp2, label2)

    print_report(m1, m2)
    save_report(m1, m2, args.save)


if __name__ == "__main__":
    main()
