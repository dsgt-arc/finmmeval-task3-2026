"""
Performance metrics and analysis functions for market timing results.

Performance is computed by replaying BUY/SELL/HOLD decisions against actual asset prices:
  - BUY  (long)    → capture +asset_return the next day
  - SELL (short)   → capture -asset_return the next day
  - HOLD (neutral) → 0 contribution

Hit rate measures directional accuracy: did the signal correctly predict next-day direction?
"""

from datetime import datetime

import numpy as np
import polars as pl

from decision_making.ama_data import load_data
from decision_making.analysis.queries import get_all_decisions, get_all_signals
from decision_making.database.sqlite_helper import SQLiteDB


# ---------------------------------------------------------------------------
# Timing performance
# ---------------------------------------------------------------------------

def calculate_timing_performance(db: SQLiteDB, config_id: str, ticker: str) -> pl.DataFrame:
    """
    Compute cumulative return from market timing signals for a single ticker.

    For each decision on date T:
      - BUY  (long)  → timing_return =  asset_daily_return(T→T+1)
      - SELL (short) → timing_return = -asset_daily_return(T→T+1)
      - HOLD         → timing_return =  0

    Args:
        db: SQLiteDB instance
        config_id: Configuration ID
        ticker: Ticker symbol

    Returns:
        DataFrame with columns:
          trading_date, action, price, asset_return_pct,
          timing_return_pct, cumulative_return_pct
    """
    decisions = get_all_decisions(db, config_id)
    decisions = decisions.filter(pl.col("ticker") == ticker)

    if len(decisions) == 0:
        return pl.DataFrame()

    price_data = load_data(ticker, competition_data=True)
    if price_data["date"].dtype != pl.Datetime and price_data["date"].dtype != pl.Date:
        price_data = price_data.with_columns(pl.col("date").str.to_datetime())

    # Build a price map: date → next_day_price for forward return calculation
    price_sorted = price_data.sort("date")
    price_with_next = price_sorted.with_columns(
        pl.col("prices").shift(-1).alias("next_price")
    ).drop_nulls("next_price")

    # Normalise decision trading_date to date only for joining
    if decisions["trading_date"].dtype == pl.Utf8 or decisions["trading_date"].dtype == pl.String:
        decisions = decisions.with_columns(pl.col("trading_date").str.to_datetime())

    decisions = decisions.with_columns(
        pl.col("trading_date").dt.date().cast(pl.Datetime).alias("join_date")
    )
    price_with_next = price_with_next.with_columns(
        pl.col("date").dt.date().cast(pl.Datetime).alias("join_date")
    )

    joined = decisions.join(
        price_with_next.select(["join_date", "prices", "next_price"]),
        on="join_date",
        how="left",
    )

    # asset daily return (%)
    joined = joined.with_columns(
        ((pl.col("next_price") - pl.col("prices")) / pl.col("prices") * 100).alias("asset_return_pct")
    )

    # timing return based on position direction
    joined = joined.with_columns(
        pl.when(pl.col("action") == "Buy")
        .then(pl.col("asset_return_pct"))
        .when(pl.col("action") == "Sell")
        .then(-pl.col("asset_return_pct"))
        .otherwise(0.0)
        .alias("timing_return_pct")
    )

    # cumulative return: product of (1 + r/100) - 1, expressed as %
    timing_returns = joined["timing_return_pct"].to_numpy() / 100.0
    cumulative = (np.cumprod(1 + timing_returns) - 1) * 100

    joined = joined.with_columns(
        pl.Series("cumulative_return_pct", cumulative)
    )

    return joined.select([
        "trading_date",
        "action",
        "price",
        "asset_return_pct",
        "timing_return_pct",
        "cumulative_return_pct",
    ]).sort("trading_date")


# ---------------------------------------------------------------------------
# Hit rate — decisions
# ---------------------------------------------------------------------------

def calculate_decision_hit_rate(
    db: SQLiteDB, config_id: str, lookahead_days: int = 1
) -> pl.DataFrame:
    """
    Directional accuracy of BUY/SELL decisions.

    A BUY is a hit when price(T+lookahead) > price(T).
    A SELL is a hit when price(T+lookahead) < price(T).
    HOLD decisions are excluded.

    Args:
        db: SQLiteDB instance
        config_id: Configuration ID
        lookahead_days: Number of trading days to look ahead (default 1)

    Returns:
        DataFrame with columns:
          ticker, action, total_decisions, hits, hit_rate_pct
    """
    decisions = get_all_decisions(db, config_id)
    trades = decisions.filter(pl.col("action").is_in(["Buy", "Sell"]))

    if len(trades) == 0:
        return pl.DataFrame(schema={
            "ticker": pl.Utf8,
            "action": pl.Utf8,
            "total_decisions": pl.Int64,
            "hits": pl.Int64,
            "hit_rate_pct": pl.Float64,
        })

    if trades["trading_date"].dtype == pl.Utf8 or trades["trading_date"].dtype == pl.String:
        trades = trades.with_columns(pl.col("trading_date").str.to_datetime())

    results = []
    for ticker in trades["ticker"].unique().to_list():
        ticker_trades = trades.filter(pl.col("ticker") == ticker)

        price_data = load_data(ticker, competition_data=True)
        if price_data["date"].dtype != pl.Datetime and price_data["date"].dtype != pl.Date:
            price_data = price_data.with_columns(pl.col("date").str.to_datetime())

        price_sorted = price_data.sort("date")
        price_with_forward = price_sorted.with_columns(
            pl.col("prices").shift(-lookahead_days).alias("forward_price")
        ).drop_nulls("forward_price")

        ticker_trades = ticker_trades.with_columns(
            pl.col("trading_date").dt.date().cast(pl.Datetime).alias("join_date")
        )
        price_with_forward = price_with_forward.with_columns(
            pl.col("date").dt.date().cast(pl.Datetime).alias("join_date")
        )

        joined = ticker_trades.join(
            price_with_forward.select(["join_date", "prices", "forward_price"]),
            on="join_date",
            how="left",
        ).drop_nulls("forward_price")

        joined = joined.with_columns(
            pl.when(
                (pl.col("action") == "Buy") & (pl.col("forward_price") > pl.col("prices"))
            ).then(1)
            .when(
                (pl.col("action") == "Sell") & (pl.col("forward_price") < pl.col("prices"))
            ).then(1)
            .otherwise(0)
            .alias("hit")
        )

        summary = (
            joined.group_by("action")
            .agg([
                pl.len().alias("total_decisions"),
                pl.col("hit").sum().alias("hits"),
            ])
            .with_columns([
                pl.lit(ticker).alias("ticker"),
                (pl.col("hits") / pl.col("total_decisions") * 100).alias("hit_rate_pct"),
            ])
        )
        results.append(summary)

    if not results:
        return pl.DataFrame()

    return (
        pl.concat(results)
        .select(["ticker", "action", "total_decisions", "hits", "hit_rate_pct"])
        .sort(["ticker", "action"])
    )


# ---------------------------------------------------------------------------
# Hit rate — analyst signals
# ---------------------------------------------------------------------------

def calculate_signal_hit_rate(
    db: SQLiteDB, config_id: str, lookahead_days: int = 1
) -> pl.DataFrame:
    """
    Directional accuracy of analyst signals.

    Bullish is a hit when price(T+lookahead) > price(T).
    Bearish is a hit when price(T+lookahead) < price(T).
    Neutral signals are excluded.

    Args:
        db: SQLiteDB instance
        config_id: Configuration ID
        lookahead_days: Number of trading days to look ahead (default 1)

    Returns:
        DataFrame with columns:
          analyst, ticker, signal, total_signals, hits, hit_rate_pct
    """
    signals = get_all_signals(db, config_id)
    directional = signals.filter(pl.col("signal").is_in(["Bullish", "Bearish"]))

    if len(directional) == 0:
        return pl.DataFrame(schema={
            "analyst": pl.Utf8,
            "ticker": pl.Utf8,
            "signal": pl.Utf8,
            "total_signals": pl.Int64,
            "hits": pl.Int64,
            "hit_rate_pct": pl.Float64,
        })

    if directional["trading_date"].dtype == pl.Utf8 or directional["trading_date"].dtype == pl.String:
        directional = directional.with_columns(pl.col("trading_date").str.to_datetime())

    results = []
    for ticker in directional["ticker"].unique().to_list():
        ticker_signals = directional.filter(pl.col("ticker") == ticker)

        price_data = load_data(ticker, competition_data=True)
        if price_data["date"].dtype != pl.Datetime and price_data["date"].dtype != pl.Date:
            price_data = price_data.with_columns(pl.col("date").str.to_datetime())

        price_sorted = price_data.sort("date")
        price_with_forward = price_sorted.with_columns(
            pl.col("prices").shift(-lookahead_days).alias("forward_price")
        ).drop_nulls("forward_price")

        ticker_signals = ticker_signals.with_columns(
            pl.col("trading_date").dt.date().cast(pl.Datetime).alias("join_date")
        )
        price_with_forward = price_with_forward.with_columns(
            pl.col("date").dt.date().cast(pl.Datetime).alias("join_date")
        )

        joined = ticker_signals.join(
            price_with_forward.select(["join_date", "prices", "forward_price"]),
            on="join_date",
            how="left",
        ).drop_nulls("forward_price")

        joined = joined.with_columns(
            pl.when(
                (pl.col("signal") == "Bullish") & (pl.col("forward_price") > pl.col("prices"))
            ).then(1)
            .when(
                (pl.col("signal") == "Bearish") & (pl.col("forward_price") < pl.col("prices"))
            ).then(1)
            .otherwise(0)
            .alias("hit")
        )

        summary = (
            joined.group_by(["analyst", "signal"])
            .agg([
                pl.len().alias("total_signals"),
                pl.col("hit").sum().alias("hits"),
            ])
            .with_columns([
                pl.lit(ticker).alias("ticker"),
                (pl.col("hits") / pl.col("total_signals") * 100).alias("hit_rate_pct"),
            ])
        )
        results.append(summary)

    if not results:
        return pl.DataFrame()

    return (
        pl.concat(results)
        .select(["analyst", "ticker", "signal", "total_signals", "hits", "hit_rate_pct"])
        .sort(["analyst", "ticker", "signal"])
    )


# ---------------------------------------------------------------------------
# Benchmark & statistical helpers (unchanged)
# ---------------------------------------------------------------------------

def calculate_buy_hold_benchmark(
    ticker: str, start_date: datetime, end_date: datetime, initial_capital: float = 100000.0
) -> pl.DataFrame:
    """Buy-and-hold benchmark for a single ticker."""
    price_data = load_data(ticker, competition_data=True)

    if "date" not in price_data.columns:
        raise ValueError(f"Price data for {ticker} must have 'date' column")

    if price_data["date"].dtype != pl.Datetime and price_data["date"].dtype != pl.Date:
        price_data = price_data.with_columns(pl.col("date").str.to_datetime())

    filtered = price_data.filter((pl.col("date") >= start_date) & (pl.col("date") <= end_date)).sort("date")

    if len(filtered) == 0:
        raise ValueError(f"No price data found for {ticker} between {start_date} and {end_date}")

    initial_price = filtered["prices"][0]
    shares = initial_capital / initial_price

    return filtered.select(
        pl.col("date"),
        pl.col("prices").alias("price"),
        (pl.col("prices") * shares).alias("portfolio_value"),
        ((pl.col("prices") / initial_price - 1) * 100).alias("cumulative_return_pct"),
    )


def calculate_portfolio_benchmark(
    tickers: list[str],
    weights: list[float],
    start_date: datetime,
    end_date: datetime,
    initial_capital: float = 100000.0,
) -> pl.DataFrame:
    """Multi-asset portfolio benchmark with fixed weights."""
    if len(tickers) != len(weights):
        raise ValueError("Tickers and weights must have same length")
    if not np.isclose(sum(weights), 1.0):
        raise ValueError("Weights must sum to 1.0")

    benchmarks = []
    for ticker, weight in zip(tickers, weights):
        bm = calculate_buy_hold_benchmark(ticker, start_date, end_date, initial_capital * weight)
        bm = bm.rename({"portfolio_value": f"{ticker}_value"})
        benchmarks.append(bm)

    result = benchmarks[0]
    for bm in benchmarks[1:]:
        result = result.join(bm.select(["date", f"{bm.columns[2]}"]), on="date", how="inner")

    value_cols = [col for col in result.columns if col.endswith("_value")]
    result = result.with_columns(pl.sum_horizontal(value_cols).alias("portfolio_value"))

    initial_value = result["portfolio_value"][0]
    result = result.with_columns(((pl.col("portfolio_value") / initial_value - 1) * 100).alias("cumulative_return_pct"))

    return result.select(["date", "portfolio_value", "cumulative_return_pct"])


def calculate_sharpe_ratio(returns: pl.Series, risk_free_rate: float = 0.0, annualization_factor: float = 252) -> float:
    """Annualized Sharpe ratio from a return series (as percentages)."""
    returns_clean = returns.drop_nulls()

    if len(returns_clean) == 0:
        return 0.0

    mean_return = returns_clean.mean()
    std_return = returns_clean.std()

    if std_return is None or std_return == 0:
        return 0.0

    return (mean_return - risk_free_rate / annualization_factor) / std_return * np.sqrt(annualization_factor)


def calculate_max_drawdown(portfolio_values: pl.Series) -> tuple[float, int | None, int | None]:
    """Maximum drawdown from a value series. Returns (max_dd_pct, peak_idx, trough_idx)."""
    if len(portfolio_values) == 0:
        return (0.0, None, None)

    values = portfolio_values.to_numpy()
    running_max = np.maximum.accumulate(values)
    drawdown = (values - running_max) / running_max * 100

    max_dd_idx = np.argmin(drawdown)
    max_dd = drawdown[max_dd_idx]
    peak_idx = np.argmax(values[: max_dd_idx + 1]) if max_dd_idx > 0 else 0

    return (max_dd, int(peak_idx), int(max_dd_idx))


def calculate_annualized_return(total_return_pct: float, num_days: int) -> float:
    """Annualized return from total return (%) and number of days."""
    if num_days <= 0:
        return 0.0
    return ((1 + total_return_pct / 100) ** (365 / num_days) - 1) * 100


# ---------------------------------------------------------------------------
# Portfolio position helpers
# ---------------------------------------------------------------------------

def extract_ticker_positions(portfolio_df: pl.DataFrame, ticker: str) -> pl.DataFrame:
    """
    Extract per-ticker position value from the positions JSON column.

    The positions column stores JSON of the form:
        {"TSLA": {"shares": N, "value": V}, "BTC": {"shares": N, "value": V}}

    Args:
        portfolio_df: DataFrame returned by get_portfolio_timeseries / get_all_portfolio_records
        ticker: Ticker symbol to extract

    Returns:
        DataFrame with columns: trading_date, value
    """
    import json as _json

    def _get_value(positions_raw) -> float:
        try:
            pos = _json.loads(positions_raw) if isinstance(positions_raw, str) else positions_raw
            return float(pos.get(ticker, {}).get("value", 0.0))
        except Exception:
            return 0.0

    values = [_get_value(p) for p in portfolio_df["positions"].to_list()]
    return pl.DataFrame({"trading_date": portfolio_df["trading_date"], "value": values})


def calculate_realized_pnl(
    decisions_df: pl.DataFrame, initial_capital: float = 100000.0
) -> pl.DataFrame:
    """
    Compute normalized P&L assuming exactly 1 share per trade.

    Cash-flow convention (per decision):
      - BUY  → cashflow = -price  (spend 1 share worth)
      - SELL → cashflow = +price  (receive 1 share worth)
      - HOLD → cashflow = 0

    cumulative_return_pct is the running cashflow expressed as a percentage
    of initial_capital, giving a position-size-neutral view of timing quality.

    Args:
        decisions_df: DataFrame from get_all_decisions (all tickers combined)
        initial_capital: Denominator for return normalisation

    Returns:
        DataFrame with columns:
          trading_date, ticker, action, price, cashflow,
          cumulative_cashflow, cumulative_return_pct
    """
    df = decisions_df
    if df["trading_date"].dtype in (pl.Utf8, pl.String):
        df = df.with_columns(pl.col("trading_date").str.to_datetime())

    df = df.sort("trading_date")

    df = df.with_columns(
        pl.when(pl.col("action") == "Buy")
        .then(-pl.col("price"))
        .when(pl.col("action") == "Sell")
        .then(pl.col("price"))
        .otherwise(0.0)
        .alias("cashflow")
    )

    cumsum = df["cashflow"].cum_sum()
    df = df.with_columns([
        cumsum.alias("cumulative_cashflow"),
        (cumsum / initial_capital * 100).alias("cumulative_return_pct"),
    ])

    return df.select([
        "trading_date", "ticker", "action", "price",
        "cashflow", "cumulative_cashflow", "cumulative_return_pct",
    ])
