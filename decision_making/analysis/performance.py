"""
Performance metrics and analysis functions for DeepFund trading results.
"""

from datetime import datetime
import json

import numpy as np
import polars as pl

from decision_making.ama_data import load_data
from decision_making.analysis.queries import get_all_portfolio_records
from decision_making.database.sqlite_helper import SQLiteDB


def get_portfolio_timeseries(db: SQLiteDB, config_id: str) -> pl.DataFrame:
    """
    Get portfolio timeseries with parsed positions and calculated metrics.

    Args:
        db: SQLiteDB instance
        config_id: Configuration ID

    Returns:
        Polars DataFrame with columns:
        - trading_date: datetime
        - cashflow: float
        - total_assets: float
        - positions_dict: dict (parsed JSON)
        - daily_return: float (percentage)
        - cumulative_return: float (percentage from start)
    """
    df = get_all_portfolio_records(db, config_id)

    # Parse positions JSON
    df = df.with_columns(
        pl.col("positions").map_elements(lambda x: json.loads(x) if x else {}, return_dtype=pl.Object).alias("positions_dict")
    )

    # Convert trading_date to datetime if it's a string
    if df["trading_date"].dtype == pl.Utf8 or df["trading_date"].dtype == pl.String:
        df = df.with_columns(pl.col("trading_date").str.to_datetime())

    # Calculate daily returns
    df = df.with_columns(
        (pl.col("total_assets").pct_change() * 100).alias("daily_return_pct"),
    )

    # Calculate cumulative returns from start
    initial_value = df["total_assets"][0]
    df = df.with_columns(((pl.col("total_assets") / initial_value - 1) * 100).alias("cumulative_return_pct"))

    return df


def extract_ticker_positions(portfolio_df: pl.DataFrame, ticker: str) -> pl.DataFrame:
    """
    Extract position history for a specific ticker from portfolio timeseries.

    Args:
        portfolio_df: DataFrame from get_portfolio_timeseries()
        ticker: Ticker symbol (e.g., 'TSLA', 'BTC')

    Returns:
        DataFrame with columns:
        - trading_date
        - shares: int
        - value: float
        - price_per_share: float (calculated if shares > 0)
    """

    def extract_position(positions_dict: dict) -> tuple:
        """Extract shares and value for ticker."""
        if not positions_dict or ticker not in positions_dict:
            return (0, 0.0)
        pos = positions_dict[ticker]
        return (pos.get("shares", 0), pos.get("value", 0.0))

    # Extract ticker data
    result = portfolio_df.select(
        pl.col("trading_date"),
        pl.col("positions_dict").map_elements(lambda x: extract_position(x)[0], return_dtype=pl.Int64).alias("shares"),
        pl.col("positions_dict").map_elements(lambda x: extract_position(x)[1], return_dtype=pl.Float64).alias("value"),
    )

    # Calculate price per share (avoid division by zero)
    result = result.with_columns(
        pl.when(pl.col("shares") > 0).then(pl.col("value") / pl.col("shares")).otherwise(0.0).alias("price_per_share")
    )

    return result


def calculate_sharpe_ratio(returns: pl.Series, risk_free_rate: float = 0.0, annualization_factor: float = 252) -> float:
    """
    Calculate Sharpe ratio from return series.

    Args:
        returns: Series of returns (as percentages or decimals)
        risk_free_rate: Annual risk-free rate (default: 0)
        annualization_factor: Number of periods per year (252 for daily trading days)

    Returns:
        Annualized Sharpe ratio
    """
    returns_clean = returns.drop_nulls()

    if len(returns_clean) == 0:
        return 0.0

    mean_return = returns_clean.mean()
    std_return = returns_clean.std()

    if std_return is None or std_return == 0:
        return 0.0

    # Annualize
    sharpe = (mean_return - risk_free_rate / annualization_factor) / std_return * np.sqrt(annualization_factor)

    return sharpe


def calculate_max_drawdown(portfolio_values: pl.Series) -> tuple[float, int | None, int | None]:
    """
    Calculate maximum drawdown from portfolio value series.

    Args:
        portfolio_values: Series of portfolio values

    Returns:
        Tuple of (max_drawdown_pct, peak_idx, trough_idx)
    """
    if len(portfolio_values) == 0:
        return (0.0, None, None)

    # Convert to numpy for easier calculation
    values = portfolio_values.to_numpy()

    # Calculate running maximum
    running_max = np.maximum.accumulate(values)

    # Calculate drawdown at each point
    drawdown = (values - running_max) / running_max * 100

    # Find maximum drawdown
    max_dd_idx = np.argmin(drawdown)
    max_dd = drawdown[max_dd_idx]

    # Find the peak that led to this drawdown
    peak_idx = np.argmax(values[: max_dd_idx + 1]) if max_dd_idx > 0 else 0

    return (max_dd, int(peak_idx), int(max_dd_idx))


def calculate_buy_hold_benchmark(
    ticker: str, start_date: datetime, end_date: datetime, initial_capital: float = 100000.0
) -> pl.DataFrame:
    """
    Calculate buy-and-hold benchmark for a ticker.

    Args:
        ticker: Ticker symbol ('TSLA' or 'BTC')
        start_date: Start date for benchmark
        end_date: End date for benchmark
        initial_capital: Initial investment amount

    Returns:
        DataFrame with columns: date, price, portfolio_value, cumulative_return_pct
    """
    # Load price data
    price_data = load_data(ticker, competition_data=True)

    # Ensure date column is datetime
    if "date" not in price_data.columns:
        raise ValueError(f"Price data for {ticker} must have 'date' column")

    if price_data["date"].dtype != pl.Datetime and price_data["date"].dtype != pl.Date:
        price_data = price_data.with_columns(pl.col("date").str.to_datetime())

    # Filter to date range
    filtered = price_data.filter((pl.col("date") >= start_date) & (pl.col("date") <= end_date)).sort("date")

    if len(filtered) == 0:
        raise ValueError(f"No price data found for {ticker} between {start_date} and {end_date}")

    # Get initial price and calculate shares
    initial_price = filtered["prices"][0]
    shares = initial_capital / initial_price

    # Calculate portfolio value and returns
    result = filtered.select(
        pl.col("date"),
        pl.col("prices").alias("price"),
        (pl.col("prices") * shares).alias("portfolio_value"),
        ((pl.col("prices") / initial_price - 1) * 100).alias("cumulative_return_pct"),
    )

    return result


def calculate_portfolio_benchmark(
    tickers: list[str],
    weights: list[float],
    start_date: datetime,
    end_date: datetime,
    initial_capital: float = 100000.0,
) -> pl.DataFrame:
    """
    Calculate multi-asset portfolio benchmark with fixed weights.

    Args:
        tickers: List of ticker symbols
        weights: List of weights (must sum to 1.0)
        start_date: Start date
        end_date: End date
        initial_capital: Initial investment amount

    Returns:
        DataFrame with portfolio value over time
    """
    if len(tickers) != len(weights):
        raise ValueError("Tickers and weights must have same length")
    if not np.isclose(sum(weights), 1.0):
        raise ValueError("Weights must sum to 1.0")

    # Calculate individual benchmarks
    benchmarks = []
    for ticker, weight in zip(tickers, weights):
        bm = calculate_buy_hold_benchmark(ticker, start_date, end_date, initial_capital * weight)
        bm = bm.rename({"portfolio_value": f"{ticker}_value"})
        benchmarks.append(bm)

    # Merge on date
    result = benchmarks[0]
    for bm in benchmarks[1:]:
        result = result.join(bm.select(["date", f"{bm.columns[2]}"]), on="date", how="inner")

    # Calculate total portfolio value
    value_cols = [col for col in result.columns if col.endswith("_value")]
    result = result.with_columns(pl.sum_horizontal(value_cols).alias("portfolio_value"))

    # Calculate cumulative return
    initial_value = result["portfolio_value"][0]
    result = result.with_columns(((pl.col("portfolio_value") / initial_value - 1) * 100).alias("cumulative_return_pct"))

    return result.select(["date", "portfolio_value", "cumulative_return_pct"])


def calculate_win_rate(decisions_df: pl.DataFrame, portfolio_df: pl.DataFrame) -> dict[str, float]:
    """
    Calculate win rate for Buy and Sell decisions.

    A Buy decision is considered a "win" if the portfolio value increased after it.
    A Sell decision is considered a "win" if the portfolio value decreased or didn't increase as much after it.

    Note: This is a simplified calculation and may not fully capture profitability.

    Args:
        decisions_df: DataFrame from get_all_decisions()
        portfolio_df: DataFrame from get_portfolio_timeseries()

    Returns:
        Dictionary with win rates: {'buy_win_rate': X.X, 'sell_win_rate': X.X, 'overall_win_rate': X.X}
    """
    # Filter out Hold decisions
    trades = decisions_df.filter(pl.col("action").is_in(["Buy", "Sell"]))

    if len(trades) == 0:
        return {"buy_win_rate": 0.0, "sell_win_rate": 0.0, "overall_win_rate": 0.0}

    # Simple heuristic: compare portfolio value before and after (next trading day)
    # This is approximate since we don't have exact transaction timestamps

    # For now, return placeholder values
    # A proper implementation would track P&L for each position
    return {
        "buy_win_rate": 0.0,  # Placeholder
        "sell_win_rate": 0.0,  # Placeholder
        "overall_win_rate": 0.0,  # Placeholder
    }


def calculate_annualized_return(total_return_pct: float, num_days: int) -> float:
    """
    Calculate annualized return from total return and number of days.

    Args:
        total_return_pct: Total return as percentage
        num_days: Number of days in the period

    Returns:
        Annualized return as percentage
    """
    if num_days <= 0:
        return 0.0

    # Convert percentage to decimal, annualize, convert back
    return ((1 + total_return_pct / 100) ** (365 / num_days) - 1) * 100


def calculate_realized_pnl(decisions_df: pl.DataFrame, initial_capital: float = 100000.0) -> pl.DataFrame:
    """
    Calculate cumulative P&L assuming 1-share trades (normalizes by position size).

    Treats every trade as buying/selling exactly 1 share:
    - Buy: -price (spend price to buy 1 share)
    - Sell: +price (receive price from selling 1 share)
    - Hold: 0
    - Cumulative P&L: Running sum of normalized cash flows

    This removes position sizing effects and focuses purely on price timing.

    Args:
        decisions_df: DataFrame from get_all_decisions() with columns:
                      [trading_date, ticker, action, shares, price]
        initial_capital: Starting capital amount (for calculating return %)

    Returns:
        DataFrame with columns:
        - trading_date: datetime
        - ticker: str
        - action: str
        - cashflow: float (cash in/out from 1-share trade)
        - cumulative_cashflow: float (running total)
        - cumulative_return_pct: float (cumulative as % of initial capital)
    """
    # Convert to pandas for easier processing
    df = decisions_df.sort("trading_date").to_pandas()

    # Calculate cash flow for each decision (normalized to 1 share)
    cashflow_list = []

    for _, row in df.iterrows():
        action = row["action"]
        price = row["price"]

        if action == "Buy":
            # Buy 1 share: spend the price
            cashflow = -price
        elif action == "Sell":
            # Sell 1 share: receive the price
            cashflow = price
        else:  # Hold
            cashflow = 0.0

        cashflow_list.append(cashflow)

    # Add cashflow to dataframe
    df["cashflow"] = cashflow_list

    # Calculate cumulative cash flow
    df["cumulative_cashflow"] = df["cashflow"].cumsum()

    # Calculate cumulative return as % of initial capital
    df["cumulative_return_pct"] = (df["cumulative_cashflow"] / initial_capital) * 100

    # Convert back to polars
    result = pl.from_pandas(
        df[["trading_date", "ticker", "action", "shares", "price", "cashflow", "cumulative_cashflow", "cumulative_return_pct"]]
    )

    return result
