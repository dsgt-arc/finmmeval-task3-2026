"""
SQL query functions for extracting data from the DS@GT StockTron SQLite database.
"""

from datetime import datetime
from typing import Any

import polars as pl

from decision_making.database.sqlite_helper import SQLiteDB


def get_all_portfolio_records(db: SQLiteDB, config_id: str, filter_corrupted: bool = True) -> pl.DataFrame:
    """
    Get all portfolio records for a config ordered by trading date.

    Args:
        db: SQLiteDB instance
        config_id: Configuration ID
        filter_corrupted: If True, exclude records with unrealistic values (default: True)

    Returns:
        Polars DataFrame with columns: trading_date, cashflow, total_assets, positions
    """
    conn = db._get_connection()
    try:
        corruption_filter = "AND total_assets > 0" if filter_corrupted else ""

        query = f"""
            SELECT
                trading_date,
                cashflow,
                total_assets,
                positions
            FROM portfolio
            WHERE config_id = ?
            {corruption_filter}
            ORDER BY trading_date
        """
        # Execute query with parameters
        cursor = conn.cursor()
        cursor.execute(query, (config_id,))

        # Fetch results and convert to list of dicts
        columns = [desc[0] for desc in cursor.description]
        rows = cursor.fetchall()

        # Convert to list of dicts for better type inference
        data = [{col: row[i] for i, col in enumerate(columns)} for row in rows]

        # Create polars DataFrame from dicts
        df = pl.from_dicts(data)
        return df
    finally:
        conn.close()


def get_portfolio_timeseries(db: SQLiteDB, config_id: str) -> pl.DataFrame:
    """
    Portfolio records with derived daily_return_pct and cumulative_return_pct columns.

    Returns:
        Polars DataFrame with columns: trading_date, cashflow, total_assets, positions,
        daily_return_pct, cumulative_return_pct
    """
    df = get_all_portfolio_records(db, config_id)
    if len(df) == 0:
        return df
    if df["trading_date"].dtype in (pl.Utf8, pl.String):
        df = df.with_columns(pl.col("trading_date").str.to_datetime())
    initial_assets = df["total_assets"][0]
    return df.with_columns([
        ((pl.col("total_assets") / pl.col("total_assets").shift(1) - 1) * 100).alias("daily_return_pct"),
        ((pl.col("total_assets") / initial_assets - 1) * 100).alias("cumulative_return_pct"),
    ])


def get_all_decisions(db: SQLiteDB, config_id: str) -> pl.DataFrame:
    """
    Get all trading decisions for a config.

    Args:
        db: SQLiteDB instance
        config_id: Configuration ID

    Returns:
        Polars DataFrame with columns: id, trading_date, ticker, action, shares, price, justification
    """
    conn = db._get_connection()
    try:
        query = """
            SELECT
                d.id,
                d.trading_date,
                d.ticker,
                d.action,
                d.shares,
                d.price,
                d.justification
            FROM decision d
            JOIN portfolio p ON d.portfolio_id = p.id
            WHERE p.config_id = ?
            ORDER BY d.trading_date, d.ticker
        """
        cursor = conn.cursor()
        cursor.execute(query, (config_id,))
        columns = [desc[0] for desc in cursor.description]
        rows = cursor.fetchall()
        data = [{col: row[i] for i, col in enumerate(columns)} for row in rows]
        df = pl.from_dicts(data)
        return df
    finally:
        conn.close()


def get_all_signals(db: SQLiteDB, config_id: str) -> pl.DataFrame:
    """
    Get all analyst signals with portfolio context.

    Args:
        db: SQLiteDB instance
        config_id: Configuration ID

    Returns:
        Polars DataFrame with signal details
    """
    conn = db._get_connection()
    try:
        query = """
            SELECT
                s.id,
                p.trading_date,
                s.ticker,
                s.analyst,
                s.signal,
                s.justification
            FROM signal s
            JOIN portfolio p ON s.portfolio_id = p.id
            WHERE p.config_id = ?
            ORDER BY p.trading_date, s.ticker, s.analyst
        """
        cursor = conn.cursor()
        cursor.execute(query, (config_id,))
        columns = [desc[0] for desc in cursor.description]
        rows = cursor.fetchall()
        data = [{col: row[i] for i, col in enumerate(columns)} for row in rows]
        df = pl.from_dicts(data)
        return df
    finally:
        conn.close()


def get_decision_summary(db: SQLiteDB, config_id: str) -> pl.DataFrame:
    """
    Get aggregated summary of decisions by ticker and action.

    Args:
        db: SQLiteDB instance
        config_id: Configuration ID

    Returns:
        Polars DataFrame with summary statistics
    """
    conn = db._get_connection()
    try:
        query = """
            SELECT
                d.ticker,
                d.action,
                COUNT(*) as count,
                AVG(d.price) as avg_price
            FROM decision d
            JOIN portfolio p ON d.portfolio_id = p.id
            WHERE p.config_id = ?
            GROUP BY d.ticker, d.action
            ORDER BY d.ticker, d.action
        """
        cursor = conn.cursor()
        cursor.execute(query, (config_id,))
        columns = [desc[0] for desc in cursor.description]
        rows = cursor.fetchall()
        data = [{col: row[i] for i, col in enumerate(columns)} for row in rows]
        df = pl.from_dicts(data)
        return df
    finally:
        conn.close()


def get_signal_summary(db: SQLiteDB, config_id: str) -> pl.DataFrame:
    """
    Get aggregated summary of signals by analyst, ticker, and signal type.

    Args:
        db: SQLiteDB instance
        config_id: Configuration ID

    Returns:
        Polars DataFrame with signal counts
    """
    conn = db._get_connection()
    try:
        query = """
            SELECT
                s.analyst,
                s.ticker,
                s.signal,
                COUNT(*) as count
            FROM signal s
            JOIN portfolio p ON s.portfolio_id = p.id
            WHERE p.config_id = ?
            GROUP BY s.analyst, s.ticker, s.signal
            ORDER BY s.analyst, s.ticker, s.signal
        """
        cursor = conn.cursor()
        cursor.execute(query, (config_id,))
        columns = [desc[0] for desc in cursor.description]
        rows = cursor.fetchall()
        data = [{col: row[i] for i, col in enumerate(columns)} for row in rows]
        df = pl.from_dicts(data)
        return df
    finally:
        conn.close()


def check_data_quality(db: SQLiteDB, config_id: str) -> dict[str, Any]:
    """
    Check data quality for a config — counts decisions and signals per trading day.

    Args:
        db: SQLiteDB instance
        config_id: Configuration ID

    Returns:
        Dictionary with data quality metrics
    """
    conn = db._get_connection()
    try:
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT
                COUNT(DISTINCT p.id) as trading_days,
                COUNT(d.id) as total_decisions,
                MIN(p.trading_date) as first_date,
                MAX(p.trading_date) as last_date
            FROM portfolio p
            LEFT JOIN decision d ON d.portfolio_id = p.id
            WHERE p.config_id = ?
        """,
            (config_id,),
        )
        row = cursor.fetchone()

        cursor.execute(
            """
            SELECT COUNT(s.id) as total_signals
            FROM signal s
            JOIN portfolio p ON s.portfolio_id = p.id
            WHERE p.config_id = ?
        """,
            (config_id,),
        )
        sig_row = cursor.fetchone()

        return {
            "trading_days": row["trading_days"],
            "total_decisions": row["total_decisions"],
            "total_signals": sig_row["total_signals"],
            "first_date": row["first_date"],
            "last_date": row["last_date"],
        }
    finally:
        conn.close()


def get_experiment_metadata(db: SQLiteDB, config_id: str) -> dict[str, Any]:
    """
    Get experiment configuration and metadata.

    Args:
        db: SQLiteDB instance
        config_id: Configuration ID

    Returns:
        Dictionary with experiment details
    """
    # Get config
    config = db.get_config(config_id)
    if not config:
        return {}

    # Get date range from portfolio
    conn = db._get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT
                MIN(trading_date) as start_date,
                MAX(trading_date) as end_date,
                COUNT(*) as portfolio_count
            FROM portfolio
            WHERE config_id = ?
        """,
            (config_id,),
        )
        row = cursor.fetchone()

        return {
            "config_id": config_id,
            "exp_name": config["exp_name"],
            "tickers": config["tickers"],
            "llm_model": config["llm_model"],
            "llm_provider": config["llm_provider"],
            "has_planner": bool(config["has_planner"]),
            "start_date": datetime.fromisoformat(row["start_date"]) if row["start_date"] else None,
            "end_date": datetime.fromisoformat(row["end_date"]) if row["end_date"] else None,
            "portfolio_snapshots": row["portfolio_count"],
            "trading_days": (
                (datetime.fromisoformat(row["end_date"]) - datetime.fromisoformat(row["start_date"])).days + 1
                if row["start_date"] and row["end_date"]
                else 0
            ),
        }
    finally:
        conn.close()
