"""
SQL query functions for extracting data from the DeepFund SQLite database.
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


def get_all_decisions(db: SQLiteDB, config_id: str) -> pl.DataFrame:
    """
    Get all trading decisions with portfolio context.

    Args:
        db: SQLiteDB instance
        config_id: Configuration ID

    Returns:
        Polars DataFrame with decision details and portfolio context
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
                d.justification,
                p.total_assets as portfolio_value_at_decision,
                p.cashflow as cashflow_at_decision
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
                SUM(d.shares) as total_shares,
                AVG(d.price) as avg_price,
                SUM(d.shares * d.price) as total_value
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
    Check for data quality issues in the database.

    Args:
        db: SQLiteDB instance
        config_id: Configuration ID

    Returns:
        Dictionary with data quality metrics
    """
    conn = db._get_connection()
    try:
        cursor = conn.cursor()

        # Check for corrupted records
        cursor.execute(
            """
            SELECT
                COUNT(*) as total_records,
                SUM(CASE WHEN total_assets <= 0 THEN 1 ELSE 0 END) as corrupted_records,
                MIN(CASE WHEN total_assets > 0 THEN trading_date END) as first_valid_date,
                MAX(CASE WHEN total_assets > 0 THEN trading_date END) as last_valid_date,
                MIN(CASE WHEN total_assets <= 0 THEN trading_date END) as first_corrupted_date
            FROM portfolio
            WHERE config_id = ?
        """,
            (config_id,),
        )
        row = cursor.fetchone()

        return {
            "total_records": row["total_records"],
            "valid_records": row["total_records"] - row["corrupted_records"],
            "corrupted_records": row["corrupted_records"],
            "first_valid_date": row["first_valid_date"],
            "last_valid_date": row["last_valid_date"],
            "first_corrupted_date": row["first_corrupted_date"],
            "corruption_percentage": (row["corrupted_records"] / row["total_records"] * 100) if row["total_records"] > 0 else 0,
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
