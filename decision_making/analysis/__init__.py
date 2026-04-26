"""
DeepFund trading performance analysis utilities.

This package provides tools for analyzing market timing results from the SQLite database,
including timing performance, hit rate metrics, and benchmark comparisons.
"""

from decision_making.analysis.performance import (
    calculate_annualized_return,
    calculate_buy_hold_benchmark,
    calculate_decision_hit_rate,
    calculate_max_drawdown,
    calculate_portfolio_benchmark,
    calculate_sharpe_ratio,
    calculate_signal_hit_rate,
    calculate_timing_performance,
)
from decision_making.analysis.queries import (
    check_data_quality,
    get_all_decisions,
    get_all_portfolio_records,
    get_all_signals,
    get_decision_summary,
    get_experiment_metadata,
    get_signal_summary,
)

__all__ = [
    # Performance metrics
    "calculate_annualized_return",
    "calculate_buy_hold_benchmark",
    "calculate_decision_hit_rate",
    "calculate_max_drawdown",
    "calculate_portfolio_benchmark",
    "calculate_sharpe_ratio",
    "calculate_signal_hit_rate",
    "calculate_timing_performance",
    # Query functions
    "check_data_quality",
    "get_all_decisions",
    "get_all_portfolio_records",
    "get_all_signals",
    "get_decision_summary",
    "get_experiment_metadata",
    "get_signal_summary",
]
