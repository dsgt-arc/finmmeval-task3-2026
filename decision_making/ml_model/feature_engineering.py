"""Feature engineering for cross-sectional stock return prediction.

This module transforms wide-format returns data into a long-format feature matrix
suitable for machine learning models. It includes lagged returns, technical indicators,
and cross-sectional features.
"""

import math

import numpy as np
import pandas as pd

from ..sp500_data import load_single_stocks, load_stock_metric_long, load_stock_sector
from .technical_indicators import _calculate_bollinger, _calculate_rsi


def create_lagged_returns(prices_wide: pd.DataFrame, lags: list[int]) -> pd.DataFrame:
    """Convert wide prices to long format with lagged return features.

    Args:
        prices_wide: DataFrame with date index and ticker columns containing prices
        lags: List of lag periods (e.g., [1, 5, 21] for 1-day, 5-day, 21-day returns)

    Returns:
        Long-format DataFrame with columns:
            - date: observation date
            - ticker: stock ticker
            - return_lag_1: 1-day return from t-1 to t-2
            - return_lag_5: 5-day cumulative return from t-1 to t-6
            - return_lag_21: 21-day cumulative return from t-1 to t-22
            - target: next-day return (for prediction)

    Note:
        Multi-period returns are computed as cumulative returns over the period,
        not single-day returns from N days ago. This captures momentum more accurately.

    Example:
        >>> prices_wide = load_stock_metric_long("Adj_Close", min_obs=400)
        >>> features_df = create_lagged_returns(prices_wide, lags=[1, 5])
        >>> features_df.head()
    """
    # List to store results for each ticker
    all_data = []

    for ticker in prices_wide.columns:
        # Get price series for this ticker
        prices_series = prices_wide[ticker].copy()

        # Create a DataFrame with the ticker's data
        ticker_df = pd.DataFrame({"date": prices_series.index, "ticker": ticker})

        # Create lagged return features (multi-period cumulative returns)
        for lag in lags:
            # Compute return from t-1 to t-(lag+1)
            # This gives us the cumulative return over the lag period
            price_t_minus_1 = prices_series.shift(1)
            price_t_minus_lag_plus_1 = prices_series.shift(lag + 1)
            ticker_df[f"return_lag_{lag}"] = ((price_t_minus_1 - price_t_minus_lag_plus_1) / price_t_minus_lag_plus_1).values

        # Create target (next-day return: from t to t+1)
        price_t = prices_series
        price_t_plus_1 = prices_series.shift(-1)
        ticker_df["target"] = ((price_t_plus_1 - price_t) / price_t).values

        all_data.append(ticker_df)

    # Concatenate all tickers
    result_df = pd.concat(all_data, ignore_index=True)

    # Drop rows with NaN values (from lags and target shift)
    result_df = result_df.dropna()

    return result_df


def compute_technical_indicators_panel(returns_wide: pd.DataFrame, adj_close_wide: pd.DataFrame) -> pd.DataFrame:
    """Compute technical indicators for all stocks in panel data.

    Args:
        returns_wide: Returns DataFrame (dates x tickers)
        adj_close_wide: Adjusted close prices DataFrame (dates x tickers)

    Returns:
        Long-format DataFrame with columns:
            - date, ticker
            - rsi_14: 14-day RSI
            - bb_position: position within Bollinger Bands (0-1)
            - volatility_21d: 21-day rolling volatility (annualized)
            - ema_8, ema_21, ema_55: exponential moving averages
            - ema_cross_short: boolean if ema_8 > ema_21
            - ema_cross_long: boolean if ema_21 > ema_55

    Note:
        Stocks with < 50 valid observations are skipped.
    """
    all_indicators = []

    for ticker in adj_close_wide.columns:
        # Get price and return series
        prices = adj_close_wide[ticker].dropna()

        # Skip if insufficient data
        if len(prices) < 50:
            continue

        # Create DataFrame for this ticker
        ticker_df = pd.DataFrame({"date": prices.index, "ticker": ticker})

        # Calculate RSI
        ticker_df["rsi_14"] = _calculate_rsi(prices, period=14).values

        # Calculate Bollinger Bands position
        bb_position, _, _ = _calculate_bollinger(prices, window=20)
        ticker_df["bb_position"] = bb_position.values

        # Calculate volatility (21-day rolling, annualized)
        if ticker in returns_wide.columns:
            returns = returns_wide[ticker]
            volatility = returns.rolling(21).std() * math.sqrt(252)
            # Align with prices index
            volatility_aligned = volatility.reindex(prices.index)
            ticker_df["volatility_21d"] = volatility_aligned.values
        else:
            ticker_df["volatility_21d"] = np.nan

        # Calculate EMAs
        ema_8 = prices.ewm(span=8, adjust=False).mean()
        ema_21 = prices.ewm(span=21, adjust=False).mean()
        ema_55 = prices.ewm(span=55, adjust=False).mean()

        ticker_df["ema_8"] = ema_8.values
        ticker_df["ema_21"] = ema_21.values
        ticker_df["ema_55"] = ema_55.values

        # Create EMA crossover signals
        ticker_df["ema_cross_short"] = (ema_8 > ema_21).astype(int).values
        ticker_df["ema_cross_long"] = (ema_21 > ema_55).astype(int).values

        all_indicators.append(ticker_df)

    # Concatenate all tickers
    if not all_indicators:
        raise ValueError("No valid tickers found for technical indicators")

    result_df = pd.concat(all_indicators, ignore_index=True)

    # Drop rows with NaN (from rolling windows)
    result_df = result_df.dropna()

    return result_df


def add_cross_sectional_features(features_long: pd.DataFrame) -> pd.DataFrame:
    """Add cross-sectional features like sector dummies and percentile ranks.

    Args:
        features_long: Long-format feature DataFrame with date, ticker columns

    Returns:
        Enhanced DataFrame with additional columns:
            - sector_*: one-hot encoded sector dummies
            - return_rank_pct: percentile rank of lagged returns vs peers (within date)
            - volatility_rank_pct: percentile rank of volatility vs peers (within date)

    Note:
        Requires 'return_lag_1' and 'volatility_21d' columns in input.
    """
    # Load sector mapping
    try:
        sector_df = load_stock_sector(min_obs=400)

        # Create a mapping dict: ticker -> sector
        sector_map = {}
        for ticker in sector_df.columns:
            sector_values = sector_df[ticker].dropna()
            if len(sector_values) > 0:
                sector_map[ticker] = sector_values.iloc[0]

        # Add sector column
        features_long["sector"] = features_long["ticker"].map(sector_map)

        # Fill missing sectors
        features_long["sector"] = features_long["sector"].fillna("Unknown")

        # One-hot encode sectors
        sector_dummies = pd.get_dummies(features_long["sector"], prefix="sector")
        features_long = pd.concat([features_long, sector_dummies], axis=1)

        # Drop the original sector column (keep dummies)
        features_long = features_long.drop(columns=["sector"])

    except Exception as e:
        print(f"Warning: Could not load sector data: {e}")
        print("Continuing without sector features.")

    # Add percentile ranks within each date
    if "return_lag_1" in features_long.columns:
        features_long["return_rank_pct"] = features_long.groupby("date")["return_lag_1"].rank(pct=True)

    if "volatility_21d" in features_long.columns:
        features_long["volatility_rank_pct"] = features_long.groupby("date")["volatility_21d"].rank(pct=True)

    return features_long


def build_feature_matrix(
    lags: list[int],
    min_obs: int = 400,
    min_date: str | None = "2001-01-01",
) -> pd.DataFrame:
    """Build complete feature matrix for ML model.

    This is the main function that orchestrates all feature engineering steps:
    1. Load returns and prices
    2. Create lagged returns
    3. Compute technical indicators
    4. Merge features
    5. Add cross-sectional features
    6. Create binary target
    7. Clean and filter data

    Args:
        min_obs: Minimum observations per stock (passed to data loaders)
        lags: List of lag periods for lagged returns
        min_date: Minimum date to include (to ensure complete rolling windows)

    Returns:
        Complete feature matrix with columns:
            - date, ticker (identifiers)
            - Lagged returns, technical indicators, cross-sectional features
            - target: binary (1 if next-day return > 0, else 0)

    Example:
        >>> df = build_feature_matrix(min_obs=400, lags=[1, 5, 21])
        >>> print(df.shape)
        >>> print(df.columns.tolist())
        >>> print(df["target"].value_counts())
    """
    print("Building feature matrix...")

    # Step 1: Load data
    print(f"  [1/6] Loading adjusted close prices (min_obs={min_obs})...")
    adj_close_wide = load_stock_metric_long("Adj_Close", min_obs=min_obs)
    # Convert to numeric (handle any string values)
    adj_close_wide = adj_close_wide.apply(pd.to_numeric, errors="coerce")
    print(f"    - Loaded {len(adj_close_wide.columns)} tickers")
    print(f"    - Date range: {adj_close_wide.index.min()} to {adj_close_wide.index.max()}")

    print("  [2/6] Loading returns for volatility calculation...")
    returns_wide = load_single_stocks(min_obs=min_obs)
    print(f"    - Loaded {len(returns_wide.columns)} tickers")

    # Step 2: Create lagged returns from prices (multi-period cumulative returns)
    print(f"  [3/6] Creating lagged returns from prices (lags={lags})...")
    lagged_df = create_lagged_returns(adj_close_wide, lags=lags)
    print(f"    - Created {len(lagged_df)} stock-date observations")

    # Step 3: Compute technical indicators
    print("  [4/6] Computing technical indicators...")
    technical_df = compute_technical_indicators_panel(returns_wide, adj_close_wide)
    print(f"    - Computed indicators for {len(technical_df)} stock-date observations")

    # Step 4: Merge features
    print("  [5/6] Merging features...")
    # Merge on date and ticker
    features_df = pd.merge(lagged_df, technical_df, on=["date", "ticker"], how="inner")
    print(f"    - Merged dataset: {len(features_df)} observations")

    # Step 5: Add cross-sectional features
    print("  [6/6] Adding cross-sectional features...")
    features_df = add_cross_sectional_features(features_df)

    # Create binary target (1 if positive return, 0 otherwise)
    features_df["target"] = (features_df["target"] > 0).astype(int)

    # Filter by minimum date if specified
    if min_date is not None:
        features_df["date"] = pd.to_datetime(features_df["date"])
        min_date_parsed = pd.to_datetime(min_date)
        before_filter = len(features_df)
        features_df = features_df[features_df["date"] >= min_date_parsed]
        print(f"    - Filtered to dates >= {min_date}: {before_filter} -> {len(features_df)} observations")

    # Final cleanup: drop any remaining NaNs
    before_dropna = len(features_df)
    features_df = features_df.dropna()
    dropped = before_dropna - len(features_df)
    if dropped > 0:
        print(f"    - Dropped {dropped} rows with NaN values")

    # Summary
    print("\nFeature matrix complete:")
    print(f"  - Shape: {features_df.shape}")
    print(f"  - Date range: {features_df['date'].min()} to {features_df['date'].max()}")
    print(f"  - Unique tickers: {features_df['ticker'].nunique()}")
    print(f"  - Target distribution: {features_df['target'].value_counts().to_dict()}")

    return features_df
