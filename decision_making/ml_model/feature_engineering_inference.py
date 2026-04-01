"""Feature engineering for single-stock inference."""

import math

import pandas as pd
import polars as pl
from scipy.stats import percentileofscore

from decision_making.agents.analysts.technical import _calculate_bollinger, _calculate_rsi


def build_single_stock_features(
    prices_df: pl.DataFrame,
    ticker: str,
    reference_data: dict,
    lags: list[int] | None = None,
) -> pd.DataFrame:
    """Build feature matrix for single stock matching training format.

    Args:
        prices_df: Polars DataFrame with ['date', 'prices'] from load_specific_data
        ticker: Stock ticker (e.g., 'TSLA')
        reference_data: Reference data from training with sectors, sector_map, feature_distributions
        lags: List of lag periods (default: [1, 5, 21])

    Returns:
        pandas DataFrame with 1 row and all features matching training format

    Raises:
        ValueError: If insufficient historical data
    """
    if lags is None:
        lags = [1, 5, 21]

    # Convert to pandas
    prices_pd = prices_df.to_pandas()
    prices_pd["date"] = pd.to_datetime(prices_pd["date"])
    prices_pd = prices_pd.sort_values("date")
    prices = prices_pd["prices"]

    # Check sufficient data
    min_required = max(lags) + 55
    if len(prices) < min_required:
        raise ValueError(f"Need {min_required} observations, have {len(prices)}")

    # Initialize features
    features = {"ticker": ticker}

    # 1. Lagged returns (cumulative)
    for lag in lags:
        price_t_minus_1 = prices.iloc[-1]
        price_t_minus_lag_plus_1 = prices.iloc[-(lag + 1)]
        features[f"return_lag_{lag}"] = (price_t_minus_1 - price_t_minus_lag_plus_1) / price_t_minus_lag_plus_1

    # 2. Technical indicators (using shared functions)
    rsi = _calculate_rsi(prices, period=14)
    bb_pos, _, _ = _calculate_bollinger(prices, window=20)
    returns = prices.pct_change()
    volatility = returns.rolling(21).std() * math.sqrt(252)

    features["rsi_14"] = rsi.iloc[-1]
    features["bb_position"] = bb_pos.iloc[-1]
    features["volatility_21d"] = volatility.iloc[-1]

    # 3. EMAs
    ema_8 = prices.ewm(span=8, adjust=False).mean()
    ema_21 = prices.ewm(span=21, adjust=False).mean()
    ema_55 = prices.ewm(span=55, adjust=False).mean()

    features["ema_8"] = ema_8.iloc[-1]
    features["ema_21"] = ema_21.iloc[-1]
    features["ema_55"] = ema_55.iloc[-1]
    features["ema_cross_short"] = int(ema_8.iloc[-1] > ema_21.iloc[-1])
    features["ema_cross_long"] = int(ema_21.iloc[-1] > ema_55.iloc[-1])

    # 4. Cross-sectional features
    sectors = reference_data.get("sectors", [])
    sector_map = reference_data.get("sector_map", {})
    ticker_sector = sector_map.get(ticker, "Technology")

    # Sector dummies
    for sector in sectors:
        features[f"sector_{sector}"] = int(sector == ticker_sector)

    # Percentile ranks using scipy
    feature_distributions = reference_data.get("feature_distributions", {})

    if "return_lag_1" in feature_distributions:
        dist = feature_distributions["return_lag_1"]
        percentiles = [float(v) for v in dist["percentiles"].values()]
        features["return_rank_pct"] = percentileofscore(percentiles, features["return_lag_1"]) / 100
    else:
        features["return_rank_pct"] = 0.5

    if "volatility_21d" in feature_distributions:
        dist = feature_distributions["volatility_21d"]
        percentiles = [float(v) for v in dist["percentiles"].values()]
        features["volatility_rank_pct"] = percentileofscore(percentiles, features["volatility_21d"]) / 100
    else:
        features["volatility_rank_pct"] = 0.5

    return pd.DataFrame([features])
