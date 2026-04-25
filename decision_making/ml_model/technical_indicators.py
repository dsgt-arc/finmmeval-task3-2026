"""Shared technical indicator helpers used by both agents and ML feature engineering."""

import pandas as pd


def _calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Calculate Relative Strength Index.

    Args:
        prices: Price series
        period: RSI period (default: 14)

    Returns:
        RSI series (values from 0 to 100)
    """
    delta = prices.diff()
    gain = delta.where(delta > 0, 0).fillna(0)
    loss = -delta.where(delta < 0, 0).fillna(0)

    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()

    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))

    return rsi


def _calculate_bollinger(prices: pd.Series, window: int = 20) -> pd.Series:
    """Calculate position within Bollinger Bands.

    Args:
        prices: Price series
        window: Bollinger Band window (default: 20)

    Returns:
        Position series (0 = at lower band, 1 = at upper band, 0.5 = at middle)
    """
    sma = prices.rolling(window).mean()
    std_dev = prices.rolling(window).std()

    upper_band = sma + (std_dev * 2)
    lower_band = sma - (std_dev * 2)

    band_width = upper_band - lower_band
    position = (prices - lower_band) / (band_width + 1e-10)
    position = position.clip(0, 1)

    return position, upper_band, lower_band
