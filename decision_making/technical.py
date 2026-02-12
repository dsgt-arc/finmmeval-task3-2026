from datetime import datetime
import math

import pandas as pd
from sklearn.preprocessing import StandardScaler

from .signals import SignalNumerical as Signal

thresholds = {
    "trend": {
        "short": 8,
        "medium": 21,
        "long": 55,
    },
    "mean_reversion": {"bollinger_window": 20, "rolling_window": 50, "z_score_extreme": 2.0, "bb_position_threshold": 0.2},
    "rsi": {
        "period": 14,
        "bullish": 30,
        "bearish": 70,
    },
    "volatility": {
        "bullish": 0.8,
        "bearish": 1.2,
    },
    "volume": {
        "trend": 20,
        "correlation": 20,
        "unusual_volume": 2.0,
    },
    "support_resistance": {
        "pivot_window": 5,
        "lookback_period": 20,
    },
}

PRICE_COL = "prices"


def get_trend_signal(prices_df, params):
    """Advanced trend following strategy using multiple timeframes and indicators"""

    def _calculate_ema(prices_df, window):
        return prices_df[PRICE_COL].ewm(span=window, adjust=False).mean()

    # Calculate EMAs for multiple timeframes
    ema_short = _calculate_ema(prices_df, params["short"])
    ema_medium = _calculate_ema(prices_df, params["medium"])
    ema_long = _calculate_ema(prices_df, params["long"])

    # Determine trend direction and strength
    short_trend = ema_short > ema_medium
    medium_trend = ema_medium > ema_long

    if short_trend.iloc[-1] and medium_trend.iloc[-1]:
        signal = Signal.BULLISH
    elif not short_trend.iloc[-1] and not medium_trend.iloc[-1]:
        signal = Signal.BEARISH
    else:
        signal = Signal.NEUTRAL

    return signal


def get_mean_reversion_signal(prices_df, params):
    """Mean reversion strategy using statistical measures and Bollinger Bands"""

    def _calculate_bollinger_bands(prices_df: pd.DataFrame, window: int) -> tuple[pd.Series, pd.Series]:
        sma = prices_df[PRICE_COL].rolling(window).mean()
        std_dev = prices_df[PRICE_COL].rolling(window).std()
        upper_band = sma + (std_dev * 2)
        lower_band = sma - (std_dev * 2)
        return upper_band, lower_band

    # Calculate Bollinger Bands with configured window
    bb_upper, bb_lower = _calculate_bollinger_bands(prices_df, params["bollinger_window"])

    # Calculate z-score with configured rolling window
    ma = prices_df[PRICE_COL].rolling(window=params["rolling_window"]).mean()
    std = prices_df[PRICE_COL].rolling(window=params["rolling_window"]).std()
    z_score = (prices_df[PRICE_COL] - ma) / std

    # Calculate normalized position within Bollinger Bands
    price_vs_bb = (prices_df[PRICE_COL].iloc[-1] - bb_lower.iloc[-1]) / (bb_upper.iloc[-1] - bb_lower.iloc[-1])

    # Use threshold values for signal conditions
    if z_score.iloc[-1] < params["z_score_extreme"] and price_vs_bb < params["bb_position_threshold"]:
        signal = Signal.BULLISH
    elif z_score.iloc[-1] > params["z_score_extreme"] and price_vs_bb > (1 - params["bb_position_threshold"]):
        signal = Signal.BEARISH
    else:
        signal = Signal.NEUTRAL

    return signal


def get_rsi_signal(prices_df, params):
    """RSI signal that indicate overbought/oversold conditions"""

    def _calculate_rsi(prices_df: pd.DataFrame, period: int) -> pd.Series:
        delta = prices_df[PRICE_COL].diff()
        gain = (delta.where(delta > 0, 0)).fillna(0)
        loss = (-delta.where(delta < 0, 0)).fillna(0)
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    rsi = _calculate_rsi(prices_df, params["period"])
    if rsi.iloc[-1] > params["bearish"]:
        signal = Signal.BEARISH
    elif rsi.iloc[-1] < params["bullish"]:
        signal = Signal.BULLISH
    else:
        signal = Signal.NEUTRAL

    return signal


def get_volatility_signal(prices_df, params):
    """Volatility-based trading strategy"""
    # Calculate various volatility metrics
    returns = prices_df[PRICE_COL].pct_change()

    # Historical volatility
    hist_vol = returns.rolling(21).std() * math.sqrt(252)

    # Volatility regime detection
    vol_ma = hist_vol.rolling(63).mean()
    vol_regime = hist_vol / vol_ma

    # Volatility mean reversion
    vol_z_score = (hist_vol - vol_ma) / hist_vol.rolling(63).std()

    # Generate signal based on volatility regime
    current_vol_regime = vol_regime.iloc[-1]
    vol_z = vol_z_score.iloc[-1]

    if current_vol_regime < params["bullish"] and vol_z < -1:
        # Low vol regime, potential for expansion
        signal = Signal.BULLISH
    elif current_vol_regime > params["bearish"] and vol_z > 1:
        # High vol regime, potential for contraction
        signal = Signal.BEARISH
    else:
        signal = Signal.NEUTRAL

    return signal


def get_support_resistance(prices_df, params):
    """Calculate support and resistance levels"""

    def _is_level(prices: pd.Series, i: int, level_type: str, pivot_window: int = params["pivot_window"]) -> bool:
        """Check if the price point is a support/resistance level by comparing with surrounding prices"""
        start_idx = max(0, i - pivot_window)
        end_idx = min(len(prices), i + pivot_window + 1)
        window_prices = prices.iloc[start_idx:end_idx]
        current_price = prices.iloc[i]

        left_prices = window_prices.iloc[:pivot_window]
        right_prices = window_prices.iloc[pivot_window + 1 :]

        if level_type == "support":
            return len(left_prices[left_prices > current_price]) >= 2 and len(right_prices[right_prices > current_price]) >= 2
        elif level_type == "resistance":
            return len(left_prices[left_prices < current_price]) >= 2 and len(right_prices[right_prices < current_price]) >= 2
        # else:
        #     raise ValueError("level_type must be 'support' or 'resistance'")

    def _find_levels(prices: pd.Series, lookback_period: int = params["lookback_period"]):
        levels = []
        for i in range(lookback_period, len(prices)):
            if _is_level(prices, i, "support") or _is_level(prices, i, "resistance"):
                levels.append((i, prices.iloc[i]))
        return levels

    price_data = prices_df[PRICE_COL]
    current_price = price_data.iloc[-1]
    levels = _find_levels(price_data)

    support_levels = [price for _, price in levels if price < current_price]
    resistance_levels = [price for _, price in levels if price > current_price]

    support = max(support_levels) if support_levels else None
    resistance = min(resistance_levels) if resistance_levels else None

    if support is None or resistance is None:
        return "Failed to analyze support and resistance levels"
    else:
        result = f"- Current price: {current_price}\n"
        result += f"- Nearest support: {support}\n"
        result += f"- Nearest resistance: {resistance}\n"
        result += f"- Price to support: {(current_price - support) / support}\n"
        result += f"- Price to resistance: {(resistance - current_price) / current_price}\n"
        return result


def get_technical_analysis(prices_df: pd.DataFrame) -> dict[str, Signal]:
    """Comprehensive technical analysis function that aggregates multiple indicators"""

    # Analyze technical indicators
    signal_results = {
        "trend": get_trend_signal(prices_df, thresholds["trend"]),
        "mean_reversion": get_mean_reversion_signal(prices_df, thresholds["mean_reversion"]),
        "rsi": get_rsi_signal(prices_df, thresholds["rsi"]),
        "volatility": get_volatility_signal(prices_df, thresholds["volatility"]),
        "price_levels": get_support_resistance(prices_df, thresholds["support_resistance"]),
    }

    return signal_results


def get_technical_analyses(prices_df: pd.DataFrame) -> dict[str, Signal]:
    """Wrapper function to get technical analyses"""
    date_range = prices_df["date"].to_list()

    signals = {}
    # Iterate through df and collect signals
    for date in date_range:
        subset_df = prices_df[prices_df["date"] <= date]
        signals[date] = get_technical_analysis(subset_df)
    return signals


def transform_signal_to_df(signal: dict[datetime.datetime, dict[str, Signal]]) -> pd.DataFrame:
    """Standardize technical signals into a numerical feature matrix"""
    signal_mapping = {
        Signal.BULLISH: 1,
        Signal.BEARISH: -1,
        Signal.NEUTRAL: 0,
    }

    # Convert signals to numerical values
    numerical_signals = {key: signal_mapping.get(value, 0) for key, value in signal.items() if key != "price_levels"}

    # Create DataFrame
    signal_df = pd.DataFrame([numerical_signals])

    return signal_df


def standardize_technical_signals(
    technical_analyses: dict[datetime.datetime, dict[str, Signal]], feature_cols: list[str]
) -> pd.DataFrame:
    """Standardize multiple technical analyses into a numerical feature matrix"""
    all_signals = []
    for date, analysis in technical_analyses.items():
        standardized_df = transform_signal_to_df(analysis)
        standardized_df["date"] = date
        all_signals.append(standardized_df)

    final_df = pd.concat(all_signals).reset_index(drop=True)

    scaler = StandardScaler()
    final_df[feature_cols] = scaler.fit_transform(final_df[feature_cols])

    return final_df
