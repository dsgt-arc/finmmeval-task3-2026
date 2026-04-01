import datetime

from graph.constants import AgentKey, Signal
from graph.schema import AnalystSignal, FundState
from util.db_helper import get_db
from util.logger import logger

from decision_making.ama_data import load_specific_data
from decision_making.ml_model.feature_engineering_inference import build_single_stock_features
from decision_making.ml_model.ml_model_manager import get_model_manager

"""ML analyst using Random Forest trained on SP500 data."""


def ml_model_agent(state: FundState):
    """ML analyst predicting return direction with Random Forest.

    Args:
        state: FundState with ticker, trading_date, portfolio

    Returns:
        Dict with analyst_signals list
    """
    ticker = state["ticker"]
    trading_date = state["trading_date"]

    try:
        # Load model manager (cached after first call)
        manager = get_model_manager()

        # Get price data and engineer features
        prices_df = load_specific_data(ticker, trading_date - datetime.timedelta(days=1), type="price")
        features = build_single_stock_features(prices_df, ticker, manager.reference_data)

        # Predict
        X = features[manager.metadata["feature_names"]].values.reshape(1, -1)
        proba = manager.predict(X)[0, 1]

        # Convert to signal
        if proba > 0.55:
            signal = Signal.BULLISH
        elif proba < 0.45:
            signal = Signal.BEARISH
        else:
            signal = Signal.NEUTRAL

        justification = f"ML predicts {proba:.1%} prob of positive return"

        # Schedule observation for online learning
        manager.add_observation(features=X, target=None)

    except Exception as e:
        logger.warning(f"ML model failed for {ticker}: {e}")
        signal = Signal.NEUTRAL
        justification = "ML model unavailable"

    # Save and return signal
    analyst_signal = AnalystSignal(signal=signal, justification=justification)
    get_db().save_signal(state["portfolio"].id, AgentKey.ML_MODEL, ticker, "ML", analyst_signal)
    return {"analyst_signals": [analyst_signal]}
