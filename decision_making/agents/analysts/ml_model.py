import datetime

from graph.constants import AgentKey, Signal
from graph.schema import AnalystSignal, FundState
from llm.cost_estimation import estimate_cost
from llm.inference import agent_call
from llm.prompt import ML_MODEL_PROMPT
import polars as pl
from util.db_helper import get_db
from util.logger import logger

from decision_making.ama_data import load_specific_data
from decision_making.ml_model.config import ML_TICKER_PROXY, N_NEW_TREES_CROSS_SECTIONAL
from decision_making.ml_model.feature_engineering_inference import (
    build_cross_sectional_features_for_date,
    build_single_stock_features,
)

"""ML analyst using Random Forest trained on SP500 data."""


def _prev_trading_day(date: datetime.date) -> datetime.datetime:
    """Return the most recent NYSE trading day strictly before date."""
    import pandas_market_calendars as mcal
    nyse = mcal.get_calendar("NYSE")
    # Look back up to 10 calendar days to find the previous session
    start = (date - datetime.timedelta(days=10)).strftime("%Y-%m-%d")
    end = (date - datetime.timedelta(days=1)).strftime("%Y-%m-%d")
    schedule = nyse.schedule(start_date=start, end_date=end)
    return datetime.datetime.combine(schedule.index[-1].date(), datetime.time.min)


def _load_prices_for_inference(ticker: str, date, adj_close_wide) -> pl.DataFrame:
    """Load price history for a ticker, merging yfinance + competition data.

    For tickers present in both sources (e.g. TSLA), concatenates the full
    yfinance history with the competition dataset, preferring competition prices
    on overlapping dates. For tickers only in the competition dataset (e.g. BTC),
    returns competition data as-is.

    Args:
        ticker: Stock symbol
        date: Latest date to include (passed to load_specific_data)
        adj_close_wide: Wide-format SP500 adj close DataFrame (or None)

    Returns:
        Polars DataFrame with columns ['date', 'prices'], sorted ascending
    """
    comp_df = load_specific_data(ticker, date, type="price")  # Polars ['date', 'prices']

    if adj_close_wide is None or ticker not in adj_close_wide.columns:
        return comp_df

    # Build Polars DF from yfinance wide cache
    yf_series = adj_close_wide[ticker].dropna()
    yf_pd = yf_series.reset_index()
    yf_pd.columns = ["date", "prices"]
    yf_pd["date"] = yf_pd["date"].dt.date
    yf_pl = pl.from_pandas(yf_pd).with_columns(pl.col("date").cast(pl.Date))

    # Ensure competition df date column is pl.Date
    comp_df = comp_df.with_columns(pl.col("date").cast(pl.Date))

    # Concatenate; competition data wins on duplicate dates (keep="last" after sort)
    combined = pl.concat([yf_pl, comp_df]).unique(subset=["date"], keep="last").sort("date")
    return combined


def ml_model_agent_online(state: FundState):
    """ML analyst with cross-sectional online learning.

    For each trading day:
    1. Loads full SP500 cross-section for the previous day (targets now known)
    2. Online-learns the model from the cross-sectional batch
    3. Predicts return direction for the competition ticker

    Args:
        state: FundState with ticker, trading_date, portfolio

    Returns:
        Dict with analyst_signals list
    """
    # Import the model manager lazily so workflows that do not use this analyst
    # do not need the optional SP500/yfinance stack at startup.
    from decision_making.ml_model.ml_model_manager import get_model_manager

    agent_name = AgentKey.ML_MODEL_ONLINE
    ticker = state["ticker"]
    trading_date = state["trading_date"]
    llm_config = state["llm_config"]
    portfolio_id = state["portfolio_id"]
    prev_date = _prev_trading_day(trading_date)

    # Get db instance
    db = get_db()

    manager = get_model_manager()
    feature_names = manager.metadata["feature_names"]
    last_train_date = manager.reference_data["last_obs_date"]
    last_train_date = datetime.datetime.strptime(last_train_date, "%Y-%m-%d")
    # Always load price_data — needed for Step 2 inference even when cross-sectional
    # learning is skipped (e.g. second ticker in the same run).
    try:
        price_data = manager.get_data(through_date=prev_date)
    except Exception as e:
        logger.warning(f"Failed to load SP500 price data: {e}")
        price_data = None

    # --- Step 1: Cross-sectional online learning from previous day ---
    if last_train_date >= trading_date:
        logger.info(f"Model already trained through {trading_date}, skipping online learning")
    elif price_data is not None:
        try:

            # Check that the previous day falls within the SP500 data range
            available_dates = price_data.index[price_data.index <= prev_date]
            if len(available_dates) >= 2:
                t_minus_1 = available_dates[-1]
                t_minus_2 = available_dates[-2]

                # Build cross-sectional features for prev_date
                cs_df = build_cross_sectional_features_for_date(
                    date=prev_date,
                    adj_close_wide=price_data,
                    sectors=manager.reference_data.get("sectors", []),
                    sector_map=manager.reference_data.get("sector_map", {}),
                )

                if not cs_df.empty:
                    # Compute targets: sign of return on t_minus_1 (now known)
                    prices_t1 = price_data.loc[t_minus_1]
                    prices_t2 = price_data.loc[t_minus_2]
                    day_returns = (prices_t1 - prices_t2) / prices_t2
                    targets = (day_returns > 0).astype(int)

                    cs_df["target"] = cs_df["ticker"].map(targets)
                    cs_df = cs_df.dropna(subset=["target"])

                    # Fill any feature columns missing from inference (e.g. sector_Unknown)
                    for col in feature_names:
                        if col not in cs_df.columns:
                            cs_df[col] = 0

                    # Select only columns present in the trained model
                    valid_rows = cs_df[feature_names].notna().all(axis=1)
                    cs_df = cs_df[valid_rows]

                    if not cs_df.empty:
                        X_batch = cs_df[feature_names].values
                        y_batch = cs_df["target"].values.astype(int)
                        manager.cross_sectional_retrain(X_batch, y_batch, n_new_trees=N_NEW_TREES_CROSS_SECTIONAL)
                        # Mark this trading date as trained so other tickers in the same
                        # run skip cross-sectional learning (avoids duplicate retraining).
                        manager.reference_data["last_obs_date"] = trading_date.strftime("%Y-%m-%d")
                        logger.info(f"Online-learned from {len(X_batch)} SP500 stocks for {prev_date}")
            else:
                logger.warning(f"SP500 data does not cover {prev_date}, skipping cross-sectional learning")

        except Exception as e:
            logger.warning(f"Cross-sectional online learning failed: {e}")

    # --- Step 2: Predict for competition ticker ---
    prompt = ""
    try:
        prices_df = _load_prices_for_inference(ticker, prev_date, price_data)
        try:
            features = build_single_stock_features(prices_df, ticker, manager.reference_data)
        except ValueError:
            proxy = ML_TICKER_PROXY.get(ticker)
            if proxy is None:
                raise
            logger.info(f"[{agent_name}] Insufficient obs for {ticker}, using proxy {proxy}")
            proxy_prices_df = _load_prices_for_inference(proxy, prev_date, price_data)
            features = build_single_stock_features(proxy_prices_df, proxy, manager.reference_data)

        # Fill any feature columns missing from inference (e.g. sector_Unknown)
        for col in feature_names:
            if col not in features.columns:
                features[col] = 0

        X = features[feature_names].values.reshape(1, -1)
        proba = manager.predict(X)[0, 1]

        prompt = ML_MODEL_PROMPT.format(ticker=ticker, proba=proba, trading_date=trading_date)

        # Estimate and log cost before making the call
        costs = estimate_cost(prompt, llm_config)
        logger.info(f"[{agent_name}] Estimated cost for {ticker}: ${costs:.6f}")

        signal = agent_call(prompt=prompt, llm_config=llm_config, pydantic_model=AnalystSignal)

    except Exception as e:
        logger.warning(f"ML model failed for {ticker}: {e}")
        signal = AnalystSignal(signal=Signal.NEUTRAL, justification="ML model unavailable")

    # save signal
    logger.log_signal(agent_name, ticker, signal)
    db.save_signal(portfolio_id, agent_name, ticker, prompt, signal)

    return {"analyst_signals": [signal]}
