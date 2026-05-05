"""Central configuration for the ML return-prediction model."""

# Earliest date used as training data in the competition dataset and thus in the backtest.
# Rows before this date are excluded for the machine-learning model training.
COMPETITION_TRAIN_START = "2024-08-01"

# Lookback windows (in trading days) used to compute lagged return features.
# [1, 5, 21] ≈ yesterday, last week, last month.
# Wider lags capture slower momentum; narrower lags capture short-term mean-reversion.
LAGS = [1, 5, 21]

# Minimum number of historical observations required before the model will train.
# Below this threshold the ticker is considered to have insufficient history and the
# cross-sectional proxy is used instead.
MIN_OBS = 400

# Number of new trees added to the online random forest during each incremental update step.
# Higher values improve accuracy but increase update latency.
N_NEW_TREES_CROSS_SECTIONAL = 100

MODEL_PARAMS = {
    # Total number of decision trees in the random forest.
    # More trees reduce variance but increase training time.
    "n_estimators": 500,
    # Maximum depth of each tree. Deeper trees fit more complex patterns but overfit more easily.
    "max_depth": 5,
    # Minimum samples required to split an internal node.
    # Higher values act as regularisation, preventing the tree from memorising noise.
    "min_samples_split": 100,
    # Number of features considered at each split: "sqrt" ≈ sqrt(n_features).
    # Introduces diversity across trees, which is the core of bagging.
    "max_features": "sqrt",
    # Fixed seed for reproducibility across runs.
    "random_state": 42,
    # Parallel jobs for fitting trees. -1 uses all available cores; 4 is a safe default.
    "n_jobs": 4,
}

TRAIN_CONFIG = {
    # Mirrors MIN_OBS — passed into the walk-forward trainer so it skips folds with too few rows.
    "min_obs": MIN_OBS,
    # Mirrors LAGS — passed into feature engineering during walk-forward training.
    "lags": LAGS,
    # Number of calendar years of history used for the very first training fold.
    "initial_train_years": 7,
    # How far forward each walk-forward fold advances (in months).
    "step_months": 6,
    # Length of the out-of-sample test window for each fold (in months).
    "test_months": 3,
    # Trading days excluded between train end and test start to prevent leakage
    # from overlapping return labels (21 days ≈ 1 month).
    "embargo_days": 21,
    "model_params": MODEL_PARAMS,
}

# Features whose distributions are logged for monitoring.
FEATURE_DIST_COLS = ["return_lag_1", "return_lag_5", "return_lag_21", "volatility_21d"]
# Percentiles computed for each feature distribution column.
FEATURE_DIST_PERCENTILES = [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]

# Proxy tickers used by the ML model when the primary ticker lacks enough price history.
# Remove an entry once the ticker has accumulated sufficient observations.
ML_TICKER_PROXY: dict[str, str] = {
    # BTC has limited history in the competition dataset; TSLA is used as a stand-in
    # because both are high-volatility assets with similar cross-sectional behaviour.
    "BTC": "TSLA",
}

# Sector overrides for tickers not in the S&P 500 sector_map.
# BTC maps to "Financial Services" — the sector of bitcoin-exposed S&P 500 companies
# (COIN, RIOT, MARA), matching their yfinance classification.
EXTRA_TICKER_SECTORS: dict[str, str] = {
    "BTC": "Financial Services",
}
