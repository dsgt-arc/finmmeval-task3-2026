"""Central configuration for the ML return-prediction model.

All other modules import from here — never define these values elsewhere.
"""

COMPETITION_TRAIN_START = "2024-08-01"

LAGS = [1, 5, 21]
MIN_OBS = 400
N_NEW_TREES_CROSS_SECTIONAL = 100

MODEL_PARAMS = {
    "n_estimators": 500,
    "max_depth": 5,
    "min_samples_split": 100,
    "min_samples_leaf": 50,
    "random_state": 42,
    "n_jobs": 4,
}

TRAIN_CONFIG = {
    "min_obs": MIN_OBS,
    "lags": LAGS,
    "initial_train_years": 7,
    "step_months": 6,
    "test_months": 3,
    "embargo_days": 21,
    "model_params": MODEL_PARAMS,
}

FEATURE_DIST_COLS = ["return_lag_1", "return_lag_5", "return_lag_21", "volatility_21d"]
FEATURE_DIST_PERCENTILES = [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]
