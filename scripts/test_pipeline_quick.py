"""Quick test of the ML pipeline with a small dataset."""

from pathlib import Path

import pytest

pytest.importorskip("yfinance")

if not (Path(__file__).resolve().parents[1] / "data" / "data_sp500" / "stock_data_long").exists():
    pytest.skip("SP500 cache is not present; skipping quick pipeline script", allow_module_level=True)

from decision_making.ml_model.feature_engineering import build_feature_matrix
from decision_making.models import RandomForestReturnModel
from decision_making.validation import run_walk_forward_validation

print("=" * 80)
print("QUICK PIPELINE TEST (Small Dataset)")
print("=" * 80)

# Use very high min_obs to get only a few stocks with very long histories
print("\n[1/3] Building feature matrix (min_obs=10000 for fewer stocks)...")
df = build_feature_matrix(min_obs=10000, lags=[1, 5], min_date="2010-01-01")

# Define features
print("\n[2/3] Defining features...")
exclude_cols = ["date", "ticker", "target"]
feature_cols = [col for col in df.columns if col not in exclude_cols]
print(f"  - Number of features: {len(feature_cols)}")
print(f"  - Features: {feature_cols}")

# Quick validation with just 2-3 folds
print("\n[3/3] Running walk-forward validation (2-3 folds)...")
print("  - Using 5-day embargo to prevent feature leakage")
results = run_walk_forward_validation(
    df=df,
    feature_cols=feature_cols,
    model_class=RandomForestReturnModel,
    model_params={
        "n_estimators": 50,  # Fewer trees for speed
        "max_depth": 5,  # Shallower for speed
        "random_state": 42,
    },
    initial_train_years=2,  # Shorter initial window
    step_months=12,  # Fewer folds
    test_months=3,
    embargo_days=5,  # Embargo equals max lag (5 in this test)
)

print("\n" + "=" * 80)
print("TEST COMPLETE")
print("=" * 80)
print(f"\nResults shape: {results.shape}")
print(f"\nMean accuracy: {results['accuracy'].mean():.4f}")
print(f"Mean AUC: {results['auc'].mean():.4f}")
print("\nPipeline test successful!")
