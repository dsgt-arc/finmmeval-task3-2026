"""Walk-forward validation framework for time-series machine learning.

This module implements expanding window walk-forward validation, which is critical
for evaluating models on time-series data without look-ahead bias. The training
window expands over time while the test window moves forward.
"""

from datetime import timedelta
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def walk_forward_split(
    df: pd.DataFrame,
    initial_train_years: int = 3,
    step_months: int = 6,
    test_months: int = 3,
    embargo_days: int = 0,
) -> list[dict]:
    """Generate walk-forward train/test splits preserving temporal order.

    Uses an expanding window approach where the training data accumulates over time
    while the test window moves forward. This mimics real-world deployment where
    models are trained on all historical data.

    Args:
        df: DataFrame with 'date' column
        initial_train_years: Initial training period in years (default: 3)
        step_months: How often to retrain in months (default: 6)
        test_months: Test period length in months (default: 3)
        embargo_days: Gap between train and test to prevent leakage from lagged features
                     (default: 0). Recommended: set to max lag period (e.g., 21 for 21-day lags)

    Returns:
        List of dicts, each containing:
            - train_start, train_end: training period dates
            - test_start, test_end: testing period dates
            - train_idx: boolean index for training rows
            - test_idx: boolean index for testing rows
            - fold_num: fold number (0-indexed)

    Note:
        The embargo period creates a gap to prevent data leakage. If your features include
        21-day lagged returns, use embargo_days=21 to ensure test features don't overlap
        with training observations.

    Example:
        >>> splits = walk_forward_split(df, initial_train_years=3, step_months=6, test_months=3, embargo_days=21)
        >>> print(f"Generated {len(splits)} folds")
        >>> # Fold 1: Train [2000, 2003-01-01], Embargo [2003-01-01, 2003-01-22],
        >>> #         Test [2003-01-22 to 2003-04-22]
    """
    # Ensure date column is datetime
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    min_date = df["date"].min()
    max_date = df["date"].max()

    # Calculate initial train end date
    train_start = min_date
    train_end = train_start + timedelta(days=365 * initial_train_years)

    # Initialize splits list
    splits = []
    fold_num = 0

    while True:
        # Calculate test period with embargo
        embargo_end = train_end + timedelta(days=embargo_days)
        test_start = embargo_end
        test_end = test_start + timedelta(days=30 * test_months)

        # Check if we've gone beyond available data
        if test_end > max_date:
            break

        # Create boolean indices (embargo period excluded from both train and test)
        train_idx = (df["date"] >= train_start) & (df["date"] < train_end)
        test_idx = (df["date"] >= test_start) & (df["date"] < test_end)

        # Only add split if both train and test sets are non-empty
        if train_idx.sum() > 0 and test_idx.sum() > 0:
            splits.append({
                "fold_num": fold_num,
                "train_start": train_start,
                "train_end": train_end,
                "test_start": test_start,
                "test_end": test_end,
                "train_idx": train_idx,
                "test_idx": test_idx,
            })
            fold_num += 1

        # Expand training window (keep same start, move end forward)
        train_end = train_end + timedelta(days=30 * step_months)

    return splits


def evaluate_fold(model: Any, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """Train and evaluate model on a single fold.

    Args:
        model: Instance of BaseModel (e.g., RandomForestReturnModel)
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels

    Returns:
        Dict with metrics:
            - accuracy: classification accuracy
            - precision: precision for positive class
            - recall: recall for positive class
            - f1: F1 score
            - auc: ROC AUC score
            - n_train: number of training samples
            - n_test: number of test samples

    Example:
        >>> from decision_making.models import RandomForestReturnModel
        >>> model = RandomForestReturnModel()
        >>> metrics = evaluate_fold(model, X_train, y_train, X_test, y_test)
        >>> print(f"Accuracy: {metrics['accuracy']:.4f}")
    """
    # Fit model
    model.fit(X_train, y_train)

    # Get predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]  # Probability of positive class

    # Compute metrics
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "auc": roc_auc_score(y_test, y_prob),
        "n_train": len(y_train),
        "n_test": len(y_test),
    }

    return metrics


def run_walk_forward_validation(
    df: pd.DataFrame,
    feature_cols: list[str],
    model_class: type,
    model_params: dict,
    **split_params,
) -> pd.DataFrame:
    """Execute complete walk-forward validation.

    This is the main function that orchestrates the entire validation process:
    1. Generate walk-forward splits
    2. For each fold, train and evaluate the model
    3. Collect metrics across all folds
    4. Return results as DataFrame

    Args:
        df: Complete feature DataFrame with date, ticker, features, target
        feature_cols: List of feature column names to use for training
        model_class: Model class (e.g., RandomForestReturnModel)
        model_params: Dict of model initialization parameters
        **split_params: Parameters for walk_forward_split()

    Returns:
        DataFrame with one row per fold containing:
            - fold_num, train_start, train_end, test_start, test_end
            - All metrics from evaluate_fold()
            - trained_model: fitted model object (for feature importance analysis)

    Example:
        >>> from decision_making.models import RandomForestReturnModel
        >>> results = run_walk_forward_validation(
        ...     df=features_df,
        ...     feature_cols=["return_lag_1", "rsi_14", ...],
        ...     model_class=RandomForestReturnModel,
        ...     model_params={"n_estimators": 100, "max_depth": 10},
        ...     initial_train_years=3,
        ...     step_months=6,
        ...     test_months=3,
        ... )
        >>> print(results[["fold_num", "accuracy", "auc"]].head())
    """
    print("Running walk-forward validation...")

    # Generate folds
    print("  Generating folds...")
    splits = walk_forward_split(df, **split_params)
    print(f"    - Generated {len(splits)} folds")

    if len(splits) == 0:
        raise ValueError("No valid folds generated. Check date range and split parameters.")

    # Iterate through folds
    results = []

    for split in splits:
        fold_num = split["fold_num"]
        print(
            f"  Fold {fold_num + 1}/{len(splits)}: "
            f"Train [{split['train_start'].date()}, {split['train_end'].date()}], "
            f"Test [{split['test_start'].date()}, {split['test_end'].date()}]"
        )

        # Extract train/test data
        train_data = df[split["train_idx"]]
        test_data = df[split["test_idx"]]

        X_train = train_data[feature_cols].values
        y_train = train_data["target"].values
        X_test = test_data[feature_cols].values
        y_test = test_data["target"].values

        # Initialize fresh model
        model = model_class(**model_params)

        # Evaluate fold
        fold_metrics = evaluate_fold(model, X_train, y_train, X_test, y_test)

        # Store results
        fold_result = {
            "fold_num": fold_num,
            "train_start": split["train_start"],
            "train_end": split["train_end"],
            "test_start": split["test_start"],
            "test_end": split["test_end"],
            **fold_metrics,
            "trained_model": model,  # Store for feature importance later
        }
        results.append(fold_result)

        # Print fold metrics
        print(
            f"    Accuracy: {fold_metrics['accuracy']:.4f}, "
            f"Precision: {fold_metrics['precision']:.4f}, "
            f"Recall: {fold_metrics['recall']:.4f}, "
            f"F1: {fold_metrics['f1']:.4f}, "
            f"AUC: {fold_metrics['auc']:.4f}"
        )

    # Convert to DataFrame
    results_df = pd.DataFrame(results)

    # Print summary
    print("\n" + "=" * 80)
    print("WALK-FORWARD VALIDATION SUMMARY")
    print("=" * 80)
    print(f"Number of folds: {len(results_df)}")
    print("\nMetrics (mean ± std):")
    for metric in ["accuracy", "precision", "recall", "f1", "auc"]:
        mean_val = results_df[metric].mean()
        std_val = results_df[metric].std()
        print(f"  {metric.upper():10s}: {mean_val:.4f} ± {std_val:.4f}")
    print("=" * 80)

    return results_df
