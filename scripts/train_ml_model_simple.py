"""Train Random Forest model with a simple chronological train/val split.

Simpler and faster alternative to train_return_model.py (walk-forward validation).
The final production model is trained on ALL data before COMPETITION_TRAIN_START.

Output (output/rf_return_model/):
  - rf_return_model.pkl   : trained model
  - rf_return_model.json  : combined metadata + reference data (for inference)
  - feature_importance.csv/png
  - val_metrics.png

Usage:
    uv run python scripts/train_return_model_simple.py
"""

from datetime import datetime
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

from decision_making.ml_model.config import COMPETITION_TRAIN_START
from decision_making.ml_model.config import TRAIN_CONFIG as CONFIG
from decision_making.ml_model.feature_engineering import build_feature_matrix
from decision_making.ml_model.ml_model_manager import MODEL_FILENAME, REFERENCE_FILENAME, build_reference_data
from decision_making.ml_model.model_persistence import save_model
from decision_making.ml_model.models import RandomForestReturnModel

matplotlib.use("Agg")

# Fraction of (time-ordered) dates to hold out for validation
VAL_SPLIT = 0.2


def main():
    print("=" * 80)
    print("S&P 500 Return Direction Prediction - Simple Train/Val Split")
    print("=" * 80)

    # Step 1: Build feature matrix
    print("\n[1/4] Building feature matrix...")
    df = build_feature_matrix(min_obs=CONFIG["min_obs"], lags=CONFIG["lags"])
    df = df[df["date"] < COMPETITION_TRAIN_START].copy()
    print(f"  - Rows after filtering to before {COMPETITION_TRAIN_START}: {len(df):,}")

    exclude_cols = ["date", "ticker", "target"]
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    print(f"  - Features: {len(feature_cols)}")

    # Step 2: Chronological train/val split
    print(f"\n[2/4] Splitting data (val = last {VAL_SPLIT:.0%} of dates)...")
    all_dates = sorted(df["date"].unique())
    split_idx = int(len(all_dates) * (1 - VAL_SPLIT))
    split_date = all_dates[split_idx]
    print(f"  - Train: up to {split_date}")
    print(f"  - Val:   {split_date} → {all_dates[-1]}")

    train_df = df[df["date"] < split_date].dropna(subset=[*feature_cols, "target"])
    val_df = df[df["date"] >= split_date].dropna(subset=[*feature_cols, "target"])
    print(f"  - Train samples: {len(train_df):,} | Val samples: {len(val_df):,}")

    # Step 3: Train on train split, evaluate on val split
    print("\n[3/4] Training model and evaluating on val set...")
    eval_model = RandomForestReturnModel(**CONFIG["model_params"])
    eval_model.fit(train_df[feature_cols].values, train_df["target"].values)

    y_val = val_df["target"].values.astype(int)
    y_pred = eval_model.predict(val_df[feature_cols].values)
    y_proba = eval_model.predict_proba(val_df[feature_cols].values)[:, 1]

    metrics = {
        "accuracy": float(accuracy_score(y_val, y_pred)),
        "precision": float(precision_score(y_val, y_pred, zero_division=0)),
        "recall": float(recall_score(y_val, y_pred, zero_division=0)),
        "f1": float(f1_score(y_val, y_pred, zero_division=0)),
        "auc": float(roc_auc_score(y_val, y_proba)),
    }
    print("\n  Validation metrics:")
    for name, value in metrics.items():
        print(f"    {name:>10}: {value:.4f}")

    # Step 4: Train final model on ALL data, save unified artifacts
    print("\n[4/4] Training final model on all data and saving...")
    output_dir = Path(__file__).parent.parent / "output" / "rf_return_model"
    output_dir.mkdir(parents=True, exist_ok=True)

    all_df = df.dropna(subset=[*feature_cols, "target"])
    final_model = RandomForestReturnModel(**CONFIG["model_params"])
    final_model.fit(all_df[feature_cols].values, all_df["target"].values)
    print(f"  - Final model trained on {len(all_df):,} samples")

    # Build reference data and merge with metadata into a single JSON
    reference_data = build_reference_data(all_df, CONFIG)
    combined = {
        "training_date": datetime.now().isoformat(),
        "feature_names": feature_cols,
        "model_params": CONFIG["model_params"],
        "validation_metrics": metrics,
        "n_samples": len(all_df),
        "val_split": VAL_SPLIT,
        "val_split_date": str(split_date),
        **reference_data,
    }

    # Save model + combined JSON (MODEL_FILENAME → rf_return_model.pkl,
    # save_model writes the JSON to rf_return_model.json = REFERENCE_FILENAME)
    assert Path(MODEL_FILENAME).stem == Path(REFERENCE_FILENAME).stem, (
        "MODEL_FILENAME and REFERENCE_FILENAME must share the same stem so that "
        "save_model writes exactly one JSON file used by the inference pipeline."
    )
    model_path = save_model(model=final_model, metadata=combined, path=output_dir / MODEL_FILENAME)
    print(f"  - Saved model to:    {model_path}")
    print(f"  - Saved metadata to: {model_path.with_suffix('.json')}")

    # Feature importance
    importance_df = final_model.get_feature_importance(feature_cols)
    importance_df.to_csv(output_dir / "feature_importance.csv", index=False)

    # Plots
    _, ax = plt.subplots(figsize=(10, 8))
    top_n = min(20, len(importance_df))
    top_features = importance_df.head(top_n)
    ax.barh(range(top_n), top_features["importance"])
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(top_features["feature"])
    ax.set_xlabel("Importance")
    ax.set_title(f"Top {top_n} Feature Importances")
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, axis="x")
    plt.tight_layout()
    plt.savefig(output_dir / "feature_importance.png", dpi=150, bbox_inches="tight")
    plt.close()

    _, ax = plt.subplots(figsize=(8, 5))
    metric_series = pd.Series(metrics)
    ax.bar(metric_series.index, metric_series.values)
    ax.set_ylim(0, 1)
    ax.set_title("Validation Metrics")
    ax.set_ylabel("Score")
    ax.grid(True, alpha=0.3, axis="y")
    for i, v in enumerate(metric_series.values):
        ax.text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=10)
    plt.tight_layout()
    plt.savefig(output_dir / "val_metrics.png", dpi=150, bbox_inches="tight")
    plt.close()

    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print(f"All outputs saved to: {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
