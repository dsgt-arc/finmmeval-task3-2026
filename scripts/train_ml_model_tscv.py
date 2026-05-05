from datetime import datetime
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt

from decision_making.ml_model.config import COMPETITION_TRAIN_START
from decision_making.ml_model.config import TRAIN_CONFIG as CONFIG
from decision_making.ml_model.feature_engineering import build_feature_matrix
from decision_making.ml_model.ml_model_manager import MODEL_FILENAME, build_reference_data
from decision_making.ml_model.model_persistence import save_model
from decision_making.ml_model.models import RandomForestReturnModel
from decision_making.ml_model.validation import run_walk_forward_validation

"""Train and evaluate Random Forest model for S&P 500 return prediction.

This script orchestrates the complete pipeline:
1. Builds feature matrix from SP500 data
2. Runs walk-forward validation
3. Generates evaluation reports
4. Saves feature importance analysis

Usage:
    python scripts/train_return_model.py
"""

# Use non-interactive backend for plotting
matplotlib.use("Agg")


def main():
    """Main function to run the complete training and evaluation pipeline."""
    print("=" * 80)
    print("S&P 500 Return Direction Prediction - Random Forest Model")

    # Step 1: Build feature matrix
    print("\n[1/4] Building feature matrix...")
    df = build_feature_matrix(min_obs=CONFIG["min_obs"], lags=CONFIG["lags"])

    # Filter to training data only (before competition period)
    df = df[df["date"] < COMPETITION_TRAIN_START]
    print(f"  - Filtered to data before {COMPETITION_TRAIN_START}: {len(df):,} rows remaining")

    # Step 2: Define features
    print("\n[2/4] Defining features...")
    # Exclude identifiers and target from features
    exclude_cols = ["date", "ticker", "target"]
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    print(f"  - Number of features: {len(feature_cols)}")
    print(f"  - Features: {', '.join(feature_cols[:10])}{'...' if len(feature_cols) > 10 else ''}")

    # Step 3: Walk-forward validation
    print("\n[3/4] Running walk-forward validation...")
    print(f"  - Using {CONFIG['embargo_days']}-day embargo to prevent feature leakage")
    results = run_walk_forward_validation(
        df=df,
        feature_cols=feature_cols,
        model_class=RandomForestReturnModel,
        model_params=CONFIG["model_params"],
        initial_train_years=CONFIG["initial_train_years"],
        step_months=CONFIG["step_months"],
        test_months=CONFIG["test_months"],
        embargo_days=CONFIG["embargo_days"],
    )

    # Step 4: Generate evaluation report
    print("\n[4/4] Generating evaluation report...")

    # Create output directory
    output_dir = Path(__file__).parent.parent / "output" / "rf_return_model"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save validation results
    results_to_save = results.drop(columns=["trained_model"])
    results_to_save.to_csv(output_dir / "validation_results.csv", index=False)
    print(f"  - Saved validation results to: {output_dir / 'validation_results.csv'}")

    # Feature importance (from last fold)
    print("\n" + "=" * 80)
    print("FEATURE IMPORTANCE (from final fold)")
    print("=" * 80)
    last_model = results.iloc[-1]["trained_model"]
    importance_df = last_model.get_feature_importance(feature_cols)
    print(importance_df.head(20))

    # Save feature importance
    importance_df.to_csv(output_dir / "feature_importance.csv", index=False)
    print(f"  - Saved feature importance to: {output_dir / 'feature_importance.csv'}")

    # Save production model for inference
    print("\nSaving production model...")
    final_model = results.iloc[-1]["trained_model"]
    reference_data = build_reference_data(df, CONFIG)
    model_path = save_model(
        model=final_model,
        metadata={
            "training_date": datetime.now().isoformat(),
            "feature_names": feature_cols,
            "model_params": CONFIG["model_params"],
            "validation_metrics": {
                "accuracy": float(results["accuracy"].mean()),
                "precision": float(results["precision"].mean()),
                "recall": float(results["recall"].mean()),
                "f1": float(results["f1"].mean()),
                "auc": float(results["auc"].mean()),
            },
            "n_samples": len(df),
            **reference_data,
        },
        path=output_dir / MODEL_FILENAME,
    )
    print(f"  - Saved model to:    {model_path}")
    print(f"  - Saved metadata to: {model_path.with_suffix('.json')}")
    print(
        f"  - Reference data includes {len(reference_data['sectors'])} sectors and {len(reference_data['feature_distributions'])} feature distributions"
    )

    # Plot metrics over time
    print("\nGenerating plots...")
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Walk-Forward Validation Results", fontsize=16, fontweight="bold")

    # Plot metrics
    metrics_to_plot = ["accuracy", "precision", "recall", "f1", "auc"]
    for idx, metric in enumerate(metrics_to_plot):
        row = idx // 3
        col = idx % 3
        ax = axes[row, col]

        ax.plot(results["fold_num"], results[metric], marker="o", linewidth=2)
        ax.axhline(
            results[metric].mean(),
            color="r",
            linestyle="--",
            label=f"Mean: {results[metric].mean():.4f}",
        )
        ax.set_xlabel("Fold Number", fontsize=10)
        ax.set_ylabel(metric.upper(), fontsize=10)
        ax.set_title(f"{metric.upper()} Over Time", fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Plot sample sizes
    ax = axes[1, 2]
    ax.plot(results["fold_num"], results["n_train"], marker="o", label="Train", linewidth=2)
    ax.plot(results["fold_num"], results["n_test"], marker="s", label="Test", linewidth=2)
    ax.set_xlabel("Fold Number", fontsize=10)
    ax.set_ylabel("Sample Size", fontsize=10)
    ax.set_title("Train/Test Sample Sizes", fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "metrics_over_time.png", dpi=150, bbox_inches="tight")
    print(f"  - Saved metrics plot to: {output_dir / 'metrics_over_time.png'}")

    # Plot feature importance
    print("\nGenerating feature importance plot...")
    fig, ax = plt.subplots(figsize=(10, 8))
    top_n = min(20, len(importance_df))
    top_features = importance_df.head(top_n)

    ax.barh(range(top_n), top_features["importance"])
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(top_features["feature"])
    ax.set_xlabel("Importance", fontsize=12)
    ax.set_title(f"Top {top_n} Feature Importances", fontsize=14, fontweight="bold")
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, axis="x")

    plt.tight_layout()
    plt.savefig(output_dir / "feature_importance.png", dpi=150, bbox_inches="tight")
    print(f"  - Saved feature importance plot to: {output_dir / 'feature_importance.png'}")

    # Final summary
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    print(f"All outputs saved to: {output_dir}")
    print("\nFiles generated:")
    print("  1. validation_results.csv - Per-fold metrics")
    print("  2. feature_importance.csv - Feature importance rankings")
    print("  3. metrics_over_time.png - Performance over validation folds")
    print("  4. feature_importance.png - Top features visualization")
    print("  5. production_model.pkl - Trained model for inference")
    print("  6. production_model_metadata.json - Model metadata")
    print("  7. reference_data.json - SP500 reference distributions for cross-sectional features")
    print("=" * 80)


if __name__ == "__main__":
    main()
