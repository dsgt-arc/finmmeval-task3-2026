"""Simple model loader with module-level caching."""

from datetime import datetime
import json
from pathlib import Path

from .config import (
    COMPETITION_TRAIN_START,
    FEATURE_DIST_COLS,
    FEATURE_DIST_PERCENTILES,
    TRAIN_CONFIG,
)
from .online_learning import OnlineModelManager

_manager_cache = None

MODEL_FILENAME = "rf_return_model.pkl"
REFERENCE_FILENAME = "rf_return_model.json"
MODEL_DIR = Path("output") / "rf_return_model"


def build_reference_data(df, config: dict) -> dict:
    """Build reference data dict from a feature matrix DataFrame.

    Loads sector mapping and computes feature distributions for percentile
    ranking at inference time. Used by both the training script and the
    auto-train path in get_model_manager().

    Args:
        df: Feature matrix DataFrame (output of build_feature_matrix)
        config: Training config dict (must contain 'min_obs')

    Returns:
        Dict with keys: sectors, sector_map, feature_distributions, last_obs_date
    """
    from decision_making.sp500_data import load_stock_sector

    try:
        sector_df = load_stock_sector(min_obs=config["min_obs"])
        sector_map = {}
        for ticker in sector_df.columns:
            vals = sector_df[ticker].dropna()
            if len(vals) > 0:
                sector_map[ticker] = vals.iloc[0]
        sectors = list(set(sector_map.values()))
    except Exception:
        sector_map, sectors = {}, []

    reference_data = {
        "sectors": sectors,
        "sector_map": sector_map,
        "feature_distributions": {},
        "last_obs_date": df["date"].max().strftime("%Y-%m-%d"),
    }
    for col in FEATURE_DIST_COLS:
        if col in df.columns:
            reference_data["feature_distributions"][col] = {
                "mean": float(df[col].mean()),
                "std": float(df[col].std()),
                "percentiles": {float(k): float(v) for k, v in df[col].quantile(FEATURE_DIST_PERCENTILES).to_dict().items()},
            }
    return reference_data


def _train_initial_model(model_dir: Path) -> None:
    """Train and save the initial model if it doesn't exist.

    Runs the same pipeline as scripts/train_return_model.py.
    This will take several minutes on first run.
    """
    from util.logger import logger

    logger.info("No trained model found. Running initial training...")
    logger.info(f"  Data cutoff: {COMPETITION_TRAIN_START}")

    from decision_making.ml_model.feature_engineering import build_feature_matrix
    from decision_making.ml_model.model_persistence import save_model
    from decision_making.models import RandomForestReturnModel
    from decision_making.validation import run_walk_forward_validation

    cfg = TRAIN_CONFIG

    df = build_feature_matrix(min_obs=cfg["min_obs"], lags=cfg["lags"])
    df = df[df["date"] < COMPETITION_TRAIN_START]

    exclude_cols = ["date", "ticker", "target"]
    feature_cols = [c for c in df.columns if c not in exclude_cols]

    results = run_walk_forward_validation(
        df=df,
        feature_cols=feature_cols,
        model_class=RandomForestReturnModel,
        model_params=cfg["model_params"],
        initial_train_years=cfg["initial_train_years"],
        step_months=cfg["step_months"],
        test_months=cfg["test_months"],
        embargo_days=cfg["embargo_days"],
    )

    model_dir.mkdir(parents=True, exist_ok=True)

    final_model = results.iloc[-1]["trained_model"]
    save_model(
        model=final_model,
        metadata={
            "training_date": datetime.now().isoformat(),
            "feature_names": feature_cols,
            "model_params": cfg["model_params"],
            "validation_metrics": {
                "accuracy": float(results["accuracy"].mean()),
                "precision": float(results["precision"].mean()),
                "recall": float(results["recall"].mean()),
                "f1": float(results["f1"].mean()),
                "auc": float(results["auc"].mean()),
            },
            "n_samples": len(df),
        },
        path=model_dir / MODEL_FILENAME,
    )

    reference_data = build_reference_data(df, cfg)
    with Path.open(model_dir / REFERENCE_FILENAME, "w") as f:
        json.dump(reference_data, f, indent=2)

    logger.info("Initial training complete. Model saved to: %s", model_dir)


def get_model_manager(model_dir=None):
    """Load ML model manager once, cache in memory.

    If the model does not exist it will be trained automatically on first call.

    Args:
        model_dir: Path to model directory (default: output/rf_return_model/)

    Returns:
        OnlineModelManager instance
    """
    global _manager_cache

    if _manager_cache is not None:
        return _manager_cache

    if model_dir is None:
        model_dir = Path(__file__).parent.parent.parent / MODEL_DIR

    model_path = model_dir / MODEL_FILENAME
    reference_path = model_dir / REFERENCE_FILENAME

    if not model_path.exists():
        _train_initial_model(model_dir)

    _manager_cache = OnlineModelManager(model_path, reference_path)
    return _manager_cache
