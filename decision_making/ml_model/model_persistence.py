"""Simple model persistence using joblib and JSON."""

import json
from pathlib import Path

import joblib


def save_model(model, metadata: dict, path: Path) -> Path:
    """Save model with joblib and metadata as JSON.

    Args:
        model: Trained model instance
        metadata: Dict with training_date, feature_names, model_params, n_samples
        path: Path to save model (e.g., 'output/model.pkl')

    Returns:
        Path to saved model
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Save model
    joblib.dump(model, path)

    # Save metadata alongside model
    meta_path = path.with_suffix('.json')
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    return path


def load_model(path: Path) -> tuple:
    """Load model and metadata.

    Args:
        path: Path to model file

    Returns:
        Tuple of (model, metadata dict)
    """
    path = Path(path)

    model = joblib.load(path)

    meta_path = path.with_suffix('.json')
    with open(meta_path) as f:
        metadata = json.load(f)

    return model, metadata
