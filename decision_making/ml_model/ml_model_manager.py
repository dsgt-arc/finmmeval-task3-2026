"""Simple model loader with module-level caching."""

from pathlib import Path

from .online_learning import OnlineModelManager

_manager_cache = None


def get_model_manager(model_dir=None):
    """Load ML model manager once, cache in memory.

    Args:
        model_dir: Path to model directory (default: output/rf_return_model/)

    Returns:
        OnlineModelManager instance
    """
    global _manager_cache

    if _manager_cache is not None:
        return _manager_cache

    if model_dir is None:
        model_dir = Path(__file__).parent.parent / "output" / "rf_return_model"

    model_path = model_dir / "production_model.pkl"
    reference_path = model_dir / "reference_data.json"

    _manager_cache = OnlineModelManager(model_path, reference_path)
    return _manager_cache
