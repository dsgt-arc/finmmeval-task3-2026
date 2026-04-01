"""Online learning manager for Random Forest with warm-start retraining."""

import json
from pathlib import Path

import numpy as np

from .model_persistence import load_model, save_model


class OnlineModelManager:
    """Manages model and incremental learning with in-memory buffer."""

    def __init__(self, model_path: Path, reference_data_path: Path):
        """Initialize manager and load model.

        Args:
            model_path: Path to model .pkl file
            reference_data_path: Path to reference_data.json
        """
        self.model, self.metadata = load_model(model_path)
        with Path.open(reference_data_path) as f:
            self.reference_data = json.load(f)
        self.model_path = model_path
        self.observations = []  # In-memory buffer: [{features, target}, ...]

    def predict(self, X):
        """Make prediction with current model."""
        return self.model.predict_proba(X)

    def add_observation(self, features, target=None):
        """Add observation to buffer.

        Args:
            features: Feature array for this observation
            target: Target value (None if not yet known)
        """
        self.observations.append({"features": features, "target": target})

    def should_retrain(self):
        """Check if enough observations for retraining.

        Returns:
            Tuple of (should_retrain: bool, retrain_type: str)
        """
        complete = [o for o in self.observations if o["target"] is not None]
        if len(complete) >= 30:
            return True, "monthly"
        elif len(complete) >= 7:
            return True, "weekly"
        return False, None

    def warm_start_retrain(self, n_new_trees=20):
        """Add new trees to existing model.

        Args:
            n_new_trees: Number of trees to add (default: 20)
        """
        complete = [o for o in self.observations if o["target"] is not None]
        if not complete:
            return

        # Prepare data
        X = np.vstack([o["features"] for o in complete])
        y = np.array([o["target"] for o in complete])

        # Warm start: add trees and retrain
        self.model.model.n_estimators += n_new_trees
        self.model.model.warm_start = True
        self.model.fit(X, y)
        self.model.model.warm_start = False

        # Clear buffer
        self.observations = []

    def save(self):
        """Save updated model and metadata."""
        save_model(self.model, self.metadata, self.model_path)
