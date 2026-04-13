"""Online learning manager for Random Forest with warm-start retraining."""

import datetime
import json
from pathlib import Path

import numpy as np
import pandas as pd

from decision_making.sp500_data import append_adjclose_to_store, fetch_sp500_adjclose_since, load_stock_metric_long

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
        self.reference_data_path = reference_data_path
        self.observations = []  # In-memory buffer: [{features, target}, ...]

    def predict(self, X):
        """Make prediction with current model."""
        return self.model.predict_proba(X)

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

    def read_local_data(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Lazy-load and cache SP500 adj close and returns wide DataFrames."""

        price_data = load_stock_metric_long("Adj_Close").astype(float)
        return price_data

    def cross_sectional_retrain(self, X_batch: np.ndarray, y_batch: np.ndarray, n_new_trees: int = 100) -> None:
        """Online-learn from a cross-sectional batch using warm-start.

        Adds n_new_trees trained on X_batch/y_batch to the existing ensemble.
        Does not touch the single-stock observation buffer.

        Args:
            X_batch: Feature matrix (n_samples, n_features)
            y_batch: Binary targets (n_samples,)
            n_new_trees: Number of trees to add (default: 10)
        """
        if len(X_batch) == 0:
            return
        self.model.model.n_estimators += n_new_trees
        self.model.model.warm_start = True
        self.model.fit(X_batch, y_batch)
        self.model.model.warm_start = False

    def get_data(self, through_date: datetime.date) -> None:
        """Fetch SP500 adj close data from yfinance for any dates not yet in cache.

        Uses last_obs_date from reference_data to determine the fetch window.
        Appends new rows to _adj_close_wide and persists updated last_obs_date.

        Args:
            through_date: datetime.date — fetch data up to and including this date
        """

        price_data = self.read_local_data()

        # check if price_data is already up to date
        through_ts = pd.Timestamp(through_date)
        check_available = price_data.index[price_data.index == through_ts]
        if check_available.empty:
            # price_data needs to be updated, needs one more increment
            end = (through_ts + datetime.timedelta(days=1)).strftime("%Y-%m-%d")
            start = (price_data.index.max() + datetime.timedelta(days=1)).strftime("%Y-%m-%d")

            new_data = fetch_sp500_adjclose_since(
                tickers=list(price_data.columns),
                start_date=start,
                end_date=end,
            )

            if not new_data.empty:
                append_adjclose_to_store(new_data)
                price_data = pd.concat([price_data, new_data]).sort_index()
                price_data = price_data[~price_data.index.duplicated(keep="last")]

        return price_data

    def save(self):
        """Save updated model and metadata."""
        save_model(self.model, self.metadata, self.model_path)
