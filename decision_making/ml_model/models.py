"""Machine learning models for financial prediction tasks.

This module provides a unified class-based interface for various classification models
used in financial decision-making. All models follow a consistent API with fit(),
predict(), predict_proba(), and score() methods.
"""

from abc import ABC, abstractmethod

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


class BaseModel(ABC):
    """Abstract base class for all models.

    Defines the common interface that all model implementations must follow.
    """

    @abstractmethod
    def fit(self, X, y):
        """Fit the model to training data.

        Args:
            X: Feature matrix (numpy array or scipy sparse matrix)
            y: Target labels (numpy array)

        Returns:
            self: Fitted model instance
        """
        pass

    @abstractmethod
    def predict(self, X):
        """Predict class labels for samples in X.

        Args:
            X: Feature matrix (numpy array or scipy sparse matrix)

        Returns:
            numpy array: Predicted class labels
        """
        pass

    @abstractmethod
    def predict_proba(self, X):
        """Predict class probabilities for samples in X.

        Args:
            X: Feature matrix (numpy array or scipy sparse matrix)

        Returns:
            numpy array: Predicted class probabilities
        """
        pass

    def score(self, X, y):
        """Calculate accuracy score on test data.

        Args:
            X: Feature matrix (numpy array or scipy sparse matrix)
            y: True labels (numpy array)

        Returns:
            float: Accuracy score
        """
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)

    @property
    @abstractmethod
    def model(self):
        """Return the underlying model object.

        Returns:
            Underlying sklearn or statsmodels model instance
        """
        pass


class SklearnModel(BaseModel):
    """Base class for sklearn-based models.

    Provides common implementation for sklearn models, delegating to the
    underlying sklearn estimator.
    """

    def __init__(self):
        """Initialize sklearn model wrapper."""
        self._model = None

    def fit(self, X, y):
        """Fit the sklearn model to training data.

        Args:
            X: Feature matrix (numpy array or scipy sparse matrix)
            y: Target labels (numpy array)

        Returns:
            self: Fitted model instance
        """
        self._model.fit(X, y)
        return self

    def predict(self, X):
        """Predict class labels using sklearn model.

        Args:
            X: Feature matrix (numpy array or scipy sparse matrix)

        Returns:
            numpy array: Predicted class labels
        """
        return self._model.predict(X)

    def predict_proba(self, X):
        """Predict class probabilities using sklearn model.

        Args:
            X: Feature matrix (numpy array or scipy sparse matrix)

        Returns:
            numpy array: Predicted class probabilities
        """
        return self._model.predict_proba(X)

    @property
    def model(self):
        """Return the underlying sklearn model.

        Returns:
            sklearn estimator: Underlying sklearn model instance
        """
        return self._model


class RandomForestReturnModel(SklearnModel):
    """Random Forest classifier for return direction prediction.

    Wraps sklearn's RandomForestClassifier with financial market-specific defaults.
    Designed for cross-sectional prediction of stock return direction (up/down).

    The default hyperparameters are conservative to prevent overfitting in financial
    time series data:
    - Moderate tree depth to avoid fitting noise
    - High minimum samples per split/leaf to ensure statistical significance
    - Balanced class weights to handle return distribution imbalances

    Args:
        n_estimators: Number of trees in forest (default: 100)
        max_depth: Maximum depth of trees (default: 10 to prevent overfitting)
        min_samples_split: Minimum samples to split node (default: 100 for large datasets)
        min_samples_leaf: Minimum samples in leaf (default: 50)
        max_features: Number of features per split (default: 'sqrt')
        class_weight: Handle class imbalance (default: 'balanced')
        random_state: Random seed for reproducibility (default: 42)
        n_jobs: Parallel jobs (default: -1 for all cores)

    Example:
        >>> model = RandomForestReturnModel(n_estimators=200, max_depth=15)
        >>> model.fit(X_train, y_train)
        >>> accuracy = model.score(X_test, y_test)
        >>> feature_importance = model.get_feature_importance(feature_names)
    """

    def __init__(
        self,
        n_estimators=100,
        max_depth=10,
        min_samples_split=100,
        min_samples_leaf=50,
        max_features="sqrt",
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    ):
        """Initialize Random Forest model."""
        super().__init__()
        self._model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            class_weight=class_weight,
            random_state=random_state,
            n_jobs=n_jobs,
        )

    def get_feature_importance(self, feature_names: list[str]) -> pd.DataFrame:
        """Return feature importances as sorted DataFrame.

        Args:
            feature_names: List of feature names corresponding to training columns

        Returns:
            DataFrame with columns:
                - feature: feature name
                - importance: importance value (sum to 1.0)
            Sorted by importance in descending order.

        Raises:
            ValueError: If model hasn't been fitted yet

        Example:
            >>> model.fit(X_train, y_train)
            >>> importance_df = model.get_feature_importance(feature_cols)
            >>> print(importance_df.head(10))  # Top 10 features
        """
        if self._model is None or not hasattr(self._model, "feature_importances_"):
            raise ValueError("Model must be fitted before accessing feature importance")

        importance_df = pd.DataFrame({"feature": feature_names, "importance": self._model.feature_importances_})
        return importance_df.sort_values("importance", ascending=False).reset_index(drop=True)
