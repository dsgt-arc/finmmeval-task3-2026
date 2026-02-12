"""Machine learning models for financial prediction tasks.

This module provides a unified class-based interface for various classification models
used in financial decision-making. All models follow a consistent API with fit(),
predict(), predict_proba(), and score() methods.
"""

from abc import ABC, abstractmethod

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
import statsmodels.api as sm


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


class LogisticRegressionModel(SklearnModel):
    """Logistic Regression classifier with elastic net regularization.

    Wraps sklearn's LogisticRegression with support for L1, L2, and elastic net
    penalties. Uses the 'saga' solver by default which supports elastic net.

    Args:
        penalty: Regularization penalty ('l1', 'l2', 'elasticnet', or 'none')
        l1_ratio: Elastic net mixing parameter (0 = L2, 1 = L1). Only used with elasticnet
        C: Inverse regularization strength (smaller = stronger regularization)
        solver: Optimization algorithm ('saga' recommended for elasticnet)
        max_iter: Maximum iterations for solver convergence
        random_state: Random seed for reproducibility

    Example:
        >>> model = LogisticRegressionModel(penalty="elasticnet", l1_ratio=0.5)
        >>> model.fit(X_train, y_train)
        >>> accuracy = model.score(X_test, y_test)
    """

    def __init__(
        self,
        penalty="elasticnet",
        l1_ratio=0.5,
        C=1.0,
        solver="saga",
        max_iter=2000,
        random_state=None,
    ):
        """Initialize Logistic Regression model."""
        super().__init__()
        self._model = LogisticRegression(
            penalty=penalty,
            l1_ratio=l1_ratio if penalty == "elasticnet" else None,
            C=C,
            solver=solver,
            max_iter=max_iter,
            random_state=random_state,
        )


class MultinomialNBModel(SklearnModel):
    """Multinomial Naive Bayes classifier.

    Suitable for classification with discrete features (e.g., word counts from
    text data). Handles sparse matrices efficiently, making it ideal for
    bag-of-words representations.

    Note: Requires non-negative features. For features with negative values,
    consider using ComplementNB or transforming the data.

    Args:
        alpha: Additive (Laplace/Lidstone) smoothing parameter (0 = no smoothing)
        fit_prior: Whether to learn class prior probabilities

    Example:
        >>> model = MultinomialNBModel(alpha=1.0)
        >>> model.fit(X_train, y_train)
        >>> predictions = model.predict(X_test)
    """

    def __init__(self, alpha=1.0, fit_prior=True):
        """Initialize Multinomial Naive Bayes model."""
        super().__init__()
        self._model = MultinomialNB(alpha=alpha, fit_prior=fit_prior)


class StatsmodelsLogitModel(BaseModel):
    """Statsmodels Logit regression with L1/L2 regularization.

    Wraps statsmodels' Logit class, providing statistical summaries not available
    in sklearn (e.g., p-values, confidence intervals). Automatically handles
    adding/removing the constant term for intercept.

    Args:
        method: Regularization method ('l1', 'l1_cvxopt_cp', or None for MLE)
        alpha: Regularization strength (only used with regularized methods)
        maxiter: Maximum iterations for optimization
        use_regularization: Whether to use regularized fit or standard MLE

    Example:
        >>> model = StatsmodelsLogitModel(method="l1", alpha=0.1)
        >>> model.fit(X_train, y_train)
        >>> print(model.summary())  # Statistical summary
        >>> accuracy = model.score(X_test, y_test)
    """

    def __init__(self, method="l1", alpha=0.1, maxiter=500, use_regularization=True):
        """Initialize Statsmodels Logit model."""
        self.method = method
        self.alpha = alpha
        self.maxiter = maxiter
        self.use_regularization = use_regularization
        self._model = None
        self._result = None

    def fit(self, X, y):
        """Fit the statsmodels Logit model.

        Automatically adds constant term for intercept estimation.

        Args:
            X: Feature matrix (numpy array)
            y: Target labels (numpy array)

        Returns:
            self: Fitted model instance
        """
        # Add constant for intercept
        X_with_const = sm.add_constant(X)

        # Create logit model
        self._model = sm.Logit(y, X_with_const)

        # Fit with or without regularization
        if self.use_regularization:
            self._result = self._model.fit_regularized(method=self.method, alpha=self.alpha, maxiter=self.maxiter, disp=True)
        else:
            self._result = self._model.fit(maxiter=self.maxiter, disp=True, method=self.method)

        return self

    def predict(self, X):
        """Predict class labels.

        Args:
            X: Feature matrix (numpy array)

        Returns:
            numpy array: Predicted class labels (0 or 1)
        """
        probas = self.predict_proba(X)
        return (probas[:, 1] > 0.5).astype(int)

    def predict_proba(self, X):
        """Predict class probabilities.

        Args:
            X: Feature matrix (numpy array)

        Returns:
            numpy array: Predicted class probabilities (shape: [n_samples, 2])
        """
        # Add constant to match training
        X_with_const = sm.add_constant(X)

        # Get probabilities for positive class
        proba_positive = self._result.predict(X_with_const)

        # Return probabilities for both classes
        proba_negative = 1 - proba_positive
        return np.column_stack([proba_negative, proba_positive])

    def summary(self):
        """Get statistical summary of the fitted model.

        Returns:
            statsmodels.iolib.summary.Summary: Model summary with coefficients,
                p-values, confidence intervals, etc.
        """
        if self._result is None:
            raise ValueError("Model must be fitted before calling summary()")
        return self._result.summary()

    @property
    def model(self):
        """Return the underlying statsmodels Logit instance.

        Returns:
            statsmodels.discrete.discrete_model.Logit: Underlying statsmodels model
        """
        return self._model


def train_test_split_temporal(X, y, test_size=0.2):
    """Split data into train/test sets preserving temporal ordering.

    Unlike sklearn's train_test_split, this function respects the time-series
    nature of financial data by using the last portion of the data as the test set.
    This prevents look-ahead bias in backtesting.

    Args:
        X: Feature matrix (numpy array or scipy sparse matrix)
        y: Target labels (numpy array)
        test_size: Fraction of data to use for testing (default: 0.2)

    Returns:
        tuple: (X_train, X_test, y_train, y_test)

    Example:
        >>> X_train, X_test, y_train, y_test = train_test_split_temporal(X, y, test_size=0.2)
        >>> # Last 20% of data is used for testing
    """
    # Use shape[0] instead of len() for compatibility with sparse matrices
    n_samples = X.shape[0]
    split_idx = int(n_samples * (1 - test_size))

    X_train = X[:split_idx]
    X_test = X[split_idx:]
    y_train = y[:split_idx]
    y_test = y[split_idx:]

    return X_train, X_test, y_train, y_test
