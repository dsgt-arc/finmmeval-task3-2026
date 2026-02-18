"""Tests for the decision_making.models module."""

import numpy as np
import pytest
from scipy.sparse import csr_matrix
from sklearn.datasets import make_classification

from decision_making.models import (
    LogisticRegressionModel,
    MultinomialNBModel,
    StatsmodelsLogitModel,
    train_test_split_temporal,
)


@pytest.fixture
def binary_classification_data():
    """Generate synthetic binary classification data."""
    X, y = make_classification(
        n_samples=200,
        n_features=10,
        n_informative=8,
        n_redundant=2,
        n_classes=2,
        random_state=42,
    )
    return X, y


@pytest.fixture
def sparse_positive_data():
    """Generate synthetic sparse data with positive values (for Multinomial NB)."""
    np.random.seed(42)
    # Create sparse matrix with positive integers (like word counts)
    X = np.random.randint(0, 10, size=(200, 10)).astype(float)
    y = np.random.randint(0, 2, size=200)
    X_sparse = csr_matrix(X)
    return X_sparse, y


class TestTrainTestSplitTemporal:
    """Tests for train_test_split_temporal function."""

    def test_split_ratio(self, binary_classification_data):
        """Test that split produces correct train/test ratio."""
        X, y = binary_classification_data
        X_train, X_test, y_train, y_test = train_test_split_temporal(X, y, test_size=0.2)

        assert len(X_train) == 160  # 80% of 200
        assert len(X_test) == 40  # 20% of 200
        assert len(y_train) == 160
        assert len(y_test) == 40

    def test_temporal_ordering(self, binary_classification_data):
        """Test that temporal ordering is preserved (test set is last portion)."""
        X, y = binary_classification_data
        X_train, X_test, y_train, y_test = train_test_split_temporal(X, y, test_size=0.2)

        # Last sample in train should be index 159
        # First sample in test should be index 160
        np.testing.assert_array_equal(X_train[-1], X[159])
        np.testing.assert_array_equal(X_test[0], X[160])

    def test_different_test_sizes(self, binary_classification_data):
        """Test split with different test sizes."""
        X, y = binary_classification_data

        for test_size in [0.1, 0.3, 0.5]:
            X_train, X_test, y_train, y_test = train_test_split_temporal(X, y, test_size=test_size)
            expected_test_size = int(len(X) * test_size)
            assert len(X_test) == expected_test_size
            assert len(X_train) == len(X) - expected_test_size


class TestLogisticRegressionModel:
    """Tests for LogisticRegressionModel."""

    def test_initialization(self):
        """Test model initialization with various parameters."""
        model = LogisticRegressionModel(penalty="elasticnet", l1_ratio=0.5, C=1.0, max_iter=1000)
        assert model.model is not None
        assert model.model.penalty == "elasticnet"
        assert model.model.l1_ratio == 0.5

    def test_fit_predict(self, binary_classification_data):
        """Test fitting and prediction."""
        X, y = binary_classification_data
        X_train, X_test, y_train, y_test = train_test_split_temporal(X, y, test_size=0.2)

        model = LogisticRegressionModel(penalty="elasticnet", l1_ratio=0.5, max_iter=2000)
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        assert predictions.shape == (len(X_test),)
        assert set(predictions).issubset({0, 1})

    def test_predict_proba(self, binary_classification_data):
        """Test probability predictions."""
        X, y = binary_classification_data
        X_train, X_test, y_train, y_test = train_test_split_temporal(X, y, test_size=0.2)

        model = LogisticRegressionModel(penalty="l2", max_iter=2000)
        model.fit(X_train, y_train)

        probas = model.predict_proba(X_test)
        assert probas.shape == (len(X_test), 2)
        # Probabilities should sum to 1
        np.testing.assert_allclose(probas.sum(axis=1), 1.0, rtol=1e-5)
        # Probabilities should be in [0, 1]
        assert np.all(probas >= 0) and np.all(probas <= 1)

    def test_score(self, binary_classification_data):
        """Test accuracy scoring."""
        X, y = binary_classification_data
        X_train, X_test, y_train, y_test = train_test_split_temporal(X, y, test_size=0.2)

        model = LogisticRegressionModel(penalty="l2", max_iter=2000)
        model.fit(X_train, y_train)

        accuracy = model.score(X_test, y_test)
        assert 0.0 <= accuracy <= 1.0

    def test_model_property(self, binary_classification_data):
        """Test that model property returns underlying sklearn object."""
        X, y = binary_classification_data
        X_train, X_test, y_train, y_test = train_test_split_temporal(X, y, test_size=0.2)

        model = LogisticRegressionModel()
        model.fit(X_train, y_train)

        sklearn_model = model.model
        assert hasattr(sklearn_model, "coef_")
        assert hasattr(sklearn_model, "intercept_")


class TestMultinomialNBModel:
    """Tests for MultinomialNBModel."""

    def test_initialization(self):
        """Test model initialization."""
        model = MultinomialNBModel(alpha=1.0, fit_prior=True)
        assert model.model is not None
        assert model.model.alpha == 1.0
        assert model.model.fit_prior is True

    def test_fit_predict_sparse(self, sparse_positive_data):
        """Test fitting and prediction with sparse data."""
        X, y = sparse_positive_data
        X_train, X_test, y_train, y_test = train_test_split_temporal(X, y, test_size=0.2)

        model = MultinomialNBModel(alpha=1.0)
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        assert predictions.shape == (X_test.shape[0],)
        assert set(predictions).issubset({0, 1})

    def test_predict_proba_sparse(self, sparse_positive_data):
        """Test probability predictions with sparse data."""
        X, y = sparse_positive_data
        X_train, X_test, y_train, y_test = train_test_split_temporal(X, y, test_size=0.2)

        model = MultinomialNBModel(alpha=1.0)
        model.fit(X_train, y_train)

        probas = model.predict_proba(X_test)
        assert probas.shape == (X_test.shape[0], 2)
        np.testing.assert_allclose(probas.sum(axis=1), 1.0, rtol=1e-5)

    def test_score(self, sparse_positive_data):
        """Test accuracy scoring."""
        X, y = sparse_positive_data
        X_train, X_test, y_train, y_test = train_test_split_temporal(X, y, test_size=0.2)

        model = MultinomialNBModel(alpha=1.0)
        model.fit(X_train, y_train)

        accuracy = model.score(X_test, y_test)
        assert 0.0 <= accuracy <= 1.0

    def test_different_alpha_values(self, sparse_positive_data):
        """Test model with different smoothing parameters."""
        X, y = sparse_positive_data
        X_train, X_test, y_train, y_test = train_test_split_temporal(X, y, test_size=0.2)

        for alpha in [0.1, 0.5, 1.0, 2.0]:
            model = MultinomialNBModel(alpha=alpha)
            model.fit(X_train, y_train)
            accuracy = model.score(X_test, y_test)
            assert 0.0 <= accuracy <= 1.0


class TestStatsmodelsLogitModel:
    """Tests for StatsmodelsLogitModel."""

    def test_initialization(self):
        """Test model initialization."""
        model = StatsmodelsLogitModel(method="l1", alpha=0.1, maxiter=500)
        assert model.method == "l1"
        assert model.alpha == 0.1
        assert model.maxiter == 500

    def test_fit_predict(self, binary_classification_data):
        """Test fitting and prediction."""
        X, y = binary_classification_data
        X_train, X_test, y_train, y_test = train_test_split_temporal(X, y, test_size=0.2)

        model = StatsmodelsLogitModel(method="l1", alpha=0.1, maxiter=500)
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        assert predictions.shape == (len(X_test),)
        assert set(predictions).issubset({0, 1})

    def test_predict_proba(self, binary_classification_data):
        """Test probability predictions."""
        X, y = binary_classification_data
        X_train, X_test, y_train, y_test = train_test_split_temporal(X, y, test_size=0.2)

        model = StatsmodelsLogitModel(method="l1", alpha=0.1, maxiter=500)
        model.fit(X_train, y_train)

        probas = model.predict_proba(X_test)
        assert probas.shape == (len(X_test), 2)
        np.testing.assert_allclose(probas.sum(axis=1), 1.0, rtol=1e-5)

    def test_score(self, binary_classification_data):
        """Test accuracy scoring."""
        X, y = binary_classification_data
        X_train, X_test, y_train, y_test = train_test_split_temporal(X, y, test_size=0.2)

        model = StatsmodelsLogitModel(method="l1", alpha=0.1, maxiter=500)
        model.fit(X_train, y_train)

        accuracy = model.score(X_test, y_test)
        assert 0.0 <= accuracy <= 1.0

    def test_summary(self, binary_classification_data):
        """Test that summary method works after fitting."""
        X, y = binary_classification_data
        X_train, X_test, y_train, y_test = train_test_split_temporal(X, y, test_size=0.2)

        model = StatsmodelsLogitModel(method="l1", alpha=0.1, maxiter=500)
        model.fit(X_train, y_train)

        summary = model.summary()
        assert summary is not None
        # Summary should have string representation
        assert len(str(summary)) > 0

    def test_summary_before_fit_raises_error(self):
        """Test that calling summary before fit raises an error."""
        model = StatsmodelsLogitModel()

        with pytest.raises(ValueError, match="Model must be fitted"):
            model.summary()

    def test_mle_fit(self, binary_classification_data):
        """Test MLE fitting without regularization."""
        X, y = binary_classification_data
        X_train, X_test, y_train, y_test = train_test_split_temporal(X, y, test_size=0.2)

        model = StatsmodelsLogitModel(use_regularization=False, maxiter=100)
        model.fit(X_train, y_train)

        accuracy = model.score(X_test, y_test)
        assert 0.0 <= accuracy <= 1.0


class TestModelInterchangeability:
    """Tests to ensure all models follow the same interface."""

    @pytest.mark.parametrize(
        "model_class,model_kwargs,data_fixture",
        [
            (LogisticRegressionModel, {"max_iter": 1000}, "binary_classification_data"),
            (MultinomialNBModel, {"alpha": 1.0}, "sparse_positive_data"),
            (StatsmodelsLogitModel, {"maxiter": 500}, "binary_classification_data"),
        ],
    )
    def test_common_interface(self, model_class, model_kwargs, data_fixture, request):
        """Test that all models implement the same interface."""
        # Get the appropriate data fixture
        X, y = request.getfixturevalue(data_fixture)
        X_train, X_test, y_train, y_test = train_test_split_temporal(X, y, test_size=0.2)

        model = model_class(**model_kwargs)

        # All models should have these methods
        assert hasattr(model, "fit")
        assert hasattr(model, "predict")
        assert hasattr(model, "predict_proba")
        assert hasattr(model, "score")
        assert hasattr(model, "model")

        # Test that methods work
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        probas = model.predict_proba(X_test)
        accuracy = model.score(X_test, y_test)

        assert predictions.shape == (X_test.shape[0],)
        assert probas.shape == (X_test.shape[0], 2)
        assert 0.0 <= accuracy <= 1.0
