import polars as pl
from scipy.sparse import csr_matrix, hstack

import decision_making.data as d
from decision_making.logger import set_up_log
from decision_making.models import (
    LogisticRegressionModel,
    MultinomialNBModel,
    StatsmodelsLogitModel,
    train_test_split_temporal,
)
from decision_making.raw_strings import text_processing_pipeline
from decision_making.technical import get_technical_analyses, standardize_technical_signals

set_up_log()

# Model selection flags
USE_STATSMODELS = True
COMPARE_MODELS = False

# Load data
tsla = d.load_data(symbol="TSLA", download_if_missing=False)

# Label
tsla = tsla.sort(d.date_col).with_columns(d.price_col.pct_change().alias(d.return_col.meta.output_name()))
# Multi-class classification: 1 for positive return, -1 for negative return, 0 for no change
tsla = tsla.with_columns(
    pl.when(d.return_col > 0)
    .then(1)
    .when(d.return_col < 0)
    .then(-1)
    .otherwise(0)
    .shift(-1)
    .alias(d.target_multi_col.meta.output_name())
)
# Binary classification: 1 for positive return, 0 for negative or no change
tsla = tsla.with_columns(pl.when(d.return_col > 0).then(1).otherwise(0).shift(-1).alias(d.target_bin_col.meta.output_name()))
# Drop NanNs after shift
tsla = tsla.drop_nulls(subset=[d.target_multi_col.meta.output_name(), d.target_bin_col.meta.output_name()])

# numerical features
num_feature_raw = tsla.select(d.date_col, d.price_col).to_pandas()

technical_analyses = get_technical_analyses(num_feature_raw)
technical_features = standardize_technical_signals(
    technical_analyses, feature_cols=["trend", "mean_reversion", "rsi", "volatility"]
)
technical_features_sparse = csr_matrix(technical_features[["trend", "mean_reversion", "rsi", "volatility"]])


# text features
tsla = tsla.with_columns(d.text_col.list.len().alias(d.text_len_col.meta.output_name()))

tsla = tsla.with_columns(d.text_col.list.join(". ").alias(d.text_str_col.meta.output_name()))
texts = tsla.get_column(d.text_str_col.meta.output_name()).to_list()
dictionary, bow_corpus, matrix = text_processing_pipeline(texts, clip=True, prune_dict=True)

# combine numerical and technical featrues
X_final = hstack([matrix, technical_features_sparse]).toarray()


# Classification
y = tsla.get_column(d.target_bin_col.meta.output_name()).to_numpy()

# Simple train/test split (temporal)
X_train, X_test, y_train, y_test = train_test_split_temporal(X_final, y, test_size=0.2)

print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
print("-" * 60)

if USE_STATSMODELS:
    # Reduce BoW to 20 latent LSA components so MLE is well-determined and
    # the summary produces meaningful p-values / confidence intervals.

    print("\nStatsmodels Logit Regression (LSA-reduced BoW + technical, MLE):")
    print(f"Features: {X_train.shape[1]} (20 LSA components + 4 technical)")
    statsmodels_model = StatsmodelsLogitModel(use_regularization=False, maxiter=500, method="bfgs")
    statsmodels_model.fit(X_train, y_train)
    print(statsmodels_model.summary())
    print(f"Test accuracy: {statsmodels_model.score(X_test, y_test):.3f}")

elif COMPARE_MODELS:
    # Compare multiple models
    print("\nModel Comparison:")
    print("-" * 60)

    # Logistic Regression with elastic net
    print("\n1. Logistic Regression (Elastic Net):")
    logistic = LogisticRegressionModel(penalty="elasticnet", l1_ratio=0.5, max_iter=2000)
    logistic.fit(X_train, y_train)
    logistic_acc = logistic.score(X_test, y_test)
    print(f"   Test accuracy: {logistic_acc:.3f}")

    # Multinomial Naive Bayes
    print("\n2. Multinomial Naive Bayes:")
    nb = MultinomialNBModel(alpha=1.0)
    nb.fit(X_train, y_train)
    nb_acc = nb.score(X_test, y_test)
    print(f"   Test accuracy: {nb_acc:.3f}")

    # Summary
    print("\n" + "-" * 60)
    print("Summary:")
    print(f"  Best model: {'Logistic Regression' if logistic_acc > nb_acc else 'Naive Bayes'}")
    print(f"  Best accuracy: {max(logistic_acc, nb_acc):.3f}")

else:
    # Default: Just train Logistic Regression
    print("\nLogistic Regression (Elastic Net):")
    model = LogisticRegressionModel(penalty="elasticnet", l1_ratio=0.5, max_iter=2000)
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)
    print(f"Test accuracy: {accuracy:.3f}")
