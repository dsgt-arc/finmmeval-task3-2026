import numpy as np
import pandas as pd
import polars as pl

from decision_making.data import load_data
from decision_making.logger import set_up_log
from decision_making.models import (
    StatsmodelsLogitModel,
)
from decision_making.raw_strings import text_processing_pipeline
from decision_making.technical import get_technical_analyses, transform_signal_to_df

set_up_log()

# Symbols for panel
SYMBOLS = ["TSLA", "BMRN", "MRNA", "MSFT"]

# Polars column expressions
price_col = pl.col("prices")
return_col = pl.col("returns")
target_multi_col = pl.col("target_mulit")
target_bin_col = pl.col("target_binary")
date_col = pl.col("date")
text_col = pl.col("news")
text_len_col = pl.col("news_length")
text_str_col = pl.col("news_str")

# ---------------------------------------------------------------------------
# Step 1: Per-stock label computation + news text collection
# ---------------------------------------------------------------------------
stock_dfs = {}
all_texts = []
text_offsets = {}  # symbol -> (start, end) slice into all_texts

for symbol in SYMBOLS:
    df = load_data(symbol=symbol, download_if_missing=False)

    # Returns and targets
    df = df.sort(date_col).with_columns(
        price_col.pct_change().alias(return_col.meta.output_name())
    )
    df = df.with_columns(
        pl.when(return_col > 0)
        .then(1)
        .when(return_col < 0)
        .then(-1)
        .otherwise(0)
        .shift(-1)
        .alias(target_multi_col.meta.output_name())
    )
    df = df.with_columns(
        pl.when(return_col > 0)
        .then(1)
        .otherwise(0)
        .shift(-1)
        .alias(target_bin_col.meta.output_name())
    )
    df = df.drop_nulls(subset=[target_multi_col.meta.output_name(), target_bin_col.meta.output_name()])

    # News string
    df = df.with_columns(text_col.list.len().alias(text_len_col.meta.output_name()))
    df = df.with_columns(text_col.list.join(". ").alias(text_str_col.meta.output_name()))

    stock_dfs[symbol] = df

    texts = df.get_column(text_str_col.meta.output_name()).to_list()
    start = len(all_texts)
    all_texts.extend(texts)
    text_offsets[symbol] = (start, start + len(texts))

# ---------------------------------------------------------------------------
# Step 2: Joint text pipeline — single shared LSA space for all stocks
# ---------------------------------------------------------------------------
dictionary, bow_corpus, matrix = text_processing_pipeline(all_texts, clip=True, prune_dict=True)
# matrix: dense ndarray [total_rows, 20]

# ---------------------------------------------------------------------------
# Step 3: Raw technical signals per stock → panel DataFrame
# ---------------------------------------------------------------------------
tech_rows = []
for symbol in SYMBOLS:
    df = stock_dfs[symbol]
    num_feature_raw = df.select(date_col, price_col).to_pandas()
    technical_analyses = get_technical_analyses(num_feature_raw)

    for date, analysis in technical_analyses.items():
        row = transform_signal_to_df(analysis)
        row["date"] = date
        row["symbol"] = symbol
        tech_rows.append(row)

panel_tech = pd.concat(tech_rows, ignore_index=True)

# Cross-sectional standardization: for each date, normalize across the 4 stocks.
# Dates where all stocks share the same signal (zero cross-sectional std) are
# left at zero after mean-centering.
feature_cols_tech = ["trend", "mean_reversion", "rsi", "volatility"]


def _cs_standardize(x: pd.Series) -> pd.Series:
    std = x.std(ddof=0)
    return (x - x.mean()) / std if std > 0 else x - x.mean()


panel_tech[feature_cols_tech] = panel_tech.groupby("date")[feature_cols_tech].transform(_cs_standardize)

# ---------------------------------------------------------------------------
# Step 4: Assemble per-stock feature arrays and stack into panel
# ---------------------------------------------------------------------------
X_parts, y_parts, date_parts, symbol_parts = [], [], [], []

for symbol in SYMBOLS:
    df = stock_dfs[symbol]
    start, end = text_offsets[symbol]
    text_feats = matrix[start:end]  # [n, 20]

    tech_feats = (
        panel_tech[panel_tech["symbol"] == symbol]
        .sort_values("date")
        .reset_index(drop=True)[feature_cols_tech]
        .to_numpy()
    )

    X_parts.append(np.hstack([text_feats, tech_feats]))  # [n, 24]
    y_parts.append(df.get_column(target_bin_col.meta.output_name()).to_numpy())
    date_parts.extend(df.get_column(date_col.meta.output_name()).to_list())
    symbol_parts.extend([symbol] * len(df))

X_all = np.vstack(X_parts)    # [total, 24]
y_all = np.concatenate(y_parts)

# ---------------------------------------------------------------------------
# Step 5: Stock fixed effects — dummy variables (TSLA = reference, dropped)
# ---------------------------------------------------------------------------
symbol_cat = pd.Categorical(symbol_parts, categories=SYMBOLS)
stock_dummies = pd.get_dummies(symbol_cat, drop_first=True).to_numpy(dtype=float)
# 3 columns: BMRN, MRNA, MSFT

X_panel = np.hstack([X_all, stock_dummies])  # [total, 27]

# ---------------------------------------------------------------------------
# Step 6: Temporal split by common date cutoff (80/20)
# ---------------------------------------------------------------------------
dates_array = np.array(date_parts)
all_unique_dates = sorted(set(date_parts))
cutoff = all_unique_dates[int(len(all_unique_dates) * 0.8)]

train_mask = dates_array <= cutoff
test_mask = dates_array > cutoff

X_train, X_test = X_panel[train_mask], X_panel[test_mask]
y_train, y_test = y_all[train_mask], y_all[test_mask]

print(f"Panel: {len(SYMBOLS)} stocks")
print(f"Training obs: {len(X_train)}, Test obs: {len(X_test)}")
print(f"Features: {X_train.shape[1]} (20 LSA + 4 technical + {len(SYMBOLS) - 1} stock FE)")
print("-" * 60)

# ---------------------------------------------------------------------------
# Step 7: Panel logit with entity fixed effects
# ---------------------------------------------------------------------------
print("\nPanel Logit (LSA + cross-sectional technical + stock FE, MLE):")
statsmodels_model = StatsmodelsLogitModel(use_regularization=False, maxiter=500, method="bfgs")
statsmodels_model.fit(X_train, y_train)
print(statsmodels_model.summary())
print(f"Test accuracy: {statsmodels_model.score(X_test, y_test):.3f}")
