# ML Model: Architecture and Learning Modes

The ML model is a **Random Forest binary classifier** that predicts whether a stock's next-day return will be positive (up) or negative (down). It integrates into the decision-making framework as an analyst agent that produces `BULLISH`, `BEARISH`, or `NEUTRAL` signals.

---

## Overview

| Aspect | Detail |
|---|---|
| Model type | `RandomForestClassifier` (sklearn) |
| Target | Binary: 1 if next-day return > 0, else 0 |
| Output | Probability of positive return → converted to analyst signal |
| Training data cutoff | `COMPETITION_TRAIN_START = "2024-08-01"` (config.py) |
| Ensemble size | 500 trees (initial), grows during online learning |

---

## Part A: Pre-Training vs. Automatic Training

### Option 1: Run the training script explicitly (recommended)

Run `scripts/train_return_model.py` before deploying the agent:

```bash
uv run python scripts/train_return_model.py
```

This script:
1. Builds a feature matrix from S&P 500 historical prices (data up to `COMPETITION_TRAIN_START`)
2. Runs **walk-forward validation** to evaluate model quality
3. Trains a final model on **all available pre-cutoff data**
4. Saves the model and supporting artifacts to `output/rf_return_model/`

**Walk-forward validation** uses an expanding training window to prevent look-ahead bias:

```
Fold 1: Train [2001 → 2008]  |  [21-day embargo]  |  Test [3 months]
Fold 2: Train [2001 → 2008-mid]  |  [embargo]  |  Test [3 months]
...
```

Key parameters (all in `config.py`):

| Parameter | Value | Meaning |
|---|---|---|
| `initial_train_years` | 7 | Length of first training window |
| `step_months` | 6 | How far each fold advances |
| `test_months` | 3 | Test window per fold |
| `embargo_days` | 21 | Gap between train and test end |

The 21-day embargo is critical: because features include 21-day lagged returns, without an embargo the test features would overlap with training observations, causing leakage.

After validation, the script trains on all data and saves:

```
output/rf_return_model/
├── rf_return_model.pkl         # Serialized model
├── rf_return_model.json        # Metadata + reference data for inference
├── validation_results.csv      # Per-fold metrics
├── feature_importance.csv      # Feature rankings
├── feature_importance.png
└── metrics_over_time.png
```

### Option 2: Lazy automatic training

If no saved model exists when the agent is first called, `get_model_manager()` (in `ml_model_manager.py`) automatically calls `_train_initial_model()`, which runs the same pipeline as the script. No manual action needed — the model will train itself on first use.

This is convenient during development but can cause a long delay on first inference. For production, prefer running the training script explicitly beforehand.

### Features used

**Lagged return features:**
- `return_lag_1` — 1-day prior return
- `return_lag_5` — 5-day cumulative return
- `return_lag_21` — 21-day cumulative return

**Technical indicators:**
- `rsi_14` — 14-period RSI
- `bb_position` — position within Bollinger Bands (0–1)
- `volatility_21d` — 21-day rolling annualised volatility
- `ema_8`, `ema_21`, `ema_55` — exponential moving averages
- `ema_cross_short` / `ema_cross_long` — binary EMA crossover flags

**Cross-sectional features (relative to S&P 500 on same date):**
- `return_rank_pct` — percentile rank of `return_lag_1` within the cross-section
- `volatility_rank_pct` — percentile rank of volatility within the cross-section
- `sector_*` — one-hot encoded sector dummies (11 GICS sectors)

---

## Part B: Online Learning

Once deployed, the model continues learning as new market data arrives. This happens inside `ml_model_agent_online()` (in `decision_making/agents/analysts/ml_model.py`) every time the agent is called for a new trading date.

### How it works: warm-start tree addition

Online learning does **not** retrain the entire forest from scratch. Instead it uses sklearn's `warm_start=True` to extend the ensemble:

```python
# From online_learning.py – cross_sectional_retrain()
self.model.model.n_estimators += n_new_trees   # e.g. 500 → 600
self.model.model.warm_start = True
self.model.fit(X_batch, y_batch)               # only new trees are trained
self.model.model.warm_start = False
```

This means:
- Existing 500 trees are **preserved** (no retraining)
- `N_NEW_TREES_CROSS_SECTIONAL = 100` new trees are grown on the latest data
- The ensemble grows over time, accumulating knowledge from live observations

### The two-step process per trading date

Every call to `ml_model_agent_online()` proceeds in two steps:

#### Step 1 — Cross-sectional online learning

Before predicting, the model learns from the most recent completed trading day:

```
Load S&P 500 price history through (trading_date - 1)
         ↓
Build features for all ~500 stocks on that date
         ↓
Compute targets: did each stock go up the next day?
         ↓
cross_sectional_retrain(X_batch, y_batch, n_new_trees=100)
         ↓
Save updated model to disk
```

Using all ~500 stocks in one batch (rather than a single stock) gives much better statistical power. The features are computed as live cross-sectional ranks — reflecting the actual market regime on that date — rather than static historical distributions.

#### Step 2 — Single-stock prediction

```
Load price history for the target ticker
         ↓
Build single-stock features
  (cross-sectional ranks use static reference_data percentiles)
         ↓
predict_proba() → P(next-day return > 0)
         ↓
P > 0.55  → BULLISH
P < 0.45  → BEARISH
Otherwise → NEUTRAL
```

The LLM then uses this probability alongside a brief justification to produce the final `AnalystSignal`.

### Reference data and feature consistency

A key requirement for online learning is that inference features match the format seen during training. This is handled through `reference_data` stored in `rf_return_model.json`:

- **`feature_distributions`** — percentile distributions of `return_lag_1`, `return_lag_5`, `return_lag_21`, `volatility_21d` (computed on training data). Used to map a single stock's raw value to a percentile rank when the full cross-section is not available.
- **`sector_map`** — ticker → GICS sector mapping for one-hot encoding
- **`last_obs_date`** — tracks how far the model has learned; prevents re-learning the same dates

### Data fetching and caching

`OnlineModelManager.get_data(through_date)` handles data sourcing:

1. First tries local S&P 500 price cache
2. If data is missing or stale, fetches from `yfinance`
3. Appends new rows to the persistent store
4. Returns a wide-format DataFrame (dates × tickers)

### Summary: pre-training vs online learning

```
Pre-training (static)
  Historical S&P 500 prices (up to 2024-08-01)
    → Walk-forward validation
    → Final model: 500 trees
    → Saved to disk

Online learning (incremental)
  Each new trading date:
    → Cross-sectional batch for prev day (~500 stocks)
    → +100 new trees via warm_start
    → Model grows: 600, 700, 800 trees ...
    → Prediction for target ticker
    → AnalystSignal returned to portfolio manager
```

The design keeps historical knowledge intact while continuously adapting to current market conditions, at low computational cost.
