# ML Model

The ML model is a **Random Forest binary classifier** that predicts whether a stock's next-day return will be positive or negative, producing `BULLISH`, `BEARISH`, or `NEUTRAL` analyst signals.

---

## Model Summary

| Aspect | Detail |
|---|---|
| Model type | `RandomForestClassifier` (sklearn) |
| Target | Binary: 1 if next-day return > 0, else 0 |
<<<<<<< HEAD
| Output | Probability of positive return → converted to analyst signal |
| Training data cutoff | `COMPETITION_TRAIN_START = "2026-05-04"` (config.py) |
| Ensemble size | 500 trees (initial), grows during online learning |
=======
| Training data cutoff | `COMPETITION_TRAIN_START = "2024-08-01"` |
| Initial ensemble size | 500 trees, grows during online learning |
>>>>>>> bb9489079d108724c2c889f5963c9649f424cf8d

---

## Features

| Group | Features |
|---|---|
| Lagged returns | `return_lag_1`, `return_lag_5`, `return_lag_21` |
| Technical | `rsi_14`, `bb_position`, `volatility_21d`, `ema_8/21/55`, EMA crossover flags |
| Cross-sectional | `return_rank_pct`, `volatility_rank_pct`, sector one-hot dummies (11 GICS) |

---

## Training

Run the training script before deploying:

```bash
uv run python scripts/train_ml_model_simple.py   # fast, single split
uv run python scripts/train_ml_model_tscv.py     # walk-forward validation
```

Both scripts train on all data up to `COMPETITION_TRAIN_START` and save artifacts to `output/rf_return_model/`:

```
output/rf_return_model/
├── rf_return_model.pkl       # serialized model
├── rf_return_model.json      # metadata + reference data for inference
├── feature_importance.csv/png
└── val_metrics.png
```

If no saved model exists when the agent is first called, `get_model_manager()` trains automatically on first use — convenient during development but slow.

### Walk-forward validation

The TSCV script uses an expanding window to prevent look-ahead bias:

```
Fold 1: Train [2001 → 2008]  |  embargo (21 days)  |  Test (3 months)
Fold 2: Train [2001 → mid-2008]  |  embargo  |  Test (3 months)
...
```

The **21-day embargo** between train and test is critical: because `return_lag_21` looks back 21 days, without a gap the test features would overlap with training observations, causing leakage.

```
Without embargo:
  Train ends 2020-12-31, test starts 2021-01-01
  return_lag_21 on 2021-01-01 uses 2020-12-10 → 2020-12-31 ← inside training set ✗

With embargo (21 days):
  Train ends 2020-12-31, test starts 2021-01-22
  return_lag_21 on 2021-01-22 uses 2020-12-31 → 2021-01-21 ← outside training set ✓
```

Key config parameters (all in `ml_model/config.py`):

| Parameter | Value | Meaning |
|---|---|---|
| `initial_train_years` | 7 | Length of first training window |
| `step_months` | 6 | How far each fold advances |
| `test_months` | 3 | Test window per fold |
| `embargo_days` | 21 | Must equal `max(LAGS)` |

---

## Online Learning

After the initial model is loaded, `ml_model_agent_online()` extends it on every trading date using sklearn's `warm_start`:

```python
model.n_estimators += N_NEW_TREES_CROSS_SECTIONAL  # e.g. 500 → 600
model.warm_start = True
model.fit(X_batch, y_batch)   # only new trees are trained; old trees preserved
```

**Each trading date runs two steps:**

1. **Cross-sectional update** — builds features for all ~500 S&P 500 stocks on `trading_date - 1`, computes targets, and adds 100 new trees. Using the full cross-section gives far better statistical power than a single stock.

2. **Single-stock prediction** — builds features for the target ticker, calls `predict_proba()`, and maps the result to a signal:
   - `P > 0.55` → `BULLISH`
   - `P < 0.45` → `BEARISH`
   - Otherwise → `NEUTRAL`

<<<<<<< HEAD
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
  Historical S&P 500 prices (up to 2026-05-04)
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
=======
**Feature consistency** during inference is maintained via `reference_data` stored in `rf_return_model.json`: pre-computed percentile distributions and a sector map that mirror the training-time feature engineering.
>>>>>>> bb9489079d108724c2c889f5963c9649f424cf8d
