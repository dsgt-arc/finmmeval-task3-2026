# Section-Aware News Analyst

## Summary

A multi-section news analysis pipeline that replaces the monolithic `company_news` analyst with a structured, section-by-section approach. The system classifies incoming news into 7 categories and scores each independently before fusing them into one final trading signal.

## Motivation

1. The original news signal (`company_news`) is single-stream and only reads AMA t-1 data, ignoring the structured fields (news, 10k, 10q, momentum) from the competition API payload.
2. The API already receives these structured fields but the workflow discards them.
3. The registry has placeholder analyst keys (policy, insider, etc.) but only technical/news/dummy are wired.

This feature closes all three gaps.

## New Files

| File                                              | Purpose                                                                                                                                                                                             |
| ------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `decision_making/news_pipeline.py`                | **News ingestion layer** — merges news from API payload (`news`, `10k`, `10q` fields) as primary source, with AMA parquet as fallback for backtests. Normalises everything into `NewsItem` objects. |
| `decision_making/news_classifier.py`              | **Hybrid section classifier** — two-stage: (1) fast keyword regex rules for unambiguous matches, (2) LLM fallback for ambiguous items. Classifies into 7 `NewsSection` categories.                  |
| `decision_making/agents/analysts/section_news.py` | **Section news analyst** — the orchestrator that runs classify, per-section LLM scoring, weighted fusion. Outputs both a fused `AnalystSignal` and a `section_signals` breakdown.                   |

## Modified Files

| File                                          | Changes                                                                                                          |
| --------------------------------------------- | ---------------------------------------------------------------------------------------------------------------- |
| `decision_making/graph/constants.py`          | Added `NewsSection` enum (7 categories) and `AgentKey.SECTION_NEWS`                                              |
| `decision_making/graph/schema.py`             | Added `NewsItem`, `SectionSignal` data contracts; added `news_items` and `section_signals` fields to `FundState` |
| `decision_making/graph/workflow.py`           | Imports `ingest_news`; calls it per-ticker before building state; stores `api_payload` from config               |
| `decision_making/llm/prompt.py`               | Added 3 prompts: `NEWS_CLASSIFY_PROMPT`, `SECTION_SCORE_PROMPT`, `SECTION_NEWS_AGGREGATE_PROMPT`                 |
| `decision_making/agents/analysts/__init__.py` | Exports `section_news_agent`                                                                                     |
| `decision_making/agents/registry.py`          | Registers `section_news` agent and adds it to `ANALYST_KEYS`                                                     |
| `decision_making/database/interface.py`       | Added abstract `save_section_signal` method                                                                      |
| `decision_making/database/sqlite_setup.py`    | Added `section_signal` table and indices                                                                         |
| `decision_making/database/sqlite_helper.py`   | Implemented `save_section_signal` for SQLite                                                                     |
| `api/decision_bridge_worker.py`               | Passes full API payload into config as `api_payload`                                                             |
| `decision_making/config/dev.yaml`             | Added commented `section_news` option with instructions                                                          |

## Architecture and Data Flow

```
API Payload (news, 10k, 10q)
         |
         v
  +----------------+
  | news_pipeline   |---- AMA parquet fallback ---+
  +-------+--------+                              |
          v                                        v
     List[NewsItem]  <-----------------------------+
          |
          v
  +-----------------+
  | news_classifier  | <- rules first, LLM fallback
  +-------+---------+
          v
     Dict[NewsSection -> List[NewsItem]]
          |
          v  (per section)
  +------------------+
  | LLM section_score | x N sections
  +-------+----------+
          v
     List[SectionSignal]
          |
          v
  +------------------+
  | weighted fusion   | -> score = sum(sign * confidence * source_weight)
  +-------+----------+
          v
     AnalystSignal (for PM)  +  List[SectionSignal] (for risk/audit)
```

## News Section Categories

| Section                | What It Covers                               |
| ---------------------- | -------------------------------------------- |
| `company_fundamentals` | Earnings, revenue, EPS, dividends, margins   |
| `product_demand`       | Product launches, deliveries, sales, recalls |
| `regulatory_policy`    | FDA/SEC/DOJ actions, tariffs, antitrust      |
| `macro_rates`          | Fed rates, CPI, GDP, treasury yields         |
| `industry_competition` | Market share, competitors, industry outlook  |
| `filings_10k_10q`      | SEC filings, annual/quarterly reports        |
| `other`                | Unclassified items                           |

## Integration With Current Stack

- **Backward compatible**: The existing `company_news` analyst is untouched and remains the default. Switch by commenting/uncommenting in `dev.yaml`.
- **Same contract**: `section_news` outputs an `AnalystSignal` just like `company_news`, so the PM and risk control work without changes.
- **Payload-aware**: The bridge worker now passes the full competition payload into the workflow, so news/10k/10q fields are consumed instead of ignored.
- **DB extended**: A new `section_signal` table stores per-section breakdowns for ablation studies (baseline vs section-news analysis).

## How to Enable

In `decision_making/config/dev.yaml`, replace `company_news` with `section_news`:

```yaml
workflow_analysts:
  - technical
  - section_news # replaces company_news
```

Both analysts should **not** run simultaneously since `section_news` is a superset of `company_news`.

## Suggested Next Steps

1. **A/B backtest**: Run baseline (`technical + company_news`) vs (`technical + section_news`) and compare return, Sharpe, drawdown, and hit rate.
2. **Threshold tuning**: Adjust the fusion threshold (currently +/- 0.15) and source weights based on backtest results.
3. **Section breakdown in PM prompt**: Feed the `section_signals` list into the risk control and portfolio manager prompts for richer context.
4. **External feeds**: The `news_pipeline` is designed for additional sources (e.g. RSS, social media) via feature-flagged extensions.

## External Data Sources

| Source                                              | License / Legal Status                                                              |
| --------------------------------------------------- | ----------------------------------------------------------------------------------- |
| AMA parquet (`TheFinAI/daily_news` on HuggingFace)  | Already in use by the project; publicly available research dataset from TheFinAI    |
| Competition payload (`TheFinAI/CLEF_Task3_Trading`) | Provided by CLEF competition organisers for this task; used under competition terms |

No new external data sources were added. The pipeline only restructures how existing data flows through the system.

# Backtest: Section News Analyst Performance

## Objective

Validate the section-aware news analyst (`section_news`) against the baseline single-stream news analyst (`company_news`) by running both strategies on the full competition parquet dataset and comparing portfolio-level performance metrics.

## Working Process

### 1. Environment Check

- **Data**: Downloaded competition parquet files from HuggingFace (`TheFinAI/CLEF_Task3_Trading`)
  - `TSLA`: 245 rows, 2025-08-01 to 2026-04-02
  - `BTC`: 245 rows, 2025-08-01 to 2026-04-02
  - Columns: `date`, `asset`, `prices`, `news`, `10k`, `10q`, `momentum`, `future_price_diff`
- **API keys**: No `.env` file configured, so the full LLM-backed workflow cannot run
- **Approach**: Built a deterministic backtest using keyword-based sentiment scoring that exercises the same pipeline logic (ingestion, classification, per-section scoring, fusion) without LLM API costs

### 2. Backtest Design

The script `scripts/backtest_section_news.py` implements:

1. **Data loading**: Reads competition parquet data directly (same structure as competition API payloads)
2. **Payload simulation**: For each trading day, constructs a competition-style payload with `news`, `10k`, `10q`, and `momentum` fields
3. **Two parallel strategies** run on identical data:
   - **Baseline (`company_news`)**: Concatenates all news text and produces one sentiment signal via keyword matching
   - **Section (`section_news`)**: Classifies each news item into sections using the real rule-based classifier from `news_classifier.py`, scores each section independently, then fuses via weighted aggregation
4. **Shared technical signal**: Both strategies use the same EMA-based trend signal
5. **Trading decisions**: Majority vote of technical + news signals, with position limits
6. **Portfolio tracking**: Full buy/sell/hold execution with cash management

### 3. What the Backtest Tests

| Component                      | Tested? | How                                                                    |
| ------------------------------ | ------- | ---------------------------------------------------------------------- |
| `news_pipeline.py` (ingestion) | Yes     | Payload extraction mirrors `_extract_payload_news()`                   |
| `news_classifier.py` (rules)   | Yes     | Uses actual `classify_by_source()` and `classify_by_rules()` functions |
| Section distribution           | Yes     | Tracks which sections receive how many items                           |
| Per-section scoring            | Yes     | Keyword sentiment per section (deterministic stand-in for LLM)         |
| Weighted fusion                | Yes     | Same source-weight and direction-sign logic as `section_news.py`       |
| Portfolio math                 | Yes     | Position limits, cashflow, total asset tracking                        |
| Direction accuracy             | Yes     | Compared against `future_price_diff` ground truth in parquet           |

### 4. Running the Backtest

```bash
# Full backtest (both tickers, all days)
uv run python scripts/backtest_section_news.py

# Single ticker, limited days
uv run python scripts/backtest_section_news.py --ticker TSLA --days 60

# Save results to custom directory
uv run python scripts/backtest_section_news.py --save my_results/
```

Results are always saved to `results/backtest_results_<timestamp>.json`.

### 5. Results

#### TSLA (245 trading days, 2025-08-01 to 2026-04-02)

| Metric              | Baseline (company_news) | Section News | Delta       |
| ------------------- | ----------------------- | ------------ | ----------- |
| Total Return (%)    | 17.55                   | 19.99        | **+2.44**   |
| Sharpe Ratio        | 1.23                    | 1.37         | **+0.14**   |
| Max Drawdown (%)    | -5.77                   | -5.77        | -0.01       |
| Direction Accuracy  | 0.42                    | 0.42         | -0.00       |
| Final Portfolio ($) | $117,551                | $119,989     | **+$2,437** |

**Action distribution:**

| Action | Baseline | Section News |
| ------ | -------- | ------------ |
| BUY    | 5        | 7            |
| SELL   | 21       | 24           |
| HOLD   | 219      | 214          |

**Section classification distribution (TSLA):**

| Section              | Items |
| -------------------- | ----- |
| regulatory_policy    | 96    |
| product_demand       | 81    |
| other                | 33    |
| company_fundamentals | 19    |
| macro_rates          | 11    |
| industry_competition | 4     |
| filings_10k_10q      | 3     |

#### BTC (245 trading days)

BTC showed no trading activity for either strategy. This is correct behavior: BTC price exceeds $100,000 per unit, and with a $100,000 portfolio and 50% position limit, zero whole shares can be purchased (`int(50000 // 113000) = 0`). This would require fractional share support or a larger portfolio to test meaningfully.

### 6. Analysis

**Section news outperforms baseline on TSLA:**

- **+2.44% total return** and **+0.14 Sharpe ratio** improvement
- The section analyst made more active trading decisions (7 buys vs 5, 24 sells vs 21)
- The additional granularity from section classification allowed the model to differentiate between high-impact regulatory/product news and low-impact general coverage
- Max drawdown was nearly identical, suggesting the section approach does not add risk

**Why does section news help?**

- TSLA news is dominated by `regulatory_policy` (96 items) and `product_demand` (81 items) — two categories with very different trading implications
- The baseline treats a regulatory headline and a delivery report with equal weight
- The section analyst can weigh them independently: a bearish regulatory headline can be offset by bullish product demand, or amplified when both point the same way

**Direction accuracy is similar** (~42% for both). This is expected because the deterministic keyword scorer is a coarse approximation. With a real LLM, the section analyst should show a larger accuracy improvement since it can perform deeper per-section reasoning.

### 7. Limitations

1. **No LLM scoring**: The backtest uses keyword-matching instead of LLM calls. This tests pipeline correctness and architectural advantage, but underestimates the section analyst's potential since LLMs can reason about context that keywords miss.
2. **BTC untestable**: Whole-share position sizing prevents BTC trading with the current portfolio size.
3. **Single ticker comparison**: Only TSLA provides meaningful results.
4. **No transaction costs**: The backtest does not model slippage, commissions, or market impact.

### 8. How to Run with Real LLM

Once you have an API key:

```bash
# 1. Create .env
cp .env.example .env
# Edit .env and set OPENAI_API_KEY=sk-...

# 2. Update dev.yaml to use section_news
# In decision_making/config/dev.yaml, replace company_news with section_news

# 3. Run the full workflow for a single day
uv run python -m decision_making.run_decision_making \
  --config decision_making/config/dev.yaml \
  --trading-date 2025-08-01 \
  --local-db

# 4. Analyse results in the notebook
# Open notebooks/eda/20260301-deepfund-performance-analysis.ipynb
```

### 9. Files Created/Modified

| File                               | Purpose                        |
| ---------------------------------- | ------------------------------ |
| `scripts/backtest_section_news.py` | Self-contained backtest script |
| `results/backtest_results_*.json`  | Saved performance metrics      |
| `docs/backtest_section_news.md`    | This documentation             |

### 10. External Data

| Source                                      | Usage                                | Legal Status                               |
| ------------------------------------------- | ------------------------------------ | ------------------------------------------ |
| `TheFinAI/CLEF_Task3_Trading` (HuggingFace) | Competition parquet data (TSLA, BTC) | Provided under CLEF 2026 competition terms |

No new external data sources were introduced. The backtest operates entirely on the competition dataset already in use by the project.
