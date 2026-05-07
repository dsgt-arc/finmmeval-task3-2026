# finmmeval-task3-2026 - DS@GT StockTron

DS@GT StockTron: a mulit-agent trading workflow plus a competition-facing HTTP API for Task 3 (CLEF 2026 FinMMEval). The DS@GT StockTron has been build by initially forking [DeepFund](https://github.com/HKUSTDial/DeepFund) and modiyfing and enhancing its workflows and agents.

## How It Works

1. **Data ingestion** — price history, news articles, and SEC filings are loaded for the target ticker and trading date
2. **Parallel analyst agents** — up to five specialist agents (`technical`, `company_news`, `company_news_enhanced`, `section_news`, `ml_model_online`) run concurrently, each emitting a `BULLISH / BEARISH / NEUTRAL` signal with confidence and rationale
3. **Portfolio manager synthesis** — an LLM-backed portfolio manager receives all signals and produces a final `BUY / HOLD / SELL` decision with position sizing (market-timing or risk-managed mode)
4. **State persistence** — decisions, signals, and portfolio state are written to SQLite for reproducibility and analysis
5. **API serving** — the FastAPI wrapper receives organizer payloads and returns the final action within the 3-minute competition deadline

This repository has following main pieces:
- `decision_making/` contains the existing trading workflow, data loading, models, and SQLite-backed state.
- `api/` contains the FastAPI wrapper that receives organizer payloads and returns the Task 3 signal: `BUY`, `HOLD`, or `SELL`.
- `notebooks/` contains jupyter notebooks on EDA and system evaluation
- `docs/` contains information on the api deployment, ml-model-tool included in the agent workflow
- `tests/` unit-tests for api and other utils

## Repository Structure

```
finmmeval-task3-2026/
├── api/                                    # FastAPI competition endpoint
│   ├── decision_bridge.py                  # Subprocess bridge into the workflow
│   ├── decision_bridge_worker.py
│   └── simple_trading_api.py               # Main API server
├── decision_making/                        # Core trading workflow
│   ├── agents/
│   │   ├── analysts/                       # Individual analyst agents
│   │   │   ├── company_news.py             # Baseline news agent
│   │   │   ├── company_news_enhanced.py    # Enhanced sentiment agent
│   │   │   ├── ml_model.py                 # RandomForest signal agent
│   │   │   ├── section_news.py             # Section-level news agent
│   │   │   └── technical.py               # Technical indicators agent
│   │   ├── portfolio_manager.py            # Final BUY/HOLD/SELL decision
│   │   └── registry.py
│   ├── config/                             # YAML workflow configs (one per experiment)
│   │   ├── api.yaml                        # Production default (all five agents)
│   │   ├── tesla_btc_baseline.yaml         # Minimal two-agent baseline
│   │   └── ablation_*.yaml                 # Ablation study configs
│   ├── database/                           # SQLite-backed state
│   │   ├── sqlite_setup.py
│   │   ├── sqlite_helper.py
│   │   └── interface.py
│   ├── graph/                              # LangGraph workflow definition
│   │   ├── workflow.py
│   │   ├── schema.py
│   │   └── constants.py
│   ├── llm/                                # LLM inference & prompt helpers
│   │   ├── inference.py
│   │   ├── prompt.py
│   │   └── provider.py
│   ├── ml_model/                           # RandomForest training & inference
│   │   ├── config.py                       # Hyperparameters & feature settings
│   │   ├── feature_engineering.py
│   │   ├── ml_model_manager.py
│   │   ├── models.py
│   │   ├── online_learning.py
│   │   └── validation.py
│   ├── news/                               # News classification pipeline
│   │   ├── classifier.py
│   │   └── pipeline.py
│   ├── analysis/                           # Performance analysis utilities
│   │   ├── performance.py
│   │   └── queries.py
│   ├── util/                               # Shared utilities (logging, config, db)
│   ├── ama_data.py                         # Competition data loader
│   ├── signals.py
│   ├── sp500_data.py                       # S&P 500 data loader (yfinance)
│   └── run_decision_making.py              # Workflow entry point
├── docs/
│   ├── deployment.md                       # Cloud Run deployment guide
│   ├── ml_model.md                         # ML model documentation
│   ├── overview.md
│   └── test_API.md                         # API usage & request examples
├── notebooks/
│   ├── 20260426-mh-market-timing-performance.ipynb   # Signal-based evaluation
│   ├── 20260427-mh-experiment-comparison.ipynb       # Multi-experiment comparison
│   └── 20261111-mh-eda.ipynb                         # Exploratory data analysis
├── scripts/
│   ├── run_date_range.sh                   # Backtest runner
│   ├── delete_config_history.sh            # Clear DB entries for a config
│   ├── run_download_ama_data.py            # Download competition data
│   ├── run_download_sp500_data.py          # Download S&P 500 training data
│   ├── train_ml_model_simple.py            # Train RandomForest (simple split)
│   ├── train_ml_model_tscv.py             # Train with time-series cross-validation
│   ├── deploy_cloud_run.py                 # Cloud Run deploy helper
│   └── smoke_api.sh                        # Live server smoke test
├── tests/
│   ├── test_api.py
│   ├── test_api_integration.py
│   └── test_llm_inference.py
├── output/                                 # Trained model artifacts (git-ignored)
├── data/                                   # Competition data (git-ignored)
├── Dockerfile
├── Makefile
├── pyproject.toml                          # Dependency source of truth (uv)
├── CLAUDE.md
├── .env.example
├── .pre-commit-config.yaml
└── README.md
```

# Quickstart

Install the project with `uv`:

```bash
uv sync
```

Before running the API or workflow, copy [.env.example](./.env.example) to `.env`
and fill in `OPENAI_API_KEY` if you plan to use the OpenAI-backed workflow.

Download the competition data first for local backtesting or development. Backtesting is the process of applying a trading strategy to historical data to estimate how it would have performed in the past.

```bash
uv run python scripts/run_download_ama_data.py
```

Additionally, we provide a machine-learning model as a tool to the multi-agent trading workflow. Here, we use external data (yfinance) to train a RandomForestClassifier based on SP500 stocks over the course of 2001 until the date of the backtest (2024-08-01) or the start of the competition 06.05.2026.

```bash
uv run python scripts/run_download_sp500_data.py
```

## Local Development & Testing

After creating the python environment, adding your e.g. OpenAI API Token, and downloading the data you may develop or backtest DS@GT StockTron multi-agent trading workflow.

Initially, train the machine-learning model via:

```bash
uv run python scripts/train_ml_model_simple.py
```

The artifacts of the trained model are stored in `output/`. If you are interested to change any hyperparameters or other machine-learning model settings, please adjust them in `decision_making/ml_model/config.py`. You can also learn more about the model and other optionalities via `docs/ml_model.md`.

### Multi-Agent Trading Configs

Each YAML file in `decision_making/config/` defines a trading workflow by specifying which analyst agents to activate (`workflow_analysts`), the target tickers, and LLM settings. `api.yaml` is the production default and runs all five agents in parallel (`technical`, `company_news`, `company_news_enhanced`, `ml_model_online`, `section_news`). The `tesla_btc_baseline.yaml` provides a minimal two-agent baseline for comparison, while configs prefixed with `ablation_` isolate individual agents to measure each one's contribution to the trading signal.

Hence, set up a config in decision_making/config or alternatively select a config from us. Run the backtest via:

```bash
uv run scripts/run_date_range.sh decision_making/config/YOUR_CONFIG_SELECTED.yaml
```

The DS@GT StockTron will now run from 2024-08-01 until the last date available in competition data. If you downloaded the competition data on day T, then the competition data is T-1. All results and agent decisions will be stored at the sqlite database `data/dsgt_stocktron.db`. Feel free to have a look inside the sqlite database, for example with DBeaver.

### Evaluation

Finally, you may use the notebook `notebooks/20260426-mh-market-timing-performance.ipynb` to evaluate the pure-signal based performance (in spirit of the FinMMEval Task 3). When you want to compare multiple backtest, use `notebooks/20260427-mh-experiment-comparison.ipynb` instead.


# Deployment

Run the workflow smoke test:

```bash
uv run python decision_making/run_decision_making.py --config decision_making/config/debug_pm.yaml --trading-date 2026-03-19 --local-db
```

Run the API tests:

```bash
make api-test
make api-integration-test
```

Run a live server smoke test before deployment:

```bash
make api-smoke
```

Start the local API server:

```bash
make api-server
```

# API

The competition endpoint lives in [docs/test_API.md](./docs/test_API.md).
That guide explains the request flow, the separation of concerns, and how to run the server locally.

The expected endpoint shape is:
- `POST /competition_action/`
- response: `{"recommended_action": "BUY"}` or `HOLD` / `SELL`

# Docker Deployment

The simplest deployment path is a Docker image plus Google Cloud Run. The
recommended helper script reads the active workflow config, trains ML artifacts
only when the workflow actually needs them, stages a minimal source tree, and
injects `OPENAI_API_KEY` from Secret Manager instead of baking it into the
image.
The current deploy helper stages a minimal source tree and lets Cloud Run do
the image build and rollout in-region.

See [docs/deployment.md](./docs/deployment.md) for the full flow.

Build the image locally:

```bash
docker build -t finmmeval-task3-2026 .
```

Run it locally:

```bash
docker run --rm -p 8080:8080 \
  -e OPENAI_API_KEY="$OPENAI_API_KEY" \
  finmmeval-task3-2026
```

Test it from the same machine:

```bash
curl http://127.0.0.1:8080/health
```

Test it from a different computer:

```bash
curl http://<server-ip-or-domain>:8080/health
```

If you later want to change the deployed analyst mix, update
[`decision_making/config/api.yaml`](./decision_making/config/api.yaml) before
building the image again, or deploy with:

```bash
uv run python scripts/deploy_cloud_run.py --config decision_making/config/api.yaml --sync-secret
```

That still targets the same single deployment; the config only changes which
analysts and artifacts are bundled into that service.
Cloud Run rolls the service to the new revision, so you do not need to stop the
existing service first.

# Results

All experiments were backtested on TSLA and BTC from **2024-08-02 to 2026-01-09** (~361 trading days) using `gpt-4.1-nano` as the LLM. The buy-and-hold benchmarks over this period were **+114.3 % (TSLA)** and **+47.1 % (BTC)**.

### Ablation Study Overview

Each experiment isolates one or two analyst agents to measure their individual contribution to the trading signal. The portfolio manager receives their outputs and produces a final `BUY / HOLD / SELL` decision. All experiments below use **market timing** mode (pure signal, no paper portfolio):

| # | Config | Agents | Purpose |
|---|--------|--------|---------|
| Base | `tesla_btc_baseline` | `technical` + `company_news` | Baseline from DeepFund — standard news + technicals |
| 1 | `ablation_company_news_enhanced` | `technical` + `company_news_enhanced` | Swap basic news for enhanced (relevance-check, sentiment-per-section, sentiment-change, sentiment trade) news agent |
| 2 | `ablation_section_news` | `technical` + `section_news` | Swap basic news for section-level news agent |
| 3 | `ablation_ml_model` | `company_news` + `ml_model_agent_online` | Replace technical agent with online RandomForest signal |
| 4 | `enhanced_tools_tsla_btc` | `company_news_enhanced` + `ml_model_agent_online` | Combine enhanced news with ML signal (no baseline) |
| 5 | `all_tools_tsla_btc` | all five agents | Full ensemble — all agents in parallel |
| 6 | `ablation_enhanced_memory` | `technical` + `company_news` + enriched memory (5 d) | Add 5-day hit-rate memory to portfolio manager |
| 7 | `ablation_enhanced_memory_longer` | `technical` + `company_news` + enriched memory (10 d) | Extend hit-rate memory window to 10 days |

Two additional experiments use **paper portfolio** mode (`risk_managed`), which sizes positions via a paper share calculation rather than emitting pure signals:

| # | Config | Agents | Purpose |
|---|--------|--------|---------|
| 8 | `tesla_btc_baseline_risk` | `technical` + `company_news` | Baseline with paper portfolio |
| 9 | `enhanced_tools_tsla_btc_risk` | `company_news_enhanced` + `ml_model_agent_online` | Best pure-signal setup with paper portfolio |

### Key Metrics (market-timing experiments)

| Experiment | TSLA CR | BTC CR | TSLA SR | BTC SR | TSLA MD | BTC MD |
|---|---:|---:|---:|---:|---:|---:|
| `tesla_btc_baseline` | −54.5 % | −18.0 % | −0.76 | −0.29 | −65.8 % | −32.5 % |
| `ablation_company_news_enhanced` | **−23.5 %** | −18.5 % | **−0.13** | −0.30 | −53.7 % | −40.6 % |
| `ablation_ml_model` | −23.8 % | +0.8 % | −0.45 | +0.15 | −41.5 % | −26.7 % |
| `ablation_section_news` | −41.6 % | +2.7 % | −0.47 | +0.21 | −52.3 % | −26.8 % |
| `enhanced_tools_tsla_btc` | −31.9 % | +11.2 % | −0.37 | +0.40 | −74.2 % | −18.4 % |
| `all_tools_tsla_btc` | −39.5 % | **+20.1 %** | −0.42 | **+0.54** | −61.5 % | −25.7 % |
| `ablation_enhanced_memory` (5 d) | −61.8 % | −15.9 % | −1.14 | −0.27 | −88.2 % | −32.5 % |
| `ablation_enhanced_memory_longer` (10 d) | −50.5 % | −7.0 % | −0.85 | −0.04 | −63.2 % | −25.6 % |

CR = Cumulative Return · SR = Sharpe Ratio · MD = Max Drawdown

### ML Model — Out-of-Sample Validation

The RandomForest signal agent was trained on all S&P 500 stocks from 2001 to 2019 (~2.6 M stock-day observations) and evaluated on a held-out test split (2019-11 onward, never seen during training).

| Metric | Value |
|--------|------:|
| Accuracy | 51.5 % |
| Precision | 52.3 % |
| Recall | 75.2 % |
| F1 | 61.7 % |
| AUC-ROC | 50.8 % |

A ~51–52 % directional accuracy is consistent with the financial ML literature. Gu, Kelly & Xiu (2020, *Review of Financial Studies*) show that tree-based models including random forests achieve statistically significant out-of-sample return predictability on U.S. equities, while Krauss, Do & Huck (2017, *European Journal of Operational Research*) report ~52–53 % next-day directional accuracy for random forests on S&P 500 stocks — closely matching our result. In efficient markets, any persistent edge above 50 % is non-trivial; the value compounds across thousands of predictions. The high recall (75 %) indicates the model skews toward calling `BULLISH`, which partially explains why BTC configurations (a trending asset over the test period) benefit more from the ML signal than TSLA. In the backtest of the whole DS@GT StockTron sytem, the machine-learning model makes **strictly out-of-sample** predictions for the entire backtest period starting 2024-08-01.

### Key Findings

- **BTC responds better to agent signals than TSLA.** The full ensemble (`all_tools_tsla_btc`) achieves +20.1 % CR and SR +0.54 on BTC, while no configuration produces positive TSLA returns.
- **ML + enhanced news is the strongest combination.** `enhanced_tools_tsla_btc` yields the best BTC Sharpe (+0.40) and lowest BTC drawdown (−18.4 %) among single-combination configs.
- **Enriched decision memory hurts.** Both memory ablations (5 d and 10 d) worsen TSLA performance significantly (SR down to −1.14), suggesting the hit-rate signal causes the PM to over-extrapolate short-term streaks.
- **All strategies underperform buy-and-hold** in this bull-market period — a known limitation of market-timing approaches in trending regimes.

For full plots (cumulative return curves, drawdown, rank heatmap) see `notebooks/20260427-mh-experiment-comparison.ipynb`.

### Research Questions

**RQ1: Does incorporating a predictive time-series ML model as an agent tool improve investment performance?**
Partially. In ablation studies, we see that replacing the technical agent with `ml_model_online` consistently improves BTC performance (SR −0.29 → +0.15 paired with basic news; −0.30 → +0.40 paired with enhanced news), but TSLA results are mixed: the ML signal helps alongside naive news (+0.31 SR delta) yet slightly hurts alongside enhanced news (−0.24 SR delta). The benefit is asset-dependent, likely driven by the model's BULLISH bias aligning better with BTC's trend than TSLA's idiosyncratic volatility.

**RQ2: Does sophisticated sentiment analysis improve performance over naive aggregation?**
Yes. Both enhanced approaches outperform the baseline (`company_news`). In ablation studies, we see that switching to `company_news_enhanced` dramatically improves TSLA (SR −0.76 → −0.13) with negligible BTC impact, while `section_news` substantially improves BTC (CR −18.0 % → +2.7 %, SR −0.29 → +0.21) and moderately improves TSLA. Neither advanced variant degrades both assets simultaneously, confirming that more sophisticated sentiment signals are reliably additive.


# Future Work

The experiments above reveal several directions that could meaningfully extend this system:

- **Expand asset coverage and time horizon.** The competition and backtests are limited to TSLA and BTC, which constrains evaluation to market-timing rather than stock selection. The short backtest horizon further limits confidence in results, as agent signal quality are better assessed over multi-year periods that capture full market cycles. Broadening to a larger universe of US equities with multi-decade data would enable a multi-agent stock selection workflow grounded in factor investing — an approach with strong empirical support for generating excess returns over long horizons ([Fama & French, 1993](https://www.sciencedirect.com/science/article/abs/pii/0304405X93900235); [Fama & French, 2015](https://www.sciencedirect.com/science/article/abs/pii/S0304405X14002323)). It would also help distinguish whether observed underperformance is idiosyncratic to TSLA or reflects a structural limitation in how agent signals generalize across asset classes.

- **Stronger LLM backbone.** All experiments use `gpt-4.1-nano` for cost efficiency. Evaluating larger models (e.g., `gpt-5.5`, `Claude Opus 4.7`) could improve portfolio manager reasoning quality, particularly for multi-signal reconciliation on ambiguous days.

- **Alternative ML models.** The RandomForest achieves 51.5 % accuracy with a high-recall BULLISH skew. Neurel network (ensembles) could improve prediction quality; additionally, training on crypto-specific features rather than S&P 500 cross-sections may better serve BTC.

- **Redesigned memory mechanism.** Both enriched-memory ablations (5-day and 10-day hit-rate windows) significantly hurt TSLA performance (SR down to −1.14). Alternative designs worth exploring include attention-weighted decision history.

# Notes

- The API accepts optional `news`, `10k`, and `10q` fields.
- Those optional context fields can also be omitted or set to `null` on sparse days.
- We added regression tests for TSLA with filings present and BTC with `10k` / `10q` set to `null`.
- If `symbol` is missing, the API falls back to the key in `price`.
- The API uses [decision_making/config/api.yaml](./decision_making/config/api.yaml) by default.
  Edit that file directly to change the deployed analyst workflow, or set
  `DECISION_BRIDGE_CONFIG` to point at a different YAML file.
- If the workflow bridge times out or fails, the API defaults to `HOLD`.
- The subprocess bridge uses a 170-second timeout so it stays safely under the
  3-minute organizer limit.
- The server uses `PORT` when deployed to a host that provides one.
- The SQLite database is created automatically the first time you run the workflow.
- If the parquet files are missing, `scripts/run_download_ama_data.py` can rebuild them from Hugging Face.
- `docs/archive/README_ARCHIVE.md` preserves the original template README for reference.
- `CLAUDE.md` at the repo root contains the active working notes for the project.
- `pyproject.toml` is the single source of truth for dependencies.
- `uv sync` creates the local environment, and `uv run ...` executes commands inside it.
