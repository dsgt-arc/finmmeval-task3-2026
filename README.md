# finmmeval-task3-2026

DeepFund-style trading workflow plus a competition-facing HTTP API for Task 3.

This repository has two main pieces:
- `decision_making/` contains the existing trading workflow, data loading, models, and SQLite-backed state.
- `api/` contains the FastAPI wrapper that receives organizer payloads and returns the Task 3 signal: `BUY`, `HOLD`, or `SELL`.

## Quickstart

Install the project with `uv`:

```bash
uv sync
```

Before running the API or workflow, copy [.env.example](./.env.example) to `.env`
and fill in `OPENAI_API_KEY` if you plan to use the OpenAI-backed workflow.

Download the competition data first:

```bash
uv run python run_download_ama_data.py
```

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

## API

The competition endpoint lives in [docs/test_API.md](./docs/test_API.md).
That guide explains the request flow, the separation of concerns, and how to run the server locally.

The expected endpoint shape is:
- `POST /competition_action/`
- response: `{"recommended_action": "BUY"}` or `HOLD` / `SELL`

## Repository Layout

- `api/`: HTTP wrapper, bridge, and worker for the competition endpoint.
- `decision_making/`: trading workflow, agents, models, DB helpers, and analysis code.
- `tests/`: fast API tests plus an end-to-end workflow integration test.
- `docs/`: API testing notes and project documentation.
- `data/`: downloaded dataset files.
- `notebooks/`: exploratory analysis.

## Notes

- The API accepts optional `news`, `10k`, and `10q` fields.
- If `symbol` is missing, the API falls back to the key in `price`.
- If the workflow bridge times out or fails, the API defaults to `HOLD`.
- The subprocess bridge uses a 170-second timeout so it stays safely under the
  3-minute organizer limit.
- The server uses `PORT` when deployed to a host that provides one.
- The SQLite database is created automatically the first time you run the workflow.
- If the parquet files are missing, `run_download_ama_data.py` can rebuild them from Hugging Face.
- `docs/archive/README_ARCHIVE.md` preserves the original template README for reference.
- `CLAUDE.md` at the repo root contains the active working notes for the project.
- `pyproject.toml` is the single source of truth for dependencies.
- `uv sync` creates the local environment, and `uv run ...` executes commands inside it.
