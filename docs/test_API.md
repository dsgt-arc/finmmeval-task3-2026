# Local API Test Guide

This repository exposes a competition-style HTTP endpoint in `api/simple_trading_api.py`.
The endpoint is designed to receive the same JSON payload that the organizers send,
call the existing DeepFund workflow, and return exactly one trading signal:

```json
{"recommended_action": "BUY"}
```

or `HOLD` / `SELL`.

Before running the server or tests, install the project once with:

```bash
uv sync
```

If you have not created one yet, copy [.env.example](../.env.example) to `.env`
and set `OPENAI_API_KEY` so the default workflow can reach OpenAI.

## Prerequisites

- `make api-test` only needs the source code and the installed Python environment.
- `make api-integration-test` needs the downloaded parquet data in `data/data/` and a valid `OPENAI_API_KEY`.
- `make api-smoke` needs the same data and `.env`, because it starts the live server and sends a real request.
- The workflow will create the SQLite database automatically if it does not already exist.

## What Calls What

The request flow is:

1. The competition organizers send an HTTPS request to your API.
2. `api/simple_trading_api.py` receives the payload at `/competition_action/`.
3. The API forwards the payload to `api.decision_bridge.recommend_action(...)`.
4. `api/decision_bridge.py` launches `api/decision_bridge_worker.py` in a subprocess.
5. `api/decision_bridge_worker.py` initializes the existing `decision_making` workflow.
6. The workflow runs and writes the latest decision into its temporary SQLite DB.
7. The worker reads back the latest action and prints it as JSON.
8. The bridge normalizes the result and returns it to the API.
9. The API returns the final competition response JSON to the caller.

## Separation of Concerns

- `api/simple_trading_api.py`
  - HTTP layer only.
  - Validates the incoming JSON payload.
  - Exposes `/competition_action/`, `/trading_action/`, and `/health`.
  - Returns the response in the exact competition format.
  - This endpoint only returns the 3-way signal, not portfolio allocations or
    any secondary metrics.

- `api/decision_bridge.py`
  - Orchestration layer.
  - Keeps subprocess handling and timeout logic out of the HTTP file.
  - Converts the worker output into a valid action string.

- `api/decision_bridge_worker.py`
  - Workflow execution layer.
  - Runs the existing `decision_making` code without changing it.
  - Uses a temporary SQLite DB so each request is isolated.

## Reading The Tests

You do not need to know all of pytest to follow the tests in `tests/`.
The main ideas are:

- A `test_...` function is just a check.
- `assert` means "this must be true".
- A fixture like `client` or `sample_payload` is reusable setup.
- `monkeypatch` temporarily swaps in a fake value during the test.
- `TestClient` lets the tests call FastAPI like a real HTTP request.

The API tests are split into two layers:

- `tests/test_api.py` is the fast contract test.
- `tests/test_api_integration.py` is the real workflow test.
- `make api-smoke` is the live server smoke test you should run before deployment.

## Local Server Test

The two pytest commands above do not need a live server because they use `TestClient`
to call the app in process. Use the server step below only when you want to run a
real Uvicorn process and hit it with `curl`.

If you want a one-command live smoke test, run:

```bash
make api-smoke
```

Start the API server in one terminal:

```bash
uv run uvicorn api.simple_trading_api:app --host 127.0.0.1 --port 62237
```

Then, in a second terminal, send a competition-style request:

```bash
curl -X POST "http://127.0.0.1:62237/competition_action/" \
  -H "Content-Type: application/json" \
  -d '{
    "date": "2026-03-19",
    "price": {"TSLA": 380.29998779296875},
    "news": {"TSLA": ["Tesla-centred news on 2026-03-19"]},
    "symbol": ["TSLA"],
    "momentum": {"TSLA": "bearish"},
    "10k": {"TSLA": []},
    "10q": {"TSLA": []},
    "history_price": {
      "TSLA": [
        {"date": "2026-03-10", "price": 399.2349853515625},
        {"date": "2026-03-11", "price": 407.82000732421875},
        {"date": "2026-03-12", "price": 395.010009765625},
        {"date": "2026-03-13", "price": 391.20001220703125},
        {"date": "2026-03-14", "price": 391.20001220703125},
        {"date": "2026-03-15", "price": 391.20001220703125},
        {"date": "2026-03-16", "price": 395.55999755859375},
        {"date": "2026-03-17", "price": 399.2699890136719},
        {"date": "2026-03-18", "price": 392.7799987792969},
        {"date": "2026-03-19", "price": 380.29998779296875}
      ]
    }
  }'
```

Expected response:

```json
{"recommended_action":"BUY"}
```

## Handy Commands

If your system has `make`, these are the quickest shortcuts:

Fast API contract test:

```bash
make api-test
```

Real workflow integration test:

```bash
make api-integration-test
```

Run the local server:

```bash
make api-server
```

Live smoke test:

```bash
make api-smoke
```

If `make` is not installed, use the raw `uv run ...` commands shown above instead.

## Notes

- The endpoint now accepts optional `news`, `10k`, and `10q` fields.
- If `symbol` is missing, the API falls back to the symbol in the `price` map.
- If the workflow bridge times out or fails at runtime, the bridge defaults the
  response to `HOLD` to match the competition fallback rule.
- The subprocess bridge uses a 170-second timeout so it stays safely under the
  3-minute organizer limit.
- The server uses the `PORT` environment variable if your hosting platform provides one.
- For public deployment, you still need a stable HTTPS URL or container service.
