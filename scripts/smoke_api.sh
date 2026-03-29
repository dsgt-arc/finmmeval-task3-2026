#!/usr/bin/env bash

set -euo pipefail

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-62237}"
SYMBOL="${1:-TSLA}"
TRADING_DATE="${2:-2026-03-19}"

cleanup() {
  if [[ -n "${SERVER_PID:-}" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
}

trap cleanup EXIT

server_log="$(mktemp)"

uv run uvicorn api.simple_trading_api:app --host "$HOST" --port "$PORT" >"$server_log" 2>&1 &
SERVER_PID=$!

for _ in {1..60}; do
  if curl -fsS "http://$HOST:$PORT/health" >/dev/null 2>&1; then
    break
  fi
  sleep 1
done

if ! curl -fsS "http://$HOST:$PORT/health" >/dev/null 2>&1; then
  echo "Server did not become ready. Logs:" >&2
  cat "$server_log" >&2
  exit 1
fi

payload="$(uv run python - "$SYMBOL" "$TRADING_DATE" <<'PY'
import json
import sys
from decision_making.data import load_data

symbol = sys.argv[1]
trading_date = sys.argv[2]

df = load_data(symbol, download_if_missing=False, competition_data=True)
row = df.filter(df["date"] == trading_date).to_dicts()[0]
history = [
    {"date": item["date"], "price": item["prices"]}
    for item in df.filter(df["date"] <= trading_date).select(["date", "prices"]).tail(10).to_dicts()
]

payload = {
    "date": trading_date,
    "price": {symbol: row["prices"]},
    "news": {symbol: row["news"]},
    "symbol": [symbol],
    "momentum": {symbol: row["momentum"]},
    "10k": {symbol: row["10k"]},
    "10q": {symbol: row["10q"]},
    "history_price": {symbol: history},
}

print(json.dumps(payload))
PY
)"

echo "Posting competition payload to http://$HOST:$PORT/competition_action/"
response="$(curl -fsS -X POST "http://$HOST:$PORT/competition_action/" -H "Content-Type: application/json" -d "$payload")"
echo "$response"
