.PHONY: setup api-test api-integration-test api-server api-smoke

setup:
	uv sync

api-test:
	uv run pytest -q tests/test_api.py

api-integration-test:
	uv run pytest -q tests/test_api_integration.py -m integration

api-server:
	uv run uvicorn api.simple_trading_api:app --host 127.0.0.1 --port 62237

api-smoke:
	bash scripts/smoke_api.sh
