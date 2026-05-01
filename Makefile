DEPLOY_CONFIG ?= decision_making/config/api.yaml
CLOUD_RUN_SERVICE ?= finmmeval-task3-2026
CLOUD_RUN_REGION ?= us-central1
CLOUD_RUN_PROJECT ?=
PROJECT_FLAG := $(if $(CLOUD_RUN_PROJECT),--project "$(CLOUD_RUN_PROJECT)")

.PHONY: setup api-test api-integration-test api-server api-smoke docker-build docker-run deploy-cloud-run

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

docker-build:
	docker build -t finmmeval-task3-2026 .

docker-run:
	docker run --rm -p 8080:8080 -e OPENAI_API_KEY="$OPENAI_API_KEY" finmmeval-task3-2026

deploy-cloud-run:
	uv run python scripts/deploy_cloud_run.py \
		--config "$(DEPLOY_CONFIG)" \
		--service "$(CLOUD_RUN_SERVICE)" \
		--region "$(CLOUD_RUN_REGION)" \
		$(PROJECT_FLAG) \
		--sync-secret
