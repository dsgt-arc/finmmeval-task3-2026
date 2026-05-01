FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install uv so we can reproduce the local project environment from uv.lock.
RUN pip install --no-cache-dir uv

# Copy the project metadata first so dependency resolution can be cached.
COPY pyproject.toml uv.lock /app/
COPY api /app/api
COPY decision_making /app/decision_making
COPY data /app/data
COPY output /app/output
COPY run_download_ama_data.py /app/

# Install the project into the container. The competition data is baked into
# the image, so the API will not re-download it at runtime.
RUN uv sync --frozen \
    --no-install-package jupyterlab \
    --no-install-package pytest \
    --no-install-package pre-commit \
    --no-install-package ruff \
    --no-install-package matplotlib \
    --no-install-package pyspark \
    --no-install-package gensim \
    --no-install-package supabase \
    --no-install-package torch

EXPOSE 8080

CMD ["/app/.venv/bin/uvicorn", "api.simple_trading_api:app", "--host", "0.0.0.0", "--port", "8080"]
