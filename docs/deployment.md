# Deployment Guide

You will operate a single deployment. The workflow config determines whether
that one service runs the plain API analysts or the ML-enabled analyst set:

- **Plain API config**: the default `decision_making/config/api.yaml` workflow
  runs `technical` and `company_news`.
- **ML-enabled config**: if the selected YAML includes `ml_model_agent_online`,
  the deployment helper will make sure `output/rf_return_model/` exists and
  will train the missing artifacts before packaging the service.

The simplest production path is Google Cloud Run with one warm instance, a
secret-backed `OPENAI_API_KEY`, and a Docker image built from this repo.
The helper stages a minimal build context locally and hands that staged source
to Cloud Run, which performs the container build and then rolls the service to
the new revision.

## Required Local Secrets

Copy [.env.example](../.env.example) to `.env` and fill in:

- `OPENAI_API_KEY` - required for the default OpenAI-backed workflow.

Optional:

- `HF_TOKEN` - only useful if you expect runtime Hugging Face downloads.

Do not put `OPENAI_API_KEY` into the Docker image or commit it to git. The
deployment script can upload it to Google Secret Manager for you.

## Simplest Deploy Command

From the repository root:

```bash
uv run python scripts/deploy_cloud_run.py --sync-secret
```

Or via `make`:

```bash
make deploy-cloud-run
```

Both commands:

1. Read `decision_making/config/api.yaml` by default.
2. Inspect `workflow_analysts`.
3. Train `output/rf_return_model/` only if the config includes
   `ml_model_agent_online` and the artifact is missing.
4. Sync `OPENAI_API_KEY` into Google Secret Manager.
5. Stage a minimal container build context.
6. Ask Cloud Run to build the image from that staged source.
7. Deploy the resulting revision with the secret injected as an environment
   variable.
8. Keep one warm instance running with `--min-instances=1` so the first request
   does not pay the full cold-start penalty.

This is a rolling deploy. You do not need to stop the existing service first;
Cloud Run keeps serving traffic while the new revision comes up.

## Custom Config

If you later decide to redeploy the same service with a different workflow
file, point the script at it:

```bash
uv run python scripts/deploy_cloud_run.py \
  --config decision_making/config/debug_ml.yaml \
  --service finmmeval-task3-api \
  --region us-central1 \
  --sync-secret
```

Only config files inside this repository are accepted. That keeps the build
context predictable and ensures the file exists inside the container image.

## What Gets Baked Into The Image

- the Python app code
- the competition parquet files under `data/data/`
- `output/rf_return_model/` when the ML analyst is enabled and the artifact is
  needed

The competition data is not re-downloaded on every run because the loader uses
the local files first. If those files are missing, the existing download helper
can still fetch them, but the deployment flow is meant to avoid that during
normal runs.

## Safe Secret Injection On Google Cloud

The deployment helper uses a Secret Manager flow instead of baking secrets into
the image:

1. Read `OPENAI_API_KEY` locally from `.env` or the shell environment.
2. Create or update the Google Cloud secret named `OPENAI_API_KEY`.
3. Grant the Cloud Run runtime service account access to that secret.
4. Deploy the Cloud Run service with `--set-secrets OPENAI_API_KEY=...`.

That means the key never appears in the container layers and does not need to
be stored in plaintext in Cloud Run service settings.

If you already created the secret manually, you can drop `--sync-secret` and
reuse the existing Cloud Run secret version.

## Testing From Another Computer

After deployment, the script will print the service URL. From a different
machine, test it like this:

```bash
curl https://<service-url>/health
```

And then send a competition payload:

```bash
curl -X POST "https://<service-url>/competition_action/" \
  -H "Content-Type: application/json" \
  -d '{
    "date": "2026-03-19",
    "price": {"TSLA": 380.30},
    "news": {"TSLA": ["Tesla-centred news on 2026-03-19"]},
    "symbol": ["TSLA"],
    "momentum": {"TSLA": "bearish"},
    "10k": {"TSLA": []},
    "10q": {"TSLA": []},
    "history_price": {
      "TSLA": [
        {"date": "2026-03-18", "price": 392.78},
        {"date": "2026-03-19", "price": 380.30}
      ]
    }
  }'
```

## Environment Variables

The deployed container only needs a small set of environment variables:

- `OPENAI_API_KEY` - injected from Secret Manager.
- `DECISION_BRIDGE_CONFIG` - optional override if you deploy a non-default YAML.
- `PORT` - provided by the host platform; the app reads it automatically.

Everything else is either baked into the image or handled internally by the
workflow.
