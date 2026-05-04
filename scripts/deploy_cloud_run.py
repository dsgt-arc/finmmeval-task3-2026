"""Package and deploy the API to Google Cloud Run.

This helper keeps the deployment flow simple:
- read the chosen workflow config
- check whether the workflow uses the online ML analyst
- train/bake ML artifacts only when they are actually needed
- optionally sync OPENAI_API_KEY into Google Secret Manager
- stage a minimal container build context and hand it to Cloud Run
- deploy with the secret injected at runtime

The script is intentionally opinionated so it is easy to run from a clean
machine with just `uv` and `gcloud` available.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Iterable

import yaml
from dotenv import load_dotenv

DEFAULT_CONFIG = Path("decision_making/config/api.yaml")
DEFAULT_REGION = "us-central1"
DEFAULT_SERVICE = "finmmeval-task3-2026"
DEFAULT_SECRET_NAME = "OPENAI_API_KEY"
ML_ANALYST_NAME = "ml_model_agent_online"
ML_MODEL_DIR = Path("output") / "rf_return_model"
ML_REQUIRED_FILES = [
    ML_MODEL_DIR / "rf_return_model.pkl",
    ML_MODEL_DIR / "rf_return_model.json",
]
COMMON_COPY_PATHS = [
    Path("Dockerfile"),
    Path(".dockerignore"),
    Path("pyproject.toml"),
    Path("uv.lock"),
    Path("api"),
    Path("decision_making"),
    Path("run_download_ama_data.py"),
]
ARTIFACT_REPO = "cloud-run-source-deploy"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train missing artifacts and deploy the API to Cloud Run.")
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG),
        help="Workflow config to deploy. Must live inside the repository.",
    )
    parser.add_argument("--service", default=DEFAULT_SERVICE, help="Cloud Run service name.")
    parser.add_argument("--region", default=DEFAULT_REGION, help="Cloud Run region.")
    parser.add_argument("--project", default=None, help="Optional Google Cloud project override.")
    parser.add_argument(
        "--secret-name",
        default=DEFAULT_SECRET_NAME,
        help="Secret Manager secret name that stores OPENAI_API_KEY.",
    )
    parser.add_argument(
        "--sync-secret",
        action="store_true",
        help="Create or update the Cloud Run secret from the local OPENAI_API_KEY value.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show the planned steps without running training or deployment.",
    )
    return parser.parse_args()


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_repo_path(path_str: str) -> Path:
    root = repo_root()
    path = Path(path_str)
    if not path.is_absolute():
        path = (root / path).resolve()
    else:
        path = path.resolve()

    try:
        path.relative_to(root)
    except ValueError as exc:
        raise SystemExit(f"Config path must live inside the repository: {path}") from exc
    return path


def load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise SystemExit(f"Config file must contain a YAML mapping: {config_path}")
    return data


def workflow_analysts(config: dict) -> list[str]:
    analysts = config.get("workflow_analysts", [])
    if not isinstance(analysts, list):
        raise SystemExit("workflow_analysts must be a YAML list")
    return [str(item) for item in analysts]


def needs_ml_artifacts(analysts: Iterable[str]) -> bool:
    return ML_ANALYST_NAME in set(analysts)


def ensure_output_dir() -> None:
    (repo_root() / "output").mkdir(parents=True, exist_ok=True)


def get_effective_project(project: str | None) -> str:
    if project:
        return project
    current = subprocess.check_output(["gcloud", "config", "get-value", "project"], text=True).strip()
    if not current or current == "(unset)":
        raise SystemExit("No Google Cloud project is configured. Use `gcloud config set project ...` or pass --project.")
    return current


def get_project_number(project: str | None) -> str:
    effective_project = get_effective_project(project)
    output = subprocess.check_output(
        ["gcloud", "projects", "describe", effective_project, "--format=value(projectNumber)"],
        text=True,
    ).strip()
    if not output:
        raise SystemExit(f"Could not determine the project number for {effective_project}")
    return output


def ensure_secret_accessor_binding(secret_name: str, project: str | None) -> None:
    project_number = get_project_number(project)
    member = f"serviceAccount:{project_number}-compute@developer.gserviceaccount.com"
    secret_cmd = [
        "gcloud",
        "secrets",
        "add-iam-policy-binding",
        secret_name,
        "--member",
        member,
        "--role",
        "roles/secretmanager.secretAccessor",
    ]
    if project:
        secret_cmd += ["--project", project]
    run_command(secret_cmd)

    project_cmd = [
        "gcloud",
        "projects",
        "add-iam-policy-binding",
        get_effective_project(project),
        "--member",
        member,
        "--role",
        "roles/secretmanager.secretAccessor",
    ]
    run_command(project_cmd)


def stage_deploy_context(source_root: Path, *, include_ml_assets: bool) -> Path:
    """Create a clean build context with only the runtime assets we need."""

    staged_root = Path(tempfile.mkdtemp(prefix="finmmeval-cloudrun-"))
    copy_paths = list(COMMON_COPY_PATHS)
    copy_paths.append(Path("data") / "data")
    if include_ml_assets:
        copy_paths.extend([Path("data") / "data_sp500", Path("output")])
    else:
        copy_paths.append(Path("output"))

    for relative in copy_paths:
        src = source_root / relative
        dst = staged_root / relative
        if src.is_dir():
            shutil.copytree(src, dst)
        elif src.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
        elif relative == Path("output"):
            dst.mkdir(parents=True, exist_ok=True)
        else:
            raise SystemExit(f"Deployment source file missing: {src}")

    return staged_root


def run_command(cmd: list[str], *, input_text: str | None = None) -> None:
    print("+", " ".join(cmd))
    subprocess.run(
        cmd,
        check=True,
        input=input_text.encode() if input_text is not None else None,
    )


def maybe_train_ml_artifacts(*, dry_run: bool) -> None:
    missing = [path for path in ML_REQUIRED_FILES if not (repo_root() / path).exists()]
    if not missing:
        print("ML artifacts already present under output/rf_return_model/.")
        return

    print("ML analyst detected and artifacts are missing; training them now.")
    if dry_run:
        print("Dry run: would execute scripts/train_return_model_simple.py")
        return

    training_script = repo_root() / "scripts" / "train_return_model_simple.py"
    run_command([sys.executable, str(training_script)])

    still_missing = [path for path in ML_REQUIRED_FILES if not (repo_root() / path).exists()]
    if still_missing:
        missing_text = ", ".join(str(path) for path in still_missing)
        raise SystemExit(f"ML training finished but the following artifacts are still missing: {missing_text}")


def ensure_gcloud() -> None:
    if shutil.which("gcloud") is None:
        raise SystemExit("gcloud is required for deployment but was not found on PATH")


def ensure_docker() -> None:
    if shutil.which("docker") is None:
        raise SystemExit("docker is required for deployment but was not found on PATH")


def ensure_secret_exists(secret_name: str, project: str | None) -> bool:
    cmd = ["gcloud", "secrets", "describe", secret_name]
    if project:
        cmd += ["--project", project]
    result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
    return result.returncode == 0


def sync_secret(secret_name: str, project: str | None) -> None:
    secret_value = os.environ.get(DEFAULT_SECRET_NAME)
    if not secret_value:
        raise SystemExit(
            f"{DEFAULT_SECRET_NAME} is not set in the environment. "
            "Copy .env.example to .env, fill in the key, and rerun with --sync-secret."
        )

    if not ensure_secret_exists(secret_name, project):
        cmd = ["gcloud", "secrets", "create", secret_name, "--replication-policy=automatic"]
        if project:
            cmd += ["--project", project]
        run_command(cmd)

    cmd = ["gcloud", "secrets", "versions", "add", secret_name, "--data-file=-"]
    if project:
        cmd += ["--project", project]
    run_command(cmd, input_text=secret_value)


def ensure_artifact_registry_repo(*, project: str | None, region: str) -> None:
    cmd = [
        "gcloud",
        "artifacts",
        "repositories",
        "describe",
        ARTIFACT_REPO,
        "--location",
        region,
    ]
    if project:
        cmd += ["--project", project]
    result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
    if result.returncode == 0:
        return

    cmd = [
        "gcloud",
        "artifacts",
        "repositories",
        "create",
        ARTIFACT_REPO,
        "--repository-format=docker",
        "--location",
        region,
        "--description",
        "Container images for finmmeval-task3-2026 deployments",
    ]
    if project:
        cmd += ["--project", project]
    run_command(cmd)


def build_and_push_image(
    *,
    source_root: Path,
    include_ml_assets: bool,
    service: str,
    region: str,
    project: str | None,
) -> str:
    effective_project = get_effective_project(project)
    ensure_artifact_registry_repo(project=project, region=region)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    image_uri = f"{region}-docker.pkg.dev/{effective_project}/{ARTIFACT_REPO}/{service}:{timestamp}"

    run_command(["gcloud", "auth", "configure-docker", f"{region}-docker.pkg.dev", "--quiet"])
    run_command(["docker", "build", "-t", image_uri, str(source_root)])
    run_command(["docker", "push", image_uri])
    return image_uri


def deploy_to_cloud_run(
    *,
    image_uri: str,
    config_path: Path,
    service: str,
    region: str,
    project: str | None,
    secret_name: str,
    dry_run: bool,
) -> None:
    if dry_run:
        print("Dry run: would deploy with the following settings:")
        print(f"  service: {service}")
        print(f"  region: {region}")
        print(f"  project: {project or '(current gcloud project)'}")
        print(f"  config: {config_path}")
        print(f"  image: {image_uri}")
        print(f"  secret: {secret_name}")
        return

    cmd = [
        "gcloud",
        "run",
        "deploy",
        service,
        "--image",
        image_uri,
        "--region",
        region,
        "--allow-unauthenticated",
        "--min-instances",
        "1",
        "--max-instances",
        "1",
        "--concurrency",
        "1",
        "--timeout",
        "300s",
        "--quiet",
        "--set-secrets",
        f"{DEFAULT_SECRET_NAME}={secret_name}:latest",
    ]
    if project:
        cmd += ["--project", project]

    if config_path != (repo_root() / DEFAULT_CONFIG).resolve():
        cmd += ["--set-env-vars", f"DECISION_BRIDGE_CONFIG={config_path.relative_to(repo_root()).as_posix()}"]

    run_command(cmd)


def main() -> None:
    root = repo_root()
    load_dotenv(root / ".env")
    args = parse_args()

    config_path = resolve_repo_path(args.config)
    config = load_config(config_path)
    analysts = workflow_analysts(config)

    print("Deployment summary:")
    print(f"  config: {config_path.relative_to(root)}")
    print(f"  analysts: {', '.join(analysts) if analysts else '(none)'}")
    print(f"  service: {args.service}")
    print(f"  region: {args.region}")
    if args.project:
        print(f"  project: {args.project}")

    ensure_output_dir()

    if needs_ml_artifacts(analysts):
        maybe_train_ml_artifacts(dry_run=args.dry_run)
    else:
        print("No ML analyst in workflow_analysts; no model artifact build is needed.")

    if args.dry_run:
        if args.sync_secret:
            print(f"Dry run: would sync {args.secret_name} from local {DEFAULT_SECRET_NAME}.")
        image_uri = f"{args.region}-docker.pkg.dev/{get_effective_project(args.project)}/{ARTIFACT_REPO}/{args.service}:dry-run"
        deploy_to_cloud_run(
            image_uri=image_uri,
            config_path=config_path,
            service=args.service,
            region=args.region,
            project=args.project,
            secret_name=args.secret_name,
            dry_run=True,
        )
        return

    ensure_gcloud()

    if args.sync_secret:
        sync_secret(args.secret_name, args.project)
    elif not ensure_secret_exists(args.secret_name, args.project):
        raise SystemExit(
            f"Secret {args.secret_name!r} does not exist in Google Cloud. "
            "Run again with --sync-secret after setting OPENAI_API_KEY locally, "
            "or create the secret manually and rerun."
        )
    ensure_secret_accessor_binding(args.secret_name, args.project)

    image_uri = build_and_push_image(
        source_root=root,
        include_ml_assets=needs_ml_artifacts(analysts),
        service=args.service,
        region=args.region,
        project=args.project,
    )
    deploy_to_cloud_run(
        image_uri=image_uri,
        config_path=config_path,
        service=args.service,
        region=args.region,
        project=args.project,
        secret_name=args.secret_name,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
