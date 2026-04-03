# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a CLEF (Conference and Labs of the Evaluation Forum) project template repository for data science tasks. The project has been customized from the template with the main package renamed to `decision_making` (instead of `my_task_package`).

The dependency source of truth is `pyproject.toml`, and local development uses `uv`.

## Common Commands

### Setup

```bash
uv sync

# Install pre-commit hooks
uv run pre-commit install
```

### Data Management

```bash
# Download all trading data from HuggingFace
uv run python -m decision_making.data

# Use in Python scripts
uv run python
>>> from decision_making.data import load_data
>>> df = load_data('BTC')  # Auto-downloads if missing
```

### Testing

```bash
# Run all tests with verbose output
uv run pytest -v tests/

# Run a specific test file
uv run pytest -v tests/test_spark.py

# Run a specific test function
uv run pytest -v tests/test_spark.py::test_get_spark
```

### Code Quality

```bash
# Run pre-commit hooks manually
uv run pre-commit run --all-files

# Format code with ruff
uv run ruff format .

# Lint and auto-fix with ruff
uv run ruff check --fix .
```

### Notebook Management

```bash
# IMPORTANT: Always clear notebook outputs before committing
jupyter nbconvert --clear-output --inplace notebooks/**/*.ipynb

# Clear outputs for a specific notebook
jupyter nbconvert --clear-output --inplace notebooks/eda/my-notebook.ipynb
```

**Critical Rule:** Notebook outputs must ALWAYS be cleared before committing to keep the repository clean and avoid bloating git history with large output data.

## Architecture

### Package Structure

- `decision_making/` - Main task package containing core functionality
- `api/` - FastAPI competition endpoint plus the bridge into the workflow
- `tests/` - Test suite with pytest fixtures
- `docs/` - Project documentation
- `notebooks/` - Jupyter notebooks for data exploration and analysis
  - `eda/` - Exploratory Data Analysis notebooks
- `user/` - Scratch space for individual users to commit work without polluting main repo
- `scripts/` - Utility scripts (e.g., log sweeping)

### PySpark Configuration

The project uses PySpark for distributed data processing:

- `get_spark()` in `spark.py` creates a local Spark session with configurable parameters
- Default configuration: all CPU cores, 16g driver memory, 1g executor memory
- Environment variables: `PYSPARK_DRIVER_MEMORY`, `PYSPARK_EXECUTOR_MEMORY`, `SPARK_LOCAL_DIR`
- `spark_resource()` context manager ensures proper cleanup of Spark sessions
- Test fixtures provide a session-scoped Spark instance for testing

### Pre-commit Hooks

The project enforces code quality through pre-commit hooks:

- YAML validation, EOF fixing, trailing whitespace removal
- Ruff formatting and linting with auto-fix
- Prettier for consistent formatting
- Codespell for spell checking (excludes package.json, .ipynb, CHANGELOG.md)

### Dependencies

Core dependencies include:

- PySpark (>= 3.4.0) with PyArrow for efficient data interchange
- Scientific stack: numpy, pandas, matplotlib, scikit-learn
- Workflow orchestration: luigi
- Development tools: jupyterlab, ruff, pytest, pre-commit
