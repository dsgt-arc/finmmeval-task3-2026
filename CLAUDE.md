# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a CLEF (Conference and Labs of the Evaluation Forum) project template repository for data science tasks. The project has been customized from the template with the main package renamed to `decision_making` (instead of `my_task_package`).

Note: The `pyproject.toml` still references `my_task_package` in the `include` field, which should be updated to `decision_making*` if not already done.

## Common Commands

### Setup
```bash
# Create and activate conda environment
conda env create -f environment.yml
conda activate finmmeval

# Install package in editable mode
pip install -e .

# Install pre-commit hooks
pre-commit install
```

Alternative setup with venv:
```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install package in editable mode
pip install -e .

# Install pre-commit hooks
pre-commit install
```

### Data Management
```bash
# Download all trading data from HuggingFace
python -m decision_making.data

# Use in Python scripts
python
>>> from decision_making.data import load_data
>>> df = load_data('BTC')  # Auto-downloads if missing
```

### Testing
```bash
# Run all tests with verbose output
pytest -v tests/

# Run a specific test file
pytest -v tests/test_spark.py

# Run a specific test function
pytest -v tests/test_spark.py::test_get_spark
```

### Code Quality
```bash
# Run pre-commit hooks manually
pre-commit run --all-files

# Format code with ruff
ruff format .

# Lint and auto-fix with ruff
ruff check --fix .
```

## Architecture

### Package Structure
- `decision_making/` - Main task package containing core functionality
  - `spark.py` - PySpark session utilities with configurable memory and cores
  - `data.py` - Data loading and processing utilities (currently empty)
- `tests/` - Test suite with pytest fixtures
  - `conftest.py` - Shared fixtures including session-scoped Spark fixture
- `notebooks/` - Jupyter notebooks for data exploration and analysis
  - `eda/` - Exploratory Data Analysis notebooks
- `user/` - Scratch space for individual users to commit work without polluting main repo
- `scripts/` - Utility scripts (e.g., log sweeping)
- `docs/` - Project documentation

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
