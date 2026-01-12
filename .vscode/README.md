# VS Code Configuration

This directory contains shared VS Code settings for the project to ensure consistent code formatting and linting across all team members.

## Setup

### 1. Install Required VS Code Extensions

When you open this project in VS Code, you should see a notification prompting you to install recommended extensions. Click "Install All" or install them manually:

**Required:**
- **Ruff** (`charliermarsh.ruff`) - Fast Python linter and formatter
- **Python** (`ms-python.python`) - Python language support
- **Jupyter** (`ms-toolsai.jupyter`) - Jupyter notebook support

**Optional but Recommended:**
- Pylance (`ms-python.vscode-pylance`) - Python language server
- Python Environment Manager (`donjayamanne.python-environment-manager`)

### 2. Verify Ruff Installation

After installing extensions, verify Ruff is working:
1. Open any Python file
2. Check the bottom-right corner of VS Code - you should see "Ruff" as the formatter
3. Save a file - it should auto-format

### 3. Manual Commands (Terminal)

You can also run Ruff from the command line:

```bash
# Activate the conda environment
conda activate finmmeval

# Check for linting issues
ruff check .

# Fix auto-fixable issues
ruff check --fix .

# Format code (replaces Black)
ruff format .

# Check formatting without changing files
ruff format --check .
```

## What's Configured

### Formatting (Black-compatible)
- **Line length**: 88 characters
- **Quote style**: Double quotes
- **Indentation**: 4 spaces
- **Auto-format on save**: Enabled for `.py` and `.ipynb` files

### Linting (Flake8-compatible + more)
Enabled rule sets:
- `E`, `W` - pycodestyle errors and warnings
- `F` - Pyflakes
- `I` - isort (import sorting)
- `B` - flake8-bugbear
- `C4` - flake8-comprehensions
- `UP` - pyupgrade
- `ARG` - flake8-unused-arguments
- `SIM` - flake8-simplify
- `PTH` - flake8-use-pathlib
- `RUF` - Ruff-specific rules

### Code Actions on Save
- Organize imports (isort)
- Fix all auto-fixable issues
- Format code
- Remove trailing whitespace
- Ensure final newline

## Jupyter Notebooks

Notebooks are fully supported with:
- Auto-formatting on save
- Import organization
- Linting with relaxed rules (ignores `E402`, `I001` for notebooks)

## Pre-commit Hooks

The project also uses pre-commit hooks to enforce these rules before commits. To install:

```bash
pip install pre-commit
pre-commit install
```

Now Ruff will automatically run on changed files before each commit.

## Troubleshooting

### Format on Save Not Working

1. Check that the Ruff extension is installed and enabled
2. Verify `settings.json` has `"editor.formatOnSave": true`
3. Check the Output panel (View → Output → Ruff) for errors
4. Restart VS Code

### Conflicts with Other Formatters

The configuration disables other Python formatters (Black, autopep8, etc.) to avoid conflicts. If you have workspace-specific settings that override this, remove them.

### Different Team Member Seeing Different Results

Ensure all team members:
1. Have the same Ruff extension version
2. Are using the shared `.vscode/settings.json`
3. Have pulled the latest `pyproject.toml`
4. Have restarted VS Code after changes

## Configuration Files

- `.vscode/settings.json` - VS Code editor settings
- `.vscode/extensions.json` - Recommended extensions
- `pyproject.toml` - Ruff configuration (under `[tool.ruff]`)
- `.pre-commit-config.yaml` - Pre-commit hook configuration
