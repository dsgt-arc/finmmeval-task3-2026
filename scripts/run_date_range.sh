#!/bin/bash

# Run decision_making for a date range
#
# Usage: ./run_date_range.sh [config_path] [start_date end_date]
#
# Arguments (all optional):
#   config_path  - Path to YAML config (default: decision_making/config/dev.yaml)
#   start_date   - Start date YYYY-MM-DD (default: min_date + 1 from data)
#   end_date     - End date YYYY-MM-DD (default: max_date from data)
#
# Defaults:
#   - Config: Uses decision_making/config/dev.yaml
#   - Dates: Queries the first ticker from config to get available date range
#   - Start: min_date + 1 day (needs previous day for t-1 analysis)
#   - End: max_date from data
#
# Examples:
#   ./run_date_range.sh
#       → Uses dev.yaml, runs from (data min + 1 day) to (data max)
#
#   ./run_date_range.sh decision_making/config/dev.yaml
#       → Uses dev.yaml, runs from (data min + 1 day) to (data max)
#
#   ./run_date_range.sh decision_making/config/dev.yaml 2024-01-15 2024-01-20
#       → Uses dev.yaml, runs from 2024-01-15 to 2024-01-20

set -e  # Exit on error

# Raise file descriptor limit — online learning opens many files (yfinance SQLite
# cache + parquet files) across ~500 SP500 tickers. macOS non-interactive shells
# default to 256; 65536 matches typical interactive terminal sessions.
ulimit -n 65536 2>/dev/null || ulimit -n 4096 2>/dev/null || true

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Change to project root
cd "$PROJECT_ROOT"

# Set PYTHONPATH
export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/decision_making"

# Default config path
CONFIG_PATH=${1:-"decision_making/config/dev_cne.yaml"}

# If dates not provided, query from data
if [ "$#" -eq 0 ] || [ "$#" -eq 1 ]; then
    echo "Querying available date range from data..."

    # Get min and max dates from data
    DATES=$(python << EOF
import sys
import yaml
from datetime import datetime, timedelta
from decision_making.ama_data import load_data

# Load config to get first ticker
with open("$CONFIG_PATH") as f:
    config = yaml.safe_load(f)

ticker = config["tickers"][0]
print(f"Loading data for {ticker} to determine date range...", file=sys.stderr, flush=True)

# Load data and get date range
df = load_data(ticker, download_if_missing=True, competition_data=True)
min_date = df["date"].min()
max_date = df["date"].max()

# Convert to datetime if needed (polars returns date objects)
if isinstance(min_date, str):
    min_date = datetime.strptime(min_date, '%Y-%m-%d').date()
if isinstance(max_date, str):
    max_date = datetime.strptime(max_date, '%Y-%m-%d').date()

# Start from min_date + 1 (need previous day's data)
start_date = min_date + timedelta(days=1)

print(f"{start_date.strftime('%Y-%m-%d')} {max_date.strftime('%Y-%m-%d')}")
EOF
)

    START_DATE=$(echo $DATES | cut -d' ' -f1)
    END_DATE=$(echo $DATES | cut -d' ' -f2)

    echo "Detected date range: $START_DATE to $END_DATE"
elif [ "$#" -eq 2 ]; then
    echo "Error: Must provide both start_date and end_date, or neither"
    echo "Usage: $0 [config_path] [start_date end_date]"
    exit 1
else
    START_DATE=$2
    END_DATE=$3
fi

echo ""
echo "Running decision making from $START_DATE to $END_DATE"
echo "Config: $CONFIG_PATH"

# Create logs directory if it doesn't exist
LOG_DIR="decision_making/logs"
mkdir -p "$LOG_DIR"

# Create log file with timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/run_${TIMESTAMP}.log"
echo "Logging to: $LOG_FILE"
echo ""

# Redirect all output to log file and terminal
exec > >(tee -a "$LOG_FILE") 2>&1

echo "=== Run started at $(date) ==="
echo "Config: $CONFIG_PATH"
echo "Date range: $START_DATE to $END_DATE"
echo ""

# Use Python to generate and iterate dates (cross-platform)
python << EOF
from datetime import datetime, timedelta
import os
import subprocess
import sys

import pandas_market_calendars as mcal

start = datetime.strptime("$START_DATE", "%Y-%m-%d")
end = datetime.strptime("$END_DATE", "%Y-%m-%d")
repo_root = os.path.abspath(".")
env = os.environ.copy()
env["PYTHONPATH"] = f"{repo_root}/decision_making:{repo_root}"

if start > end:
    print("Error: start_date must be <= end_date")
    sys.exit(1)

nyse = mcal.get_calendar("NYSE")
schedule = nyse.schedule(start_date=start.strftime("%Y-%m-%d"), end_date=end.strftime("%Y-%m-%d"))
trading_days = set(schedule.index.date)

current = start
total_days = len(trading_days)
day_num = 0

while current <= end:
    if current.date() not in trading_days:
        current += timedelta(days=1)
        continue

    day_num += 1
    trading_date = current.strftime("%Y-%m-%d")
    print(f"=== Processing date {day_num}/{total_days}: {trading_date} ===")

    result = subprocess.run([
        sys.executable,
        "decision_making/run_decision_making.py",
        "--config", "$CONFIG_PATH",
        "--trading-date", trading_date,
        "--local-db"
    ], cwd=repo_root, env=env)

    if result.returncode != 0:
        print(f"Error processing {trading_date}")
        sys.exit(1)

    print()
    current += timedelta(days=1)

print("All dates processed successfully!")
print(f"=== Run completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===")
EOF

echo ""
echo "Log saved to: $LOG_FILE"
