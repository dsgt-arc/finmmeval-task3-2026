#!/bin/bash

# Script to delete all history for a given config from the database
# Usage: ./delete_config_history.sh <exp_name_or_config_id>

set -e

# Check if argument is provided
if [ $# -eq 0 ]; then
    echo "Error: No config identifier provided"
    echo "Usage: $0 <exp_name_or_config_id>"
    echo ""
    echo "Examples:"
    echo "  $0 my_experiment_name"
    echo "  $0 abc123-config-id-456"
    exit 1
fi

CONFIG_IDENTIFIER="$1"

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Load DB_PATH from .env file
if [ -f "$PROJECT_ROOT/.env" ]; then
    # Source the .env file to load DB_PATH
    DB_PATH=$(grep "^DB_PATH=" "$PROJECT_ROOT/.env" | cut -d '=' -f2- | tr -d '"' | tr -d "'")

    if [ -z "$DB_PATH" ]; then
        echo "Error: DB_PATH not found in .env file"
        exit 1
    fi
else
    echo "Error: .env file not found at $PROJECT_ROOT/.env"
    exit 1
fi

echo "DB_PATH from .env: $DB_PATH"

# Make DB_PATH absolute if it's relative
if [[ "$DB_PATH" != /* ]]; then
    # Try resolving from project root first
    RESOLVED_PATH=$(python -c "import os; print(os.path.abspath(os.path.join('$PROJECT_ROOT', '$DB_PATH')))")

    # If not found, try resolving as if Python would (from current working dir perspective)
    if [ ! -f "$RESOLVED_PATH" ]; then
        # Remove ../ prefix and try from project root
        DB_PATH_NO_PARENT=$(echo "$DB_PATH" | sed 's|^\.\./||g')
        RESOLVED_PATH="$PROJECT_ROOT/$DB_PATH_NO_PARENT"
    fi

    DB_PATH="$RESOLVED_PATH"
fi

echo "Resolved database path: $DB_PATH"

# Check if database exists
if [ ! -f "$DB_PATH" ]; then
    echo "Error: Database file not found at $DB_PATH"
    echo ""
    echo "Searched locations:"
    echo "  - $DB_PATH"
    if [[ "$DB_PATH" =~ \.\. ]]; then
        echo "  - $PROJECT_ROOT/$(echo "$DB_PATH" | sed 's|^\.\./||g')"
    fi
    exit 1
fi

# Try to find config by exp_name first, then by id
CONFIG_ID=$(sqlite3 "$DB_PATH" "SELECT id FROM config WHERE exp_name = '$CONFIG_IDENTIFIER' OR id = '$CONFIG_IDENTIFIER' LIMIT 1;")

if [ -z "$CONFIG_ID" ]; then
    echo "Error: No config found with exp_name or id: $CONFIG_IDENTIFIER"
    echo ""
    echo "Available configs:"
    sqlite3 -header -column "$DB_PATH" "SELECT id, exp_name, llm_model FROM config;"
    exit 1
fi

# Get config details
CONFIG_INFO=$(sqlite3 -header -column "$DB_PATH" "SELECT id, exp_name, llm_model, updated_at FROM config WHERE id = '$CONFIG_ID';")

echo ""
echo "Found config:"
echo "$CONFIG_INFO"
echo ""

# Count records to be deleted
SIGNAL_COUNT=$(sqlite3 "$DB_PATH" "SELECT COUNT(*) FROM signal WHERE portfolio_id IN (SELECT id FROM portfolio WHERE config_id = '$CONFIG_ID');")
DECISION_COUNT=$(sqlite3 "$DB_PATH" "SELECT COUNT(*) FROM decision WHERE portfolio_id IN (SELECT id FROM portfolio WHERE config_id = '$CONFIG_ID');")
PORTFOLIO_COUNT=$(sqlite3 "$DB_PATH" "SELECT COUNT(*) FROM portfolio WHERE config_id = '$CONFIG_ID';")

echo "Records to be deleted:"
echo "  - Signals: $SIGNAL_COUNT"
echo "  - Decisions: $DECISION_COUNT"
echo "  - Portfolios: $PORTFOLIO_COUNT"
echo ""

# Confirmation prompt
read -p "Are you sure you want to delete this history? (yes/no): " CONFIRM

if [ "$CONFIRM" != "yes" ]; then
    echo "Deletion cancelled."
    exit 0
fi

# Delete in correct order (due to foreign key constraints)
echo ""
echo "Deleting history..."

sqlite3 "$DB_PATH" <<EOF
DELETE FROM signal WHERE portfolio_id IN (SELECT id FROM portfolio WHERE config_id = '$CONFIG_ID');
DELETE FROM decision WHERE portfolio_id IN (SELECT id FROM portfolio WHERE config_id = '$CONFIG_ID');
DELETE FROM portfolio WHERE config_id = '$CONFIG_ID';
DELETE FROM config WHERE id = '$CONFIG_ID';
EOF

echo "✓ History deleted successfully!"
