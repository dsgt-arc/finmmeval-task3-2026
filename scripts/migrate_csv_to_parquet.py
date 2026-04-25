"""Migrate stock_data_long.csv to the year-partitioned Parquet format.

Usage:
    uv run python scripts/migrate_csv_to_parquet.py

The script reads data/data_sp500/stock_data_long.csv and writes it to
data/data_sp500/stock_data_long/year=YYYY/data.parquet, which is the
format expected by decision_making.sp500_data.load_stock_metric_long.
"""

import sys
from pathlib import Path

# Allow imports from the project root
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from decision_making.sp500_data import DATA_DIR, save_to_partitioned_parquet

CSV_PATH = DATA_DIR / "stock_data_long.csv"
PARQUET_DIR = DATA_DIR / "stock_data_long"


def main():
    if not CSV_PATH.exists():
        print(f"CSV not found: {CSV_PATH}")
        sys.exit(1)

    if PARQUET_DIR.exists():
        existing = list(PARQUET_DIR.glob("year=*/data.parquet"))
        if existing:
            print(f"Parquet directory already exists with {len(existing)} partition(s): {PARQUET_DIR}")
            answer = input("Overwrite? [y/N] ").strip().lower()
            if answer != "y":
                print("Aborted.")
                sys.exit(0)

    print(f"Reading {CSV_PATH} ...")
    df = pd.read_csv(CSV_PATH)
    print(f"  {len(df):,} rows, {df['Ticker'].nunique()} tickers, metrics: {df['Metric'].unique().tolist()}")

    print(f"\nWriting partitioned Parquet to {PARQUET_DIR} ...")
    save_to_partitioned_parquet(df, PARQUET_DIR)

    written = sorted(PARQUET_DIR.glob("year=*/data.parquet"))
    total_bytes = sum(p.stat().st_size for p in written)
    csv_bytes = CSV_PATH.stat().st_size
    print(f"  Partitions written : {len(written)}")
    print(f"  Parquet total size : {total_bytes / 1_048_576:.1f} MB")
    print(f"  CSV size           : {csv_bytes / 1_048_576:.1f} MB")
    print(f"  Size reduction     : {(1 - total_bytes / csv_bytes) * 100:.0f}%")
    print("\nDone.")


if __name__ == "__main__":
    main()
