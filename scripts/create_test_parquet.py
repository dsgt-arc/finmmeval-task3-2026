"""Create a test Parquet dataset truncated to a given cutoff date.

Reads the production partitioned Parquet at data/data_sp500/stock_data_long/,
filters rows to Date <= CUTOFF_DATE, and writes a new partitioned dataset to
data/data_sp500/stock_data_long_test/.

Usage:
    uv run python scripts/create_test_parquet.py
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from decision_making.sp500_data import DATA_DIR, save_to_partitioned_parquet

CUTOFF_DATE = "2024-07-01"
SOURCE_DIR = DATA_DIR / "stock_data_long"
OUTPUT_DIR = DATA_DIR / "stock_data_long_test"


def main():
    if not SOURCE_DIR.exists():
        print(f"Source Parquet directory not found: {SOURCE_DIR}")
        sys.exit(1)

    print(f"Reading partitioned Parquet from {SOURCE_DIR} ...")
    df = pd.read_parquet(SOURCE_DIR, columns=["Ticker", "Date", "Metric", "Value"])
    print(f"  Full dataset: {len(df):,} rows")

    df_filtered = df[df["Date"] <= CUTOFF_DATE].copy()
    print(f"  After cutoff ({CUTOFF_DATE}): {len(df_filtered):,} rows")

    if OUTPUT_DIR.exists():
        import shutil
        shutil.rmtree(OUTPUT_DIR)
        print(f"  Removed existing {OUTPUT_DIR}")

    print(f"\nWriting test dataset to {OUTPUT_DIR} ...")
    save_to_partitioned_parquet(df_filtered, OUTPUT_DIR)

    written = sorted(OUTPUT_DIR.glob("year=*/data.parquet"))
    print(f"  Partitions written: {len(written)} ({[p.parent.name for p in written]})")
    print("\nDone. To use in tests, pass the path explicitly:")
    print(f'  load_stock_metric_long("Adj_Close", path=DATA_DIR / "stock_data_long_test")')


if __name__ == "__main__":
    main()
