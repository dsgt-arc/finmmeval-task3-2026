import pandas as pd

from decision_making.sp500_data import (
    DATA_DIR,
    DATA_FILE,
    SP500_SOURCE,
    convert_to_long_format,
    get_sector,
    get_stock_data,
    print_metrics_coverage,
    save_to_partitioned_parquet,
)

df = pd.read_csv(SP500_SOURCE)
all_symbols = df["Symbol"].tolist()

# Download price data
print("Downloading price data...")
data = get_stock_data(all_symbols, start="1960-01-01")

# Get sector
print("\nFetching sector data...")
sectors = get_sector(all_symbols, batch_size=10)

# Final price and sectors
merged = data.merge(sectors, on="Ticker", how="left")

# Convert to long format
data_long = convert_to_long_format(merged)

print("\nSaving data to partitioned Parquet...")
long_path = DATA_DIR / DATA_FILE
save_to_partitioned_parquet(data_long, long_path)
print(f"\n✓ Saved partitioned Parquet: {long_path}\nShape: {data_long.shape}\nColumns: {list(data_long.columns)}")
print(f"SUMMARY\n{'=' * 80}")
print(
    f"Total tickers: {data_long['Ticker'].nunique()}, Date range: {data_long['Date'].min()} to {data_long['Date'].max()}, Total monthly observations: {len(data_long)}"
)
print_metrics_coverage(df=data_long)
