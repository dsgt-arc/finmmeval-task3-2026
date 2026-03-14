from pathlib import Path

import pandas as pd
import yfinance as yf

SP500_SOURCE = "https://yfiua.github.io/index-constituents/constituents-sp500.csv"

DATA_DIR = Path(__file__).parent.parent / "data" / "data_sp500"


def get_stock_data(symbols, start="2000-01-01", end=None, batch_size=100):
    """
    Returns a DataFrame with daily stock data including prices, market cap, and fundamentals.
    """
    out = []
    symbols = list(dict.fromkeys(symbols))  # de-dup, keep order

    for i in range(0, len(symbols), batch_size):
        batch = symbols[i : i + batch_size]

        print(f"Downloading batch {i // batch_size + 1}/{(len(symbols) + batch_size - 1) // batch_size}...")

        # Download price data
        df = yf.download(
            tickers=batch,
            start=start,
            end=end,
            interval="1d",
            auto_adjust=False,  # We'll use Adj Close column
            group_by="ticker",
            threads=True,
            progress=False,
        )

        # Process each ticker in the batch
        batch_data = []
        for ticker in batch:
            try:
                if len(batch) == 1:
                    ticker_df = df.copy()
                else:
                    ticker_df = df[ticker].copy()

                if ticker_df.empty:
                    continue

                # Keep only Adj Close for price calculations
                ticker_df = pd.DataFrame({
                    "Date": ticker_df.index,
                    "Ticker": ticker,
                    "Adj_Close": ticker_df["Adj Close"],
                    "Close": ticker_df["Close"],
                    "Volume": ticker_df["Volume"],
                })

                batch_data.append(ticker_df)
            except Exception as e:
                print(f"  Error processing {ticker}: {e}")
                continue

        if batch_data:
            batch_df = pd.concat(batch_data, ignore_index=True)
            out.append(batch_df)

    # Combine all batches
    if not out:
        return pd.DataFrame()

    data = pd.concat(out, ignore_index=True)
    data = data.sort_values(["Ticker", "Date"])
    data = data.reset_index(drop=True)

    print(f"✓ Downloaded data for {data['Ticker'].nunique()} tickers")
    return data


def get_sector(symbols, batch_size=10):
    """
    Get market cap and total stockholder equity for each ticker.
    Returns DataFrame with Ticker, Date, Market_Cap, Total_Equity.
    """
    sector_data = []

    for i in range(0, len(symbols), batch_size):
        batch = symbols[i : i + batch_size]
        print(f"Fetching sector batch {i // batch_size + 1}/{(len(symbols) + batch_size - 1) // batch_size}...")

        for ticker in batch:
            try:
                stock = yf.Ticker(ticker)

                info = stock.info

                if info.get("sector"):
                    sector = info["sector"]
                    sector_data.append({"Ticker": ticker, "Sector": sector})

            except Exception as e:
                print(f"  Error fetching sector for {ticker}: {e}")
                continue

    if not sector_data:
        return pd.DataFrame(columns=["Ticker", "Sector"])
    df = pd.DataFrame(sector_data)
    print(f"✓ Fetched sector for {df['Ticker'].nunique()} tickers")
    return df


def convert_to_long_format(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert wide format to long format.

    Wide format: Ticker, Date, Adj_Close, Close, Volume, Vol_21d, Market_Cap, Total_Equity, Book_to_Market
    Long format: Ticker, Date, Metric, Value
    """
    print("\nConverting to long format...")

    # Select metric columns to melt
    id_cols = ["Ticker", "Date"]
    value_cols = ["Adj_Close", "Close", "Volume", "Sector"]

    # Keep only columns that exist
    value_cols = [col for col in value_cols if col in df.columns]

    # Melt to long format
    long_df = df[id_cols + value_cols].melt(id_vars=id_cols, value_vars=value_cols, var_name="Metric", value_name="Value")

    # Sort for better organization
    long_df = long_df.sort_values(["Ticker", "Date", "Metric"])
    long_df = long_df.reset_index(drop=True)

    print(f"✓ Converted to long format: {len(long_df)} rows")
    print(f"  Metrics: {long_df['Metric'].unique().tolist()}")

    return long_df


def print_metrics_coverage(df: pd.DataFrame) -> None:
    print("\nMetrics available:")
    for col in df.columns:
        pct_avail = (df[col].notna().sum() / len(df)) * 100
        print(f"  - {col}: {pct_avail:.1f}% available")
