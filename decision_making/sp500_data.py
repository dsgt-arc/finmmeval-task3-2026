from pathlib import Path
import uuid

import pandas as pd
import yfinance as yf

SP500_SOURCE = "https://yfiua.github.io/index-constituents/constituents-sp500.csv"

DATA_DIR = Path(__file__).parent.parent / "data" / "data_sp500"
DATA_FILE = Path("stock_data_long_test")  # partitioned Parquet directory (year=YYYY/data.parquet)


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


def fetch_sp500_adjclose_since(
    tickers: list[str],
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """Fetch adj close from yfinance for given tickers and date range.

    Args:
        tickers: List of ticker symbols
        start_date: Start date string 'YYYY-MM-DD' (inclusive)
        end_date: End date string 'YYYY-MM-DD' (inclusive)

    Returns:
        Wide-format DataFrame (datetime index x ticker columns) or empty DataFrame
    """
    raw = get_stock_data(tickers, start=start_date, end=end_date)
    if raw.empty:
        return pd.DataFrame()
    wide = raw.pivot_table(index="Date", columns="Ticker", values="Adj_Close")
    wide.index = pd.to_datetime(wide.index)
    wide.index.name = "date"
    return wide


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


def load_single_stocks(path: Path = DATA_DIR / DATA_FILE, min_obs: int = 400) -> pd.DataFrame:
    """
    Load stock returns computed from adjusted close prices.
    """
    # Try loading from new long-format data
    adj_close = load_stock_metric_long("Adj_Close", path, min_obs=min_obs).astype(float)  # Don't filter yet

    # Compute returns for each stock
    returns = adj_close.pct_change()

    return returns


def load_stock_metric_long(metric_name: str, path: Path = DATA_DIR / DATA_FILE, min_obs: int = 400) -> pd.DataFrame:
    """
    Load a specific metric from long-format stock data.

    Args:
        metric_name: Name of metric to load (e.g., 'Vol_21d', 'Market_Cap', 'Book_to_Market')
        path: Path to long-format CSV file
        min_obs: Minimum number of observations required per stock

    Returns:
        Wide-format DataFrame with date index and ticker columns
    """
    df = pd.read_parquet(path, columns=["Date", "Ticker", "Metric", "Value"])
    df = df.drop_duplicates(subset=["Date", "Ticker", "Metric"], keep="last")

    # Filter for requested metric
    metric_df = df[df["Metric"] == metric_name].copy()

    if metric_df.empty:
        raise ValueError(f"Metric '{metric_name}' not found in data. Available metrics: {df['Metric'].unique().tolist()}")

    # Pivot to wide format
    df_wide = metric_df.pivot(index="Date", columns="Ticker", values="Value")
    df_wide.index.name = "date"
    df_wide.index = pd.to_datetime(df_wide.index)

    # Filter stocks with sufficient history
    long_history_stocks = df_wide.describe().loc["count"][df_wide.describe().loc["count"] > min_obs].index

    df_wide = df_wide[long_history_stocks].dropna(axis="index", how="all")
    return df_wide


def save_to_partitioned_parquet(df: pd.DataFrame, path: Path = DATA_DIR / DATA_FILE) -> None:
    """Write long-format DataFrame to year-partitioned Parquet directory.

    Partitions by a 'year' column derived from 'Date'. Each partition is
    written to ``path/year=YYYY/data.parquet``, overwriting any existing file
    for that year.

    Args:
        df: Long-format DataFrame with columns [Ticker, Date, Metric, Value]
        path: Root directory for the partitioned dataset
    """
    path.mkdir(parents=True, exist_ok=True)
    df = df.copy()
    df["year"] = pd.to_datetime(df["Date"]).dt.year
    for year, year_df in df.groupby("year"):
        year_dir = path / f"year={year}"
        year_dir.mkdir(exist_ok=True)
        year_df.drop(columns=["year"]).reset_index(drop=True).to_parquet(year_dir / "data.parquet", index=False)


def append_adjclose_to_store(wide_df: pd.DataFrame, path: Path = DATA_DIR / DATA_FILE) -> None:
    """Append new adj close data (wide format) into the year-partitioned Parquet store.

    Converts wide_df (datetime index x ticker columns) to long format with
    Metric='Adj_Close', then for each affected year reads only that year's
    partition, merges, deduplicates, and writes back — without touching any
    other year.

    Args:
        wide_df: Wide-format DataFrame (datetime index x ticker columns)
        path: Root directory of the partitioned dataset
    """
    new_long = (
        wide_df.reset_index()
        .rename(columns={"date": "Date"})
        .melt(id_vars=["Date"], var_name="Ticker", value_name="Value")
        .assign(Metric="Adj_Close")
        .dropna(subset=["Value"])
    )
    new_long["Date"] = pd.to_datetime(new_long["Date"]).dt.strftime("%Y-%m-%d")
    new_long["year"] = pd.to_datetime(new_long["Date"]).dt.year

    for year, year_df in new_long.groupby("year"):
        year_dir = path / f"year={year}"
        year_dir.mkdir(parents=True, exist_ok=True)
        year_df.drop(columns=["year"]).reset_index(drop=True).to_parquet(year_dir / f"{uuid.uuid4()}.parquet", index=False)


def load_stock_volatility(path: Path = DATA_DIR / DATA_FILE, min_obs: int = 400) -> pd.DataFrame:
    """
    Load 21-day rolling volatility for individual stocks.

    Returns wide-format DataFrame with annualized volatility values.
    """
    return load_stock_metric_long("Vol_21d", path, min_obs)


def load_stock_market_cap(path: Path = DATA_DIR / DATA_FILE, min_obs: int = 400) -> pd.DataFrame:
    """
    Load market capitalization for individual stocks.

    Returns wide-format DataFrame with market cap values.
    """
    return load_stock_metric_long("Market_Cap", path, min_obs)


def load_stock_book_to_market(path: Path = DATA_DIR / DATA_FILE, min_obs: int = 400) -> pd.DataFrame:
    """
    Load book-to-market ratio for individual stocks.

    Book-to-market = Total Stockholder Equity / Market Cap
    Note: Equity data is lagged by 4 months to account for reporting delays.

    Returns wide-format DataFrame with book-to-market values.
    """
    return load_stock_metric_long("Book_to_Market", path, min_obs)


def load_stock_total_equity(path: Path = DATA_DIR / DATA_FILE, min_obs: int = 400) -> pd.DataFrame:
    """
    Load total stockholder equity for individual stocks.

    Note: Equity data is lagged by 4 months to account for reporting delays.

    Returns wide-format DataFrame with total equity values.
    """
    return load_stock_metric_long("Total_Equity", path, min_obs)


def load_stock_sector(path: Path = DATA_DIR / DATA_FILE, min_obs: int = 400) -> pd.DataFrame:
    """
    Load sector for individual stocks.
    """
    sector = load_stock_metric_long("Sector", path, min_obs)
    return sector
