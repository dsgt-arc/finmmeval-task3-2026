import datetime
from pathlib import Path

import polars as pl

# HuggingFace dataset configuration
HF_DATASET_AMA = "TheFinAI/daily_news"  # data of the Agents Market Arena (AMA) with more assets
SPLITS_AMA = {
    # coins
    "BTC": "data_new/BTC-00000-of-00001.parquet",
    "ETH": "data_new/ETH-00000-of-00001.parquet",
    # stocks
    "TSLA": "data_new/TSLA-00000-of-00001.parquet",
    "BMRN": "data_new/BMRN-00000-of-00001.parquet",
    "MRNA": "data_new/MRNA-00000-of-00001.parquet",
    "MSFT": "data_new/MSFT-00000-of-00001.parquet",
}

HF_DATASET_COMPETITION = "TheFinAI/CLEF_Task3_Trading"  # includes TSLA and BTC
SPLITS_COMPETITION = {
    # coins
    "BTC": "data/BTC-00000-of-00001.parquet",
    # stocks
    "TSLA": "data/TSLA-00000-of-00001.parquet",
}


# Local data directory (at project root)
DATA_DIR = Path(__file__).parent.parent / "data"


# Symbols for panel
SYMBOLS = ["TSLA", "BMRN", "MRNA", "MSFT"]

# Polars column expressions
price_col = pl.col("prices")
return_col = pl.col("returns")
target_multi_col = pl.col("target_mulit")
target_bin_col = pl.col("target_binary")
date_col = pl.col("date")
text_col = pl.col("news")
text_len_col = pl.col("news_length")
text_str_col = pl.col("news_str")


def download_data(symbol: str, force_download: bool = False) -> Path:
    """
    Download data for a specific symbol from HuggingFace.

    Args:
        symbol: The trading symbol (e.g., 'BTC', 'TSLA')
        force_download: If True, re-download even if file exists

    Returns:
        Path to the downloaded parquet file
    """
    if symbol not in SPLITS_COMPETITION:
        raise ValueError(f"Unknown symbol: {symbol}. Available: {list(SPLITS_COMPETITION.keys())}")

    from huggingface_hub import hf_hub_download

    # Create data directory if it doesn't exist
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Download from HuggingFace
    local_file = hf_hub_download(
        repo_id=HF_DATASET_COMPETITION,
        filename=SPLITS_COMPETITION[symbol],
        repo_type="dataset",
        local_dir=DATA_DIR,
        force_download=force_download,
    )

    return Path(local_file)


def download_all_data(force_download: bool = False) -> dict[str, Path]:
    """
    Download all available trading data.

    Args:
        force_download: If True, re-download even if files exist

    Returns:
        Dictionary mapping symbols to their local file paths
    """
    downloaded = {}
    for symbol in SPLITS_COMPETITION:
        print(f"Downloading {symbol}...")
        downloaded[symbol] = download_data(symbol, force_download=force_download)
    return downloaded


def load_data(symbol: str, download_if_missing: bool = True, competition_data: bool = True) -> pl.DataFrame:
    """
    Load data for a specific symbol.

    Args:
        symbol: The trading symbol (e.g., 'BTC', 'TSLA')
        download_if_missing: If True, download data if not found locally
        competition_data: If True, load from old data directory (has more history)
    Returns:
        Polars DataFrame with the trading data
    """
    if symbol not in SPLITS_AMA:
        raise ValueError(f"Unknown symbol: {symbol}. Available: {list(SPLITS_AMA.keys())}")
    if competition_data:
        local_path = DATA_DIR / "data" / Path(SPLITS_COMPETITION[symbol]).name
    else:
        local_path = DATA_DIR / "data_ana" / Path(SPLITS_AMA[symbol]).name

    # Download if file doesn't exist and download_if_missing is True
    if not local_path.exists() and download_if_missing:
        print(f"Data not found locally. Downloading {symbol}...")
        local_path = download_data(symbol)
    elif not local_path.exists():
        raise FileNotFoundError(f"Data file not found: {local_path}")

    return pl.read_parquet(local_path)


def _payload_price_frame(payload: dict | None, symbol: str, date: datetime.date | None = None) -> pl.DataFrame:
    """Build a price frame from the competition payload if available.

    The payload may contain a sparse historical series in `history_price` plus a
    current-day point in `price`. We merge those into a single time series and
    let the caller decide whether to combine them with local parquet history.
    """
    if not payload:
        return pl.DataFrame(schema={"date": pl.Date, "prices": pl.Float64})

    rows: list[dict[str, object]] = []
    history_map = payload.get("history_price") or {}
    for row in history_map.get(symbol) or []:
        if not row:
            continue
        row_date = row.get("date")
        row_price = row.get("price")
        if row_date is None or row_price is None:
            continue
        rows.append({"date": row_date, "prices": row_price})

    payload_date = payload.get("date")
    price_map = payload.get("price") or {}
    if payload_date is not None and symbol in price_map:
        rows.append({"date": payload_date, "prices": price_map[symbol]})

    if not rows:
        return pl.DataFrame(schema={"date": pl.Date, "prices": pl.Float64})

    df = pl.DataFrame(rows).with_columns(
        pl.col("date").cast(pl.Date),
        pl.col("prices").cast(pl.Float64),
    )
    if date is not None:
        df = df.filter(pl.col("date") <= date)
    return df.unique(subset=["date"], keep="last").sort("date")


def load_specific_data(
    symbol: str,
    date: str,
    type: str,
    download_if_missing: bool = True,
    competition_data: bool = True,
    api_payload: dict | None = None,
) -> pl.DataFrame:
    """Load data for a specific symbol, date, and type.
    Args:
    symbol: The trading symbol (e.g., 'BTC', 'TSLA')
    date: The date to filter by (format 'YYYY-MM-DD')
    type: The type to filter by (e.g., 'news', 'social_media')
    download_if_missing: If True, download data if not found locally
    competition_data: If True, load from old data directory (has more history)
    Returns:
    str | pl.DataFrame: Either a string (for news) or a DataFrame (for price)
    """

    symbol_data = load_data(symbol, download_if_missing=download_if_missing, competition_data=competition_data)
    # cast date column to datetime/date for reliable comparisons
    symbol_data = symbol_data.with_columns(date_col.cast(datetime.date))

    if isinstance(date, str):
        date = datetime.date.fromisoformat(date)
    if type == "news":
        # news is focusing just on the news of day t-1
        date_data = symbol_data.filter(date_col == date)
        specific_date_data = date_data.select(text_col).item().item()
    elif type == "price":
        # price needs to include all data up to day t-1 for technical analysis
        date_data = symbol_data.filter(date_col <= date)
        specific_date_data = date_data.select(date_col, price_col)

        payload_prices = _payload_price_frame(api_payload, symbol, date)
        if not payload_prices.is_empty():
            specific_date_data = (
                pl.concat([specific_date_data, payload_prices], how="vertical")
                .unique(subset=["date"], keep="last")
                .sort("date")
            )
    elif type == "current_price":
        # current price is the price of the specific date.
        # If the exact date is missing, fall back to the latest available
        # trading day at or before the requested date so weekends / lagged
        # competition data do not crash the workflow.
        payload_prices = _payload_price_frame(api_payload, symbol, date)
        if not payload_prices.is_empty():
            date_data = payload_prices.filter(date_col <= date)
        else:
            date_data = symbol_data.filter(date_col == date)
        if date_data.is_empty():
            date_data = symbol_data.filter(date_col < date)
            if date_data.is_empty():
                raise ValueError(f"No price data available on or before {date} for {symbol}")
            date_data = date_data.sort(date_col).tail(1)
        specific_date_data = date_data.select(price_col).tail(1).item()
    else:
        raise ValueError(f"Unknown type: {type}. Available types: 'news', 'price'")
    return specific_date_data
