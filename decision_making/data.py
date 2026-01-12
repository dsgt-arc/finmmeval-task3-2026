from pathlib import Path

from huggingface_hub import hf_hub_download
import polars as pl

# HuggingFace dataset configuration
HF_DATASET = "TheFinAI/CLEF_Task3_Trading"
SPLITS = {
    "BTC": "data/BTC-00000-of-00001.parquet",
    "TSLA": "data/TSLA-00000-of-00001.parquet",
}

# Local data directory (at project root)
DATA_DIR = Path(__file__).parent.parent / "data"


def download_data(symbol: str, force_download: bool = False) -> Path:
    """
    Download data for a specific symbol from HuggingFace.

    Args:
        symbol: The trading symbol (e.g., 'BTC', 'TSLA')
        force_download: If True, re-download even if file exists

    Returns:
        Path to the downloaded parquet file
    """
    if symbol not in SPLITS:
        raise ValueError(f"Unknown symbol: {symbol}. Available: {list(SPLITS.keys())}")

    # Create data directory if it doesn't exist
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Download from HuggingFace
    local_file = hf_hub_download(
        repo_id=HF_DATASET,
        filename=SPLITS[symbol],
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
    for symbol in SPLITS:
        print(f"Downloading {symbol}...")
        downloaded[symbol] = download_data(symbol, force_download=force_download)
    return downloaded


def load_data(symbol: str, download_if_missing: bool = True) -> pl.DataFrame:
    """
    Load data for a specific symbol.

    Args:
        symbol: The trading symbol (e.g., 'BTC', 'TSLA')
        download_if_missing: If True, download data if not found locally

    Returns:
        Polars DataFrame with the trading data
    """
    if symbol not in SPLITS:
        raise ValueError(f"Unknown symbol: {symbol}. Available: {list(SPLITS.keys())}")

    local_path = DATA_DIR / SPLITS[symbol]

    # Download if file doesn't exist and download_if_missing is True
    if not local_path.exists() and download_if_missing:
        print(f"Data not found locally. Downloading {symbol}...")
        local_path = download_data(symbol)
    elif not local_path.exists():
        raise FileNotFoundError(f"Data file not found: {local_path}")

    return pl.read_parquet(local_path)
