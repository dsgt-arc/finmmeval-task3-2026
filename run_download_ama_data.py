from decision_making.ama_data import download_all_data

if __name__ == "__main__":
    # Example usage: download all data when script is run directly
    print("Downloading all trading data...")
    files = download_all_data()
    print("\nDownloaded files:")
    for symbol, path in files.items():
        print(f"  {symbol}: {path}")
