# Ensure yfinance and pandas are installed. If not, run:
# pip install yfinance pandas

import yfinance as yf
import pandas as pd
import os
# from datetime import datetime, timedelta # Not strictly needed for this refactored version if only using period

TICKER_SYMBOL = "^NDX"

# Configuration for data fetching, mapping interval string to yfinance parameters
INTERVAL_CONFIGS = {
    '1h': {'period': '729d', 'interval': '1h'},
    '15m': {'period': '59d', 'interval': '15m'},
    '5m': {'period': '59d', 'interval': '5m'},
    '1m': {'period': '6d', 'interval': '1m'}
}

def fetch_and_save_interval_data(interval_str: str, data_dir: str = 'data') -> str | None:
    """
    Fetches and saves historical market data for a single specified interval.

    Args:
        interval_str: The interval string (e.g., '1h', '15m').
        data_dir: The directory to save the CSV file.

    Returns:
        The file path of the saved CSV if successful, None otherwise.
    """
    if interval_str not in INTERVAL_CONFIGS:
        print(f"Error: Interval '{interval_str}' is not configured.")
        return None

    config = INTERVAL_CONFIGS[interval_str]
    period = config['period']
    yf_interval = config['interval']

    # Ensure data directory exists
    if not os.path.exists(data_dir):
        try:
            os.makedirs(data_dir)
            print(f"Created directory: {data_dir}")
        except OSError as e:
            print(f"Error creating directory {data_dir}: {e}")
            return None

    print(f"\nFetching data for interval: {interval_str} (yfinance interval: {yf_interval}, period: {period})")
    try:
        data = yf.download(TICKER_SYMBOL, interval=yf_interval, period=period, progress=False)

        if data.empty:
            print(f"No data fetched for interval: {interval_str}. Ticker '{TICKER_SYMBOL}' might be invalid or data unavailable.")
            return None
        else:
            print(f"Successfully fetched {len(data)} rows for interval: {interval_str}")
            file_name = f"ndx_{interval_str}.csv"
            file_path = os.path.join(data_dir, file_name)
            data.to_csv(file_path)
            print(f"Data saved to {file_path}")
            return file_path

    except Exception as e:
        print(f"An error occurred while fetching or saving data for interval {interval_str}: {e}")
        return None

def fetch_all_data():
    """
    Fetches historical market data for all configured intervals and saves them to CSV files.
    """
    print(f"Starting data fetch for all intervals for ticker: {TICKER_SYMBOL}")
    saved_files = []
    for interval_key in INTERVAL_CONFIGS.keys():
        file_path = fetch_and_save_interval_data(interval_key)
        if file_path:
            saved_files.append(file_path)

    if saved_files:
        print(f"\nSuccessfully saved data for: {', '.join(saved_files)}")
    else:
        print("\nNo data was saved.")


if __name__ == '__main__':
    # Example of how to run: fetch all data
    fetch_all_data()
    # Or fetch for a specific interval:
    # fetch_and_save_interval_data('1h')
    print("\nData fetching process completed.")
