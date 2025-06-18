# Main script to run the forecasting pipeline for a specified interval.
# Expected to be run after data_fetcher.py has populated the 'data' directory
# and data_preprocessor.py can process files from 'data' to 'processed_data'.

# pip install pandas prophet matplotlib (if not already installed)

import argparse
import os
import pandas as pd

# Assuming refactored scripts are in the same directory or PYTHONPATH is set
# from data_fetcher import fetch_and_save_interval_data # Option 1: Fetch on the fly
from data_preprocessor import load_and_preprocess_data
from baseline_forecaster import generate_forecast_and_signal

# Define configurations for each interval
# (Could also be in a shared config file)
FORECAST_CONFIG = {
    '1h': {'periods': 24, 'freq': 'h', 'raw_file_template': 'data/ndx_1h.csv', 'processed_file_template': 'processed_data/preprocessed_ndx_1h.csv'},
    '15m': {'periods': 4*4, 'freq': '15min', 'raw_file_template': 'data/ndx_15m.csv', 'processed_file_template': 'processed_data/preprocessed_ndx_15m.csv'},
    '5m': {'periods': 12*2, 'freq': '5min', 'raw_file_template': 'data/ndx_5m.csv', 'processed_file_template': 'processed_data/preprocessed_ndx_5m.csv'},
    '1m': {'periods': 60, 'freq': 'min', 'raw_file_template': 'data/ndx_1m.csv', 'processed_file_template': 'processed_data/preprocessed_ndx_1m.csv'}
}

def main():
    parser = argparse.ArgumentParser(description="Run NASDAQ 100 forecast for a specific interval.")
    parser.add_argument(
        '--interval',
        type=str,
        required=True,
        choices=['1h', '15m', '5m', '1m'],
        help="The interval to forecast (e.g., '1h', '15m')."
    )
    args = parser.parse_args()
    interval = args.interval

    print(f"Starting forecast pipeline for interval: {interval}")

    # --- Step 1: Data Fetching (Assumed to be done by data_fetcher.py separately) ---
    # For this script, we'll assume the raw CSV data already exists.
    # Optionally, you could call fetch_and_save_interval_data here:
    # raw_data_path = fetch_and_save_interval_data(interval, data_dir='data')
    # if not raw_data_path:
    #     print(f"Could not fetch data for {interval}. Exiting.")
    #     return

    raw_data_path = FORECAST_CONFIG[interval]['raw_file_template']
    if not os.path.exists(raw_data_path):
        print(f"Error: Raw data file {raw_data_path} not found.")
        print(f"Please run 'python src/data_fetcher.py' first to download all data,")
        print(f"or specifically 'python src/data_fetcher.py --interval {interval}' if that option were added.")
        return

    # --- Step 2: Data Preprocessing ---
    # The load_and_preprocess_data function saves its output and returns the df
    preprocessed_df = load_and_preprocess_data(
        file_path=raw_data_path,
        processed_data_dir=os.path.dirname(FORECAST_CONFIG[interval]['processed_file_template'])
    )

    if preprocessed_df is None:
        print(f"Preprocessing failed for {interval}. Exiting.")
        return

    # --- Step 3: Forecasting and Signal Generation ---
    forecast_params = FORECAST_CONFIG[interval]
    forecast_details = generate_forecast_and_signal(
        data_df=preprocessed_df,
        forecast_periods=forecast_params['periods'],
        interval_frequency_char=forecast_params['freq'],
        interval_str_for_filename=interval, # Use the direct interval string like '1h'
        plots_dir='forecast_plots' # Standardized plots directory
    )

    if not forecast_details:
        print(f"Forecast generation failed for {interval}. Exiting.")
        return

    # --- Step 4: Print Summary ---
    print(f"\n--- Forecast for NASDAQ 100 ({interval}) ---")
    if forecast_details.get("last_actual_price") is not None:
        print(f"Last Actual Price ({forecast_details['last_actual_datetime']}): {forecast_details['last_actual_price']:.2f}")
    else:
        print("Last Actual Price: Not available")

    if forecast_details.get("forecasted_price_next_period") is not None:
        print(f"Forecasted Next Price ({forecast_details['forecasted_datetime_next_period']}): {forecast_details['forecasted_price_next_period']:.2f}")
    else:
        print("Forecasted Next Price: Not available")

    print(f"Generated Signal: {forecast_details['signal']}")
    if forecast_details['plot_paths']:
        print(f"Forecast plots saved to: {', '.join(forecast_details['plot_paths'])}")
    else:
        print("Forecast plots were not saved.")

    print(f"\nPipeline completed for interval: {interval}")

if __name__ == '__main__':
    main()
