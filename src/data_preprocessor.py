# Ensure pandas and numpy are installed. If not, run:
# pip install pandas numpy

import pandas as pd
import numpy as np
import os

def _load_raw_data(file_path: str) -> pd.DataFrame | None:
    """
    Internal function to load data from a CSV file into a pandas DataFrame,
    setting the Datetime index correctly and handling potential file issues.
    """
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return None

    try:
        # The first row is the true header.
        # The second row ("Ticker,^NDX,...") and third row ("Datetime,,,,,") are metadata/empty.
        df = pd.read_csv(file_path, header=0, skiprows=[1, 2], index_col=0, parse_dates=True)
        df.index.name = 'Datetime' # Ensure the index name is 'Datetime'
        print(f"Data loaded successfully from {file_path}, skipping metadata rows.")

        # Ensure OHLCV columns are numeric, coercing errors to NaN
        # This is important if any non-numeric strings (e.g. 'N/A') exist in data rows
        ohlcv_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in ohlcv_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # Drop rows where critical 'Close' data might have become NaN after coercion
        # This can happen if 'Close' had non-numeric data that wasn't part of skipped headers
        df.dropna(subset=['Close'], inplace=True)

        return df
    except FileNotFoundError: # Should be caught by os.path.exists, but as a fallback
        print(f"Error: File not found at {file_path} (exception caught).")
        return None
    except Exception as e:
        print(f"An error occurred during data loading or initial numeric conversion from {file_path}: {e}")
        return None

def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handles missing values in the DataFrame.
    - Fills OHLC missing values using forward-fill.
    - Fills Volume missing values using forward-fill, then 0 for any remaining NaNs.
    """
    ohlc_cols = ['Open', 'High', 'Low', 'Close']
    for col in ohlc_cols:
        if col in df.columns:
            df[col] = df[col].ffill() # Forward fill first
            df[col] = df[col].bfill() # Back fill if NaNs are at the beginning

    if 'Volume' in df.columns:
        df['Volume'] = df['Volume'].ffill()
        df['Volume'] = df['Volume'].bfill()
        df['Volume'] = df['Volume'].fillna(0) # Fill with 0 if all values were NaN

    # print("Missing values handled.") # Less verbose for library function
    return df

def calculate_log_returns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates the log return of the 'Close' price.
    log_return = numpy.log(df['Close'] / df['Close'].shift(1))
    """
    if 'Close' not in df.columns or df['Close'].isnull().all():
        print("Warning: 'Close' column not found or all values are NaN. Skipping log return calculation.")
        df['log_return'] = np.nan
        return df

    df['log_return'] = np.log(df['Close'] / df['Close'].shift(1))
    # print("Log returns calculated.")
    return df

def calculate_sma(df: pd.DataFrame, window: int) -> pd.DataFrame:
    """
    Calculates the Simple Moving Average (SMA) for the 'Close' price.
    """
    if 'Close' not in df.columns or df['Close'].isnull().all():
        print(f"Warning: 'Close' column not found or all values are NaN. Skipping SMA-{window} calculation.")
        df[f'SMA_{window}'] = np.nan
        return df

    df[f'SMA_{window}'] = df['Close'].rolling(window=window, min_periods=1).mean() # min_periods=1 to avoid NaNs at start due to window
    # print(f"SMA-{window} calculated.")
    return df

def calculate_rsi(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """
    Calculates the Relative Strength Index (RSI).
    """
    if 'Close' not in df.columns or df['Close'].isnull().all():
        print(f"Warning: 'Close' column not found or all values are NaN. Skipping RSI calculation.")
        df[f'RSI_{window}'] = np.nan
        return df

    delta = df['Close'].diff(1)

    gain = delta.where(delta > 0, 0.0) # Ensure 0.0 for float type
    loss = -delta.where(delta < 0, 0.0) # Loss is positive, ensure 0.0

    # Calculate average gain and loss using simple moving average for initial RSI
    avg_gain = gain.rolling(window=window, min_periods=1).mean()
    avg_loss = loss.rolling(window=window, min_periods=1).mean()

    # Calculate RS
    epsilon = 1e-10
    rs = avg_gain / (avg_loss + epsilon)

    rsi = 100 - (100 / (1 + rs))

    df[f'RSI_{window}'] = rsi
    # print(f"RSI-{window} calculated.")
    return df

def load_and_preprocess_data(file_path: str, processed_data_dir: str = 'processed_data') -> pd.DataFrame | None:
    """
    Loads raw data, applies all preprocessing steps, saves the processed data,
    and returns the preprocessed DataFrame.
    """
    print(f"\n--- Starting Preprocessing for {file_path} ---")
    df = _load_raw_data(file_path)
    if df is None or df.empty:
        print(f"Failed to load or data is empty for {file_path}. Skipping preprocessing.")
        return None

    # Apply preprocessing steps
    df = handle_missing_values(df)
    df = calculate_log_returns(df)
    df = calculate_sma(df, window=7)
    df = calculate_sma(df, window=20)
    df = calculate_rsi(df, window=14)

    print("Preprocessing steps applied.")

    # Ensure processed_data directory exists
    if not os.path.exists(processed_data_dir):
        try:
            os.makedirs(processed_data_dir)
            print(f"Created directory: {processed_data_dir}")
        except OSError as e:
            print(f"Error creating directory {processed_data_dir}: {e}")
            # Still return the df, just don't save it
            return df

    # Save preprocessed data
    base_name = os.path.basename(file_path)
    output_file_name = f"preprocessed_{base_name}"
    output_file_path = os.path.join(processed_data_dir, output_file_name)

    try:
        df.to_csv(output_file_path)
        print(f"Preprocessed data saved to {output_file_path}")
    except Exception as e:
        print(f"An error occurred while saving the preprocessed data to {output_file_path}: {e}")

    print(f"--- Completed Preprocessing for {file_path} ---")
    return df

def main():
    """
    Main function to demonstrate loading, preprocessing, and saving data.
    """
    sample_file_path = 'data/ndx_1h.csv' # Example input file

    df_processed = load_and_preprocess_data(sample_file_path)

    if df_processed is not None:
        print("\n--- DataFrame Head after Preprocessing (from main) ---")
        print(df_processed.head(15))

        print("\n--- DataFrame Tail after Preprocessing (from main) ---")
        print(df_processed.tail())

        print("\n--- Missing Values After Preprocessing (from main) ---")
        print(df_processed.isnull().sum())

if __name__ == '__main__':
    main()
