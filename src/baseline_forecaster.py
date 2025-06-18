# Ensure pandas and prophet are installed. If not, run:
# pip install pandas prophet matplotlib

import pandas as pd
from prophet import Prophet
import os
import matplotlib.pyplot as plt # Required for saving plots

def generate_forecast_and_signal(
        data_df: pd.DataFrame,
        forecast_periods: int,
        interval_frequency_char: str,
        interval_str_for_filename: str, # e.g. '1h', '15m' for filenames
        plots_dir: str = 'forecast_plots'
    ) -> dict | None:
    """
    Takes a preprocessed DataFrame, fits a Prophet model, makes predictions,
    generates a trading signal, saves plots, and returns forecast details.

    Args:
        data_df: Preprocessed DataFrame with 'Datetime' index and 'Close' column.
        forecast_periods: Number of periods to forecast ahead.
        interval_frequency_char: Frequency character for Prophet (e.g., 'h', '15min').
        interval_str_for_filename: Interval string for naming plot files (e.g., '1h', '15m').
        plots_dir: Directory to save plots.

    Returns:
        A dictionary with forecast details if successful, None otherwise.
    """
    if data_df is None or data_df.empty:
        print("Error: Input DataFrame is empty or None.")
        return None

    # 1. Prepare data for Prophet
    prophet_df = data_df.reset_index()
    prophet_df = prophet_df[['Datetime', 'Close']]
    prophet_df.rename(columns={'Datetime': 'ds', 'Close': 'y'}, inplace=True)

    if prophet_df['ds'].dt.tz is not None:
        # print("Removing timezone information from 'ds' column for Prophet.")
        prophet_df['ds'] = prophet_df['ds'].dt.tz_localize(None)

    prophet_df.dropna(subset=['y'], inplace=True)
    if prophet_df.empty:
        print("Error: DataFrame became empty after dropping NaNs from 'y'. Cannot proceed.")
        return None

    # print("\nData prepared for Prophet:")
    # print(prophet_df.head())

    # 2. Initialize and fit the Prophet model
    # print("\nInitializing and fitting Prophet model...")
    try:
        model = Prophet()
        model.fit(prophet_df)
        # print("Prophet model fitted successfully.")
    except Exception as e:
        print(f"Error fitting Prophet model: {e}")
        return None

    # 3. Create a future DataFrame for predictions
    try:
        future = model.make_future_dataframe(periods=forecast_periods, freq=interval_frequency_char)
        # print("\nFuture DataFrame created for predictions:")
        # print(future.tail())
    except Exception as e:
        print(f"Error creating future dataframe: {e}")
        return None

    # 4. Generate forecasts
    try:
        forecast = model.predict(future)
        # print("\nForecast generated successfully.")
    except Exception as e:
        print(f"Error generating forecast: {e}")
        return None

    # print("\nTail of the forecast DataFrame (showing predictions):")
    # print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())

    # 5. Generate Trading Signal
    signal_details = {
        "last_actual_datetime": None,
        "last_actual_price": None,
        "forecasted_datetime_next_period": None,
        "forecasted_price_next_period": None,
        "signal": "ERROR", # Default to error
        "plot_paths": []
    }

    try:
        last_actual_date = prophet_df['ds'].max()
        last_actual_price = prophet_df[prophet_df['ds'] == last_actual_date]['y'].iloc[0]
        signal_details["last_actual_datetime"] = last_actual_date
        signal_details["last_actual_price"] = last_actual_price

        next_period_forecast_ds = forecast[forecast['ds'] > last_actual_date]['ds'].min()
        next_period_forecast_row = forecast[forecast['ds'] == next_period_forecast_ds]

        if not next_period_forecast_row.empty:
            forecasted_price_next_period = next_period_forecast_row['yhat'].iloc[0]
            signal_details["forecasted_datetime_next_period"] = next_period_forecast_ds
            signal_details["forecasted_price_next_period"] = forecasted_price_next_period

            threshold_percentage = 0.001  # 0.1%
            predicted_change = forecasted_price_next_period - last_actual_price
            threshold_value = last_actual_price * threshold_percentage

            current_signal = 'HOLD'
            if predicted_change > threshold_value:
                current_signal = 'BUY'
            elif predicted_change < -threshold_value:
                current_signal = 'SELL'
            signal_details["signal"] = current_signal
        else:
            print("Error: Could not find forecast for the next period for signal generation.")
            # Keep signal as "ERROR"

    except Exception as e:
        print(f"Error generating trading signal details: {e}")
        # Signal remains "ERROR"

    # Create forecast_plots directory if it doesn't exist
    if not os.path.exists(plots_dir):
        try:
            os.makedirs(plots_dir)
            print(f"Created directory: {plots_dir}")
        except OSError as e:
            print(f"Error creating directory {plots_dir}: {e}")
            # Continue without saving plots if directory creation fails

    plot_paths = []
    # Plot the forecast
    try:
        # print("\nPlotting forecast...")
        fig1 = model.plot(forecast)
        plot_file1 = os.path.join(plots_dir, f'prophet_forecast_{interval_str_for_filename}.png')
        plt.savefig(plot_file1)
        plt.close(fig1)
        plot_paths.append(plot_file1)
        # print(f"Forecast plot saved to {plot_file1}")
    except Exception as e:
        print(f"Error plotting or saving forecast plot: {e}")

    # Plot the forecast components
    try:
        # print("\nPlotting forecast components...")
        fig2 = model.plot_components(forecast)
        plot_file2 = os.path.join(plots_dir, f'prophet_components_{interval_str_for_filename}.png')
        plt.savefig(plot_file2)
        plt.close(fig2)
        plot_paths.append(plot_file2)
        # print(f"Forecast components plot saved to {plot_file2}")
    except Exception as e:
        print(f"Error plotting or saving forecast components plot: {e}")

    signal_details["plot_paths"] = plot_paths
    return signal_details

def main_run_single_forecast(): # Renamed from run_prophet_forecast
    """
    Loads preprocessed data for 1h, fits Prophet, makes predictions,
    generates signal and saves plots. (Example usage of the refactored function)
    """
    preprocessed_file_path = 'processed_data/preprocessed_ndx_1h.csv'

    if not os.path.exists(preprocessed_file_path):
        print(f"Error: Preprocessed data file not found at {preprocessed_file_path}")
        return

    try:
        data_df = pd.read_csv(preprocessed_file_path, index_col='Datetime', parse_dates=True)
        print(f"Successfully loaded data from {preprocessed_file_path}")
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    forecast_details = generate_forecast_and_signal(
        data_df=data_df,
        forecast_periods=24,
        interval_frequency_char='h',
        interval_str_for_filename='1h_main_example' # To distinguish from run_forecast.py outputs
    )

    if forecast_details:
        print("\n--- Forecast Details (from main_run_single_forecast) ---")
        print(f"Last actual Datetime: {forecast_details['last_actual_datetime']}")
        print(f"Last actual 'Close' price: {forecast_details['last_actual_price']:.2f}")
        print(f"Forecasted Datetime for next period: {forecast_details['forecasted_datetime_next_period']}")
        print(f"Forecasted 'Close' price for next period: {forecast_details['forecasted_price_next_period']:.2f}")
        print(f"Generated Signal: {forecast_details['signal']}")
        if forecast_details['plot_paths']:
            print(f"Plots saved to: {', '.join(forecast_details['plot_paths'])}")
        else:
            print("Plots could not be saved.")


if __name__ == '__main__':
    main_run_single_forecast()
    print("\nProphet forecasting process completed.")
