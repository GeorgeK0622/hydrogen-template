import yfinance as yf

ticker_symbol = "^NDX"
ticker = yf.Ticker(ticker_symbol)

print(f"Checking data for {ticker_symbol}")

# Check 1-hour data
hist_1h = ticker.history(period="730d", interval="1h")
print(f"1-hour data available for last 730 days: {not hist_1h.empty}")
if not hist_1h.empty:
    print(f"1-hour data shape: {hist_1h.shape}")
    print(f"1-hour data head:\n{hist_1h.head()}")


# Check 15-minute data
# For intervals <1d, data is normally downloaded for last 60 days.
# Let's try to get more to see the limit.
hist_15m = ticker.history(period="60d", interval="15m")
print(f"15-minute data available for last 60 days: {not hist_15m.empty}")
if not hist_15m.empty:
    print(f"15-minute data shape: {hist_15m.shape}")

# Check 5-minute data
hist_5m = ticker.history(period="60d", interval="5m")
print(f"5-minute data available for last 60 days: {not hist_5m.empty}")
if not hist_5m.empty:
    print(f"5-minute data shape: {hist_5m.shape}")

# Check 1-minute data
hist_1m = ticker.history(period="7d", interval="1m") # Max 7 days for 1m interval
print(f"1-minute data available for last 7 days: {not hist_1m.empty}")
if not hist_1m.empty:
    print(f"1-minute data shape: {hist_1m.shape}")

print("\n--- Note on yfinance interval limitations ---")
print("1h: Can download up to 730 days (2 years) of data.")
print("15m, 5m: Data is typically available for the last 60 days when period is not specified or is large.")
print("           If period is specified as <= 60 days, it will return data for that period.")
print("1m: Data is typically available for the last 7 days (Downloads 1d data per request, max 7 requests).")
print("    To get more than 7 days of 1m data, you would need to make multiple calls with different start/end dates.")

print("\nPrimary Python libraries to be used:")
print("- Data manipulation: pandas, numpy")
print("- Data acquisition: yfinance")
print("- Time series modeling (initial focus): statsmodels, prophet, scikit-learn")
