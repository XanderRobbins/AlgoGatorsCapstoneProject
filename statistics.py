import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

window=4
stock_symbol = 'NVDA'
market_symbol = '^GSPC'
event_date = pd.to_datetime('2024-12-04')

stock_data = yf.download(stock_symbol, start='2010-01-01', end='2025-01-01')
market_data = yf.download(market_symbol, start='2010-01-01', end='2025-01-01')

# Calculate daily returns for the stock (use 'Close' for stock price)
stock_data['Returns'] = stock_data['Close'].pct_change()
# Calculate daily returns for the market (use 'Close' for market price)
market_data['Returns'] = market_data['Close'].pct_change()

def get_event_window(stock_data, event_date, window):
    # five days before event
    start_date = event_date - pd.Timedelta(days=window)
    # five days after event
    end_date = event_date + pd.Timedelta(days=window)
    # Extract the event window data
    event_window = stock_data.loc[start_date:end_date]
    return event_window

event_window = get_event_window(stock_data, event_date, window)
if event_window is not None:
    print(event_window)

# Merge stock data and market data on the Date index
merged_data = pd.merge(stock_data[['Returns']], market_data[['Returns']], left_index=True, right_index=True)
# Calculate abnormal returns (AR) by subtracting the market returns from the stock returns
merged_data['Abnormal_Returns'] = stock_data['Returns'] - market_data['Returns']
# Calculate Cumulative Abnormal Returns (CAR)
merged_data['Cumulative_Abnormal_Returns'] = merged_data['Abnormal_Returns'].cumsum()

event_start = event_date - pd.Timedelta(days=window)
event_end = event_date + pd.Timedelta(days=window)
event_window_data = merged_data.loc[event_start:event_end]
event_window_CAR = event_window_data['Abnormal_Returns'].sum()

# Drop any missing values (NaNs) before testing
abnormal_returns = merged_data['Abnormal_Returns'].dropna()

# Perform a one-sample t-test against a mean of 0
t_stat, p_value = stats.ttest_1samp(abnormal_returns, 0)

pre_event_returns = stock_data.loc[event_start:event_date, 'Returns'].dropna()
post_event_returns = stock_data.loc[event_date:event_end, 'Returns'].dropna()

# Calculate volatility (standard deviation of returns)
pre_volatility = pre_event_returns.std()
post_volatility = post_event_returns.std()

print("Event Window CAR:", event_window_CAR)
print(f"T-statistic: {t_stat:.3f}")
print(f"P-value: {p_value:.3f}")
print(f"Pre-Event Volatility: {pre_volatility:.4f}")
print(f"Post-Event Volatility: {post_volatility:.4f}")

# Create subplots
fig, axs = plt.subplots(3, 1, figsize=(12, 18))

# Plotting Cumulative Abnormal Returns for the event window
axs[0].plot(event_window_data.index, event_window_data['Cumulative_Abnormal_Returns'], label='Cumulative Abnormal Returns', color='orange')
axs[0].axvline(event_date, color='orange', linestyle='--', label='Event Date')
axs[0].axhline(y=0, color='blue', linestyle='--')
axs[0].set_title('Cumulative Abnormal Returns')
axs[0].set_xlabel('Date')
axs[0].set_ylabel('Cumulative Abnormal Return')
axs[0].legend()

# Plotting Daily Returns Around Event for the event window
event_stock_window_data = stock_data.loc[event_start:event_end]  # Extract event window data for stock
axs[1].plot(event_stock_window_data.index, event_stock_window_data['Returns'], label='Returns')
axs[1].axvline(event_date, color='orange', linestyle='--', label='Event Date')
axs[1].set_title('Daily Returns Around Event')
axs[1].set_xlabel('Date')
axs[1].set_ylabel('Daily Return')
axs[1].legend()

# Plotting Daily Abnormal Returns for the event window
axs[2].plot(event_window_data.index, event_window_data['Abnormal_Returns'], label='Abnormal Returns')
axs[2].axhline(y=0, color='blue', linestyle='--')
axs[2].axvline(event_date, color='orange', linestyle='--', label='Event Date')
axs[2].set_title('Daily Abnormal Returns')
axs[2].set_xlabel('Date')
axs[2].set_ylabel('Abnormal Return')
axs[2].legend()

# Adjust layout
plt.tight_layout()
plt.show()