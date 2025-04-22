import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# Parameters
window = 365
stock_symbol = 'TSLA'
market_symbol = '^GSPC'
event_date = pd.to_datetime('2021-04-26')

# Download stock and market data
stock_data = yf.download(stock_symbol, start='2010-01-01', end='2025-01-01')
market_data = yf.download(market_symbol, start='2010-01-01', end='2025-01-01')

# Calculate daily returns
stock_data['Returns'] = stock_data['Close'].pct_change()
market_data['Returns'] = market_data['Close'].pct_change()

# Merge stock and market data
merged = pd.merge(
    stock_data['Returns'].rename('Stock_Ret'),
    market_data['Returns'].rename('Mkt_Ret'),
    left_index=True, right_index=True
)
merged['Abnormal_Returns'] = merged['Stock_Ret'] - merged['Mkt_Ret']

# Extract event window data (±30 days)
ev = merged.loc[event_date - pd.Timedelta(days=window): event_date + pd.Timedelta(days=window)].copy()

# Compute Cumulative Abnormal Returns (CAR) starting from 0 at the first day of the event window
ev['CAR'] = ev['Abnormal_Returns'].cumsum()

# Pre-event and post-event abnormal returns for volatility calculation
pre_AR = ev.loc[ev.index < event_date, 'Abnormal_Returns'].dropna()
post_AR_for_test = ev.loc[ev.index >= event_date, 'Abnormal_Returns'].dropna()

# Perform one-sided t-test (testing if AR > 0)
t_stat, p_two = stats.ttest_1samp(post_AR_for_test, 0)
p_one = p_two / 2 if t_stat > 0 else 1 - p_two / 2

# Calculate volatility
vol_pre = pre_AR.std()
vol_post = post_AR_for_test.std()

# Output results
print(f"T-statistic: {t_stat:.3f}")
print(f"One-Sided P-value (AR > 0): {p_one:.3f}")
print(f"Event Window CAR: {ev['Abnormal_Returns'].sum():.4f}")
print(f"Pre-Event Volatility: {vol_pre:.4f}")
print(f"Post-Event Volatility: {vol_post:.4f}")

# Plotting results
fig, axs = plt.subplots(3, 1, figsize=(12, 18))

# Plot Cumulative Abnormal Returns (CAR)
axs[0].plot(ev.index, ev['CAR'], label='Cumulative Abnormal Returns', color='orange')
axs[0].axvline(event_date, color='orange', linestyle='--', label='Event Date')
axs[0].axhline(y=0, color='blue', linestyle='--')
axs[0].set_title('Cumulative Abnormal Returns')
axs[0].set_xlabel('Date')
axs[0].set_ylabel('Cumulative Abnormal Return')
axs[0].legend()

# Plot Daily Returns Around Event for the event window
axs[1].plot(ev.index, ev['Stock_Ret'], label='Stock Returns')
axs[1].axvline(event_date, color='orange', linestyle='--', label='Event Date')
axs[1].set_title('Daily Returns Around Event')
axs[1].set_xlabel('Date')
axs[1].set_ylabel('Daily Return')
axs[1].legend()

# Plot Daily Abnormal Returns for the event window
axs[2].plot(ev.index, ev['Abnormal_Returns'], label='Abnormal Returns')
axs[2].axhline(y=0, color='blue', linestyle='--')
axs[2].axvline(event_date, color='orange', linestyle='--', label='Event Date')
axs[2].set_title('Daily Abnormal Returns')
axs[2].set_xlabel('Date')
axs[2].set_ylabel('Abnormal Return')
axs[2].legend()

# Adjust layout for better spacing
plt.tight_layout()
plt.show()
