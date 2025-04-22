import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from tabulate import tabulate
summary_rows = []

def download_data(stock_symbol, market_symbol, start_date, end_date):
    stock_data = yf.download(stock_symbol, start=start_date, end=end_date)
    market_data = yf.download(market_symbol, start=start_date, end=end_date)
    return stock_data, market_data

def calculate_returns(stock_data, market_data):
    # Calculate returns for each dataset
    stock_data['Returns'] = stock_data['Close'].pct_change()
    market_data['Returns'] = market_data['Close'].pct_change()

    # Extract returns and rename to avoid conflict during merge
    stock_returns = stock_data[['Returns']]
    market_returns = market_data[['Returns']]

    # Merge without suffixes by renaming before merging
    stock_returns.columns = ['Stock_Returns']
    market_returns.columns = ['Market_Returns']

    merged_data = pd.merge(stock_returns, market_returns, left_index=True, right_index=True)

    # Calculate abnormal and cumulative abnormal returns
    merged_data['Abnormal_Returns'] = merged_data['Stock_Returns'] - merged_data['Market_Returns']
    merged_data['Cumulative_Abnormal_Returns'] = merged_data['Abnormal_Returns'].cumsum()

    return merged_data

def get_event_window(data, event_date, window):
    start_date = event_date - pd.Timedelta(days=window)
    end_date = event_date + pd.Timedelta(days=window)
    return data.loc[start_date:end_date], (data.loc[start_date:end_date].index - event_date).days

def analyze_event(stock_data, merged, ed, window, ticker):
    # 1) Build your window bounds
    pre_start  = ed - pd.Timedelta(days=window)
    pre_end    = ed - pd.Timedelta(days=1)
    post_start = ed + pd.Timedelta(days=1)
    post_end   = ed + pd.Timedelta(days=window)

    # 2) Slice abnormal returns for full window and for post-event only
    ew, days = get_event_window(merged, ed, window)
    CAR_full = ew['Abnormal_Returns'].sum()
    CAR_post = ew.loc[days >= 0, 'Abnormal_Returns'].sum()

    # 3) T-test on the post-event abnormal returns
    ar_post    = ew.loc[days >= 0, 'Abnormal_Returns'].dropna()
    t_stat, p  = stats.ttest_1samp(ar_post, 0)

    # 4) Volatility (STD of returns) excluding the event day duplication
    pre_R  = stock_data.loc[pre_start: pre_end, 'Returns'].dropna()
    post_R = stock_data.loc[post_start: post_end, 'Returns'].dropna()
    vol_pre  = pre_R.std()
    vol_post = post_R.std()

    # 5) Average volume
    vol_pre_avg  = stock_data.loc[pre_start: pre_end, 'Volume'].mean()
    vol_post_avg = stock_data.loc[post_start: post_end, 'Volume'].mean()

    summary_rows.append({
      'Ticker':      ticker,
      'Window':      f"{window}d",
      'Event Date':  ed.strftime("%Y-%m-%d"),
      'CAR (%)':     round(CAR_post * 100, 2),
      'T-stat':      round(t_stat,  3),
      'P-value':     round(p,       3),
      'Pre-Vol (%)': round(vol_pre  * 100, 2),
      'Post-Vol (%)':round(vol_post * 100, 2),
      'Pre-Volume':  round(vol_pre_avg, 0),
      'Post-Volume': round(vol_post_avg, 0)
    })

    return ew, days, pre_R, post_R

def calculate_rsi(data, window=14):
    delta = data['Close'].diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=window, min_periods=1).mean()
    avg_loss = loss.rolling(window=window, min_periods=1).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    data['RSI'] = rsi
    return data

def print_summary_table():
    print("\nEvent Analysis Summary:")
    print(tabulate(summary_rows, headers="keys", tablefmt="pretty"))

def plot_data(all_event_data, window_label):
    # --------- FIGURE 1: Cumulative Abnormal Returns + Daily Returns ---------
    fig1, axs1 = plt.subplots(2, 1, figsize=(12, 10))

    # Cumulative Abnormal Returns
    for event_data in all_event_data:
        axs1[0].plot(event_data['days_offset'], event_data['event_window_data']['Cumulative_Abnormal_Returns'],
                     label=event_data['ticker'])
    axs1[0].axhline(y=0, color='blue', linestyle='--')
    axs1[0].axvline(x=0, color='orange', linestyle='--', label='Event Date')
    axs1[0].set_title('Cumulative Abnormal Returns')
    axs1[0].set_xlabel('Days Relative to Event')
    axs1[0].set_ylabel('Cumulative Abnormal Return')
    axs1[0].legend()

    # Daily Returns
    for event_data in all_event_data:
        axs1[1].plot(event_data['days_offset'], event_data['event_window_data']['Stock_Returns'],
                     label=event_data['ticker'])
    axs1[1].axvline(x=0, color='orange', linestyle='--', label='Event Date')
    axs1[1].set_title('Daily Returns Around Event')
    axs1[1].set_xlabel('Days Relative to Event')
    axs1[1].set_ylabel('Daily Return')
    axs1[1].legend()

    plt.tight_layout()
    plt.suptitle(f"Figure 1 - Price Movements - {window_label}", y=1.03)
    plt.show()
    plt.close(fig1)

    # --------- FIGURE 2: Daily Abnormal Returns + Rolling Volatility ---------
    fig2, axs2 = plt.subplots(2, 1, figsize=(12, 10))

    # Daily Abnormal Returns
    for event_data in all_event_data:
        axs2[0].plot(event_data['days_offset'], event_data['event_window_data']['Abnormal_Returns'],
                     label=event_data['ticker'])
    axs2[0].axhline(y=0, color='blue', linestyle='--')
    axs2[0].axvline(x=0, color='orange', linestyle='--', label='Event Date')
    axs2[0].set_title('Daily Abnormal Returns')
    axs2[0].set_xlabel('Days Relative to Event')
    axs2[0].set_ylabel('Abnormal Return')
    axs2[0].legend()

    # Rolling Volatility
    window_size = 20
    for event_data in all_event_data:
        pre_volatility = event_data['pre_event_returns'].rolling(window=window_size).std()
        post_volatility = event_data['post_event_returns'].rolling(window=window_size).std()
        axs2[1].plot(pre_volatility.index, pre_volatility, label=f'{event_data["ticker"]} Pre-Event Volatility')
        axs2[1].plot(post_volatility.index, post_volatility, linestyle='--',
                     label=f'{event_data["ticker"]} Post-Event Volatility')
    axs2[1].axvline(x=event_data['days_offset'][0], color='orange', linestyle='--', label='Event Date')
    axs2[1].set_title('Rolling Volatility')
    axs2[1].set_xlabel('Days Relative to Event')
    axs2[1].set_ylabel('Volatility')
    axs2[1].legend()

    plt.tight_layout()
    plt.suptitle(f"Figure 2 - Risk Indicators - {window_label}", y=1.03)
    plt.show()
    plt.close(fig2)

events = [

    {"stock_symbol": "AAPL", "event_date": "2011-08-24"},
    {"stock_symbol": "YHOO", "event_date": "2012-07-17"},
    {"stock_symbol": "IBM", "event_date": "2012-01-01"},
    {"stock_symbol": "MSFT", "event_date": "2014-02-04"},
    {"stock_symbol": "GM", "event_date": "2014-01-15"},
    {"stock_symbol": "GOOGL", "event_date": "2015-10-01"},
    {"stock_symbol": "AMZN", "event_date": "2018-07-05"},
    {"stock_symbol": "NFLX", "event_date": "2019-01-01"},
    {"stock_symbol": "TSLA", "event_date": "2021-07-01"},
    {"stock_symbol": "META", "event_date": "2022-01-01"},
    {"stock_symbol": "IBM", "event_date": "2020-02-03"},
    {"stock_symbol": "NDAQ", "event_date": "2017-01-01"},
    {"stock_symbol": "ACN", "event_date": "2019-10-01"},
    {"stock_symbol": "SAP", "event_date": "2014-04-01"},
    {"stock_symbol": "ORCL", "event_date": "2014-10-01"},
    {"stock_symbol": "CSCO", "event_date": "2015-07-09"},
    {"stock_symbol": "QCOM", "event_date": "2018-03-01"},
    {"stock_symbol": "INTC", "event_date": "2019-01-31"},
    {"stock_symbol": "ADBE", "event_date": "2022-12-01"},
    {"stock_symbol": "AMD", "event_date": "2014-10-08"},
    {"stock_symbol": "CRM", "event_date": "2021-08-02"},
    {"stock_symbol": "SAP", "event_date": "2019-06-29"},
    {"stock_symbol": "UBER", "event_date": "2020-08-24"},
    {"stock_symbol": "LYFT", "event_date": "2021-06-25"},
    {"stock_symbol": "BABA", "event_date": "2013-05-10"},
    {"stock_symbol": "TSM", "event_date": "2018-06-15"},
    {"stock_symbol": "V", "event_date": "2012-12-01"},
    {"stock_symbol": "MA", "event_date": "2010-11-01"},
    {"stock_symbol": "JNJ", "event_date": "2012-04-01"},
    {"stock_symbol": "PFE", "event_date": "2019-01-07"},
    {"stock_symbol": "AAPL", "event_date": "2014-06-09"},
    {"stock_symbol": "AAPL", "event_date": "2020-08-31"},
    {"stock_symbol": "TSLA", "event_date": "2020-08-31"},
    {"stock_symbol": "TSLA", "event_date": "2022-08-25"},
    {"stock_symbol": "NVDA", "event_date": "2021-07-20"},
    {"stock_symbol": "NVDA", "event_date": "2024-06-10"},
    {"stock_symbol": "GOOGL", "event_date": "2014-04-03"},
    {"stock_symbol": "AMZN", "event_date": "2022-06-06"},
    {"stock_symbol": "MSFT", "event_date": "2022-09-22"},
    {"stock_symbol": "FB", "event_date": "2013-09-20"},
    {"stock_symbol": "BRK.A", "event_date": "2020-12-01"},
    {"stock_symbol": "BABA", "event_date": "2017-09-20"},
    {"stock_symbol": "ORCL", "event_date": "2020-08-15"},
    {"stock_symbol": "ADBE", "event_date": "2013-09-01"},
    {"stock_symbol": "INTC", "event_date": "2011-06-24"},
    {"stock_symbol": "CSCO", "event_date": "2024-05-01"},
    {"stock_symbol": "QCOM", "event_date": "2015-03-18"},
    {"stock_symbol": "PYPL", "event_date": "2020-03-31"},
    {"stock_symbol": "NFLX", "event_date": "2015-01-01"},
    {"stock_symbol": "UBER", "event_date": "2021-04-15"},
    {"stock_symbol": "LYFT", "event_date": "2023-05-01"},
    {"stock_symbol": "AMD", "event_date": "2016-05-31"},
    {"stock_symbol": "TSM", "event_date": "2021-01-25"},
    {"stock_symbol": "SHOP", "event_date": "2015-08-20"},
    {"stock_symbol": "SQ", "event_date": "2017-11-20"},
    {"stock_symbol": "TWTR", "event_date": "2022-07-15"},
    {"stock_symbol": "ZM", "event_date": "2023-03-05"},
    {"stock_symbol": "SNAP", "event_date": "2024-02-10"},
    {"stock_symbol": "DOCU", "event_date": "2022-08-01"},
    {"stock_symbol": "ROKU", "event_date": "2021-02-15"},
    {"stock_symbol": "AMZN",    "event_date": "2017-06-16"},
      {"stock_symbol": "MSFT",    "event_date": "2016-12-08"},
      {"stock_symbol": "TWTR",    "event_date": "2022-10-27"},
      {"stock_symbol": "DIS",     "event_date": "2019-03-20"},
      {"stock_symbol": "T",       "event_date": "2018-06-14"},
      {"stock_symbol": "DOW",     "event_date": "2017-08-31"},
      {"stock_symbol": "IBM",     "event_date": "2020-10-30"},
      {"stock_symbol": "META",    "event_date": "2014-07-22"},
      {"stock_symbol": "AAPL",    "event_date": "2014-02-27"},
      {"stock_symbol": "GOOGL",   "event_date": "2005-08-17"},
      {"stock_symbol": "ORCL",    "event_date": "2016-11-15"},
      {"stock_symbol": "SAP",     "event_date": "2016-10-09"},
      {"stock_symbol": "V",       "event_date": "2010-03-18"},
      {"stock_symbol": "MA",      "event_date": "2011-10-04"},
      {"stock_symbol": "C",       "event_date": "2013-04-27"},
      {"stock_symbol": "BAC",     "event_date": "2010-09-01"},
      {"stock_symbol": "CMG",     "event_date": "2021-12-31"},
      {"stock_symbol": "ABT",     "event_date": "2020-03-27"},
      {"stock_symbol": "UL",      "event_date": "2019-01-11"},
      {"stock_symbol": "TOT",     "event_date": "2018-05-10"},
      {"stock_symbol": "CVS",     "event_date": "2018-11-28"},
      {"stock_symbol": "CHTR",    "event_date": "2016-05-18"},
      {"stock_symbol": "CVX",     "event_date": "2011-07-17"},
      {"stock_symbol": "UNP",     "event_date": "2015-06-01"},
      {"stock_symbol": "DAL",     "event_date": "2013-10-16"},
      {"stock_symbol": "QCOM",    "event_date": "2018-09-03"},
      {"stock_symbol": "FDX",     "event_date": "2017-08-18"},
      {"stock_symbol": "DPZ",     "event_date": "2012-01-01"},
      {"stock_symbol": "JNJ",     "event_date": "2010-09-24"},
      {"stock_symbol": "MMM",     "event_date": "2015-08-06"}


]

market_symbol = '^GSPC'
start_date = '2010-01-01'
end_date = '2025-01-01'

windows = {'1w': 7, '1m': 30, '1y': 365}

for window_label, window in windows.items():
    all_event_data = []
    for event in events:
        stock_symbol = event['stock_symbol']
        event_date = pd.to_datetime(event['event_date'])

        stock_data, market_data = download_data(stock_symbol, market_symbol, start_date, end_date)

        if stock_data.empty or market_data.empty:
            print(f"Data missing for {stock_symbol} on {event_date}. Skipping this event.")
            continue

        stock_data = calculate_rsi(stock_data)  # Calculate RSI before merging data
        merged_data = calculate_returns(stock_data, market_data)

        if event_date not in stock_data.index:
            print(f"Event date {event_date} not in stock data for {stock_symbol}. Skipping this window.")
            continue

        event_window_data, days_offset, pre_event_returns, post_event_returns = analyze_event(stock_data, merged_data,
                                                                                              event_date, window,
                                                                                              stock_symbol)
        event_stock_window_data, _ = get_event_window(stock_data, event_date, window)

        all_event_data.append({
            'event_window_data': event_window_data,
            'event_stock_window_data': event_stock_window_data,
            'days_offset': days_offset,
            'pre_event_returns': pre_event_returns,
            'post_event_returns': post_event_returns,
            'ticker': stock_symbol
        })

    plot_data(all_event_data, window_label)

print_summary_table()