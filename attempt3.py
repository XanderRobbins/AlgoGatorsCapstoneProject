import yfinance as yf
from datetime import timedelta
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel

def analyze_event(df_stock, df_sp500, event_date, window=30):
    stock = df_stock.copy()
    sp500 = df_sp500.copy()

    stock['Date'] = pd.to_datetime(stock['Date'])
    sp500['Date'] = pd.to_datetime(sp500['Date'])

    stock['Return'] = stock['Close'].pct_change()
    sp500['Return'] = sp500['Close'].pct_change()

    before_stock = stock[stock['Date'] < event_date].tail(window)
    after_stock  = stock[stock['Date'] > event_date].head(window)
    before_sp500 = sp500[sp500['Date'] < event_date].tail(window)
    after_sp500  = sp500[sp500['Date'] > event_date].head(window)

    if len(before_stock) < window or len(after_stock) < window or \
       len(before_sp500) < window or len(after_sp500) < window:
        print(f"Warning: insufficient data around {event_date}")
        return None

    vol_before = before_stock['Return'].std()
    vol_after  = after_stock['Return'].std()

    agr_before = (before_stock['Close'].iloc[-1] / before_stock['Close'].iloc[0]) - 1
    agr_after  = (after_stock['Close'].iloc[-1]  / after_stock['Close'].iloc[0])  - 1

    ret_sp500_before = (before_sp500['Close'].iloc[-1] / before_sp500['Close'].iloc[0]) - 1
    ret_sp500_after  = (after_sp500['Close'].iloc[-1]  / after_sp500['Close'].iloc[0])  - 1

    rel_before = float(agr_before) - float(ret_sp500_before)
    rel_after  = float(agr_after)  - float(ret_sp500_after)

    return {
        'Volatility Before':      vol_before,
        'Volatility After':       vol_after,
        'AGR Before':             float(agr_before),
        'AGR After':              float(agr_after),
        'Relative Return Before': rel_before,
        'Relative Return After':  rel_after,
        'Before Window':          before_stock,
        'After Window':           after_stock,
    }

def batch_analyze(events, stock_data, sp500_data, window=30):
    results = {}
    for ticker, date in events.items():
        print(f"Processing {ticker} for {date}")
        res = analyze_event(stock_data[ticker], sp500_data, date, window)
        if res is not None:
            results[ticker] = res
        else:
            print(f"Skipping {ticker}")
    print(f"Total analyzed: {len(results)}")
    return results

def perform_volatility_tests(results):
    b = [r['Volatility Before'] for r in results.values()]
    a = [r['Volatility After']  for r in results.values()]
    t, p = ttest_rel(b, a, nan_policy='omit')
    print(f"Paired t-test: t={t:.3f}, p={p:.3f}")
    return t, p

def plot_price(before_df, after_df, ticker, event_date):
    plt.figure(figsize=(10, 5))
    plt.plot(before_df['Date'], before_df['Close'], label='Before')
    plt.plot(after_df['Date'],  after_df['Close'],  label='After')
    plt.axvline(event_date, color='red', linestyle='--', label='Event')
    plt.title(f"{ticker} Price Around {event_date.date()}")
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_volatility(vol_before, vol_after, ticker):
    plt.figure(figsize=(6, 4))
    plt.bar(['Before', 'After'], [vol_before, vol_after])
    plt.title(f"{ticker} Volatility")
    plt.ylabel('Std Dev of Returns')
    plt.grid(True)
    plt.show()

def plot_relative_returns(res, ticker):
    rels = [res['Relative Return Before'], res['Relative Return After']]
    plt.figure(figsize=(6, 4))
    plt.bar(['Before', 'After'], rels)
    plt.title(f"{ticker} Relative Return vs S&P500")
    plt.axhline(0, color='black', linestyle='--')
    plt.ylabel('Relative Return')
    plt.grid(True)
    plt.show()

def create_summary_table(results):
    df = pd.DataFrame(results).T
    return df.drop(columns=['Before Window', 'After Window'])

def main():
    events = {
        'T': '2018-06-14'  # AT&T - Time Warner merger close
    }
    events = {ticker: pd.to_datetime(date) for ticker, date in events.items()}

    window = 30
    start_date = min(events.values()) - timedelta(days=window * 2)
    end_date   = max(events.values()) + timedelta(days=window * 2)

    stock_data = {}
    tickers = list(events.keys())

    try:
        for ticker in tickers:
            df = yf.download(ticker, start=start_date, end=end_date)
            df.reset_index(inplace=True)
            stock_data[ticker] = df

        sp500_data = yf.download('^GSPC', start=start_date, end=end_date)
        sp500_data.reset_index(inplace=True)

        results = batch_analyze(events, stock_data, sp500_data, window=window)
        print(f"Results: {list(results.keys())}")

        summary = create_summary_table(results)
        print(summary)

        perform_volatility_tests(results)

        print("Generating plots...")
        for ticker in tickers:
            if ticker in results:
                res = results[ticker]
                plot_price(res['Before Window'], res['After Window'], ticker, events[ticker])
                plot_volatility(res['Volatility Before'], res['Volatility After'], ticker)
                plot_relative_returns(res, ticker)

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
