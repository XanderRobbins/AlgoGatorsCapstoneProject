# AlgoGators Capstone Project

Event study analysis on corporate events and their impact on stock returns and volatility. We measure abnormal returns (stock return minus S&P 500 return) around specific event dates across a set of major US equities, then run t-tests to check whether the post-event returns are statistically significant.

## What it does

- Downloads historical OHLCV data via yfinance
- Computes daily returns, abnormal returns, and cumulative abnormal returns (CAR) relative to the S&P 500
- Runs one-sample t-tests on post-event abnormal returns
- Compares pre/post event volatility and average trading volume
- Generates time series plots for CAR, daily returns, and rolling volatility

## Files

- `statistics.py` - Single stock event study (Tesla, April 2021). Good starting point.
- `attempt3.py` - Batch event framework with price, volatility, and relative return plots
- `MassStatistics.py` - Full run across 80+ events using three time windows (1 week, 1 month, 1 year), outputs a summary table

## Setup

```
pip install yfinance pandas matplotlib scipy tabulate
```

## Usage

```
python statistics.py
python attempt3.py
python MassStatistics.py
```

`MassStatistics.py` fetches data for 80+ events across three windows so it takes a while to run. The summary table prints at the end.
