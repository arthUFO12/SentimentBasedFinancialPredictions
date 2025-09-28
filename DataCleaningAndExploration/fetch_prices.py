import yfinance as yf

tickers = ["AAPL", "TSLA","GOOG", "NVDA"]

data = yf.download(tickers, start="2020-01-01", end="2025-01-01", interval='1d')
data.to_csv('../data/csv/stock_prices.csv', index=True)