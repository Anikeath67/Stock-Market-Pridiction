import pandas as pd
import yfinance as yf

def load_dataset(file_path):
    """Load dataset from CSV file"""
    df = pd.read_csv(file_path)
    print("Dataset loaded from CSV")
    print(df.head())
    return df

def load_dataset_live(ticker="AAPL", period="6mo", interval="1d"):
    """
    Fetch live stock market data from Yahoo Finance
    ticker: stock symbol (e.g., 'AAPL', 'MSFT', 'TSLA', '^NSEI')
    period: data length ('1mo','3mo','6mo','1y','5y','max')
    interval: time gap ('1d','1h','15m')
    """
    df = yf.download(ticker, period=period, interval=interval)
    df.reset_index(inplace=True)  # reset index to get Date column
    print(f"Live data for {ticker} fetched successfully!")
    print(df.head())
    return df
