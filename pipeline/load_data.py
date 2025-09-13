import pandas as pd
import yfinance as yf
import ccxt

def load_stock_dataset(ticker="AAPL", period="6mo", interval="1d"):
    """
    Fetch stock market data from Yahoo Finance
    ticker: stock symbol (e.g., 'AAPL', 'MSFT', 'TSLA', '^NSEI')
    period: data length ('1mo','3mo','6mo','1y','5y','max')
    interval: time gap ('1d','1h','15m')
    """
    df = yf.download(ticker, period=period, interval=interval)
    df.reset_index(inplace=True)  # reset index to get Date column
    print(f"Stock data for {ticker} fetched successfully!")
    return df


def load_crypto_dataset(symbol="BTC/USDT", timeframe="1h", limit=100):
    """
    Fetch crypto market data using CCXT (Binance by default)
    symbol: trading pair (e.g., 'BTC/USDT', 'ETH/USDT')
    timeframe: e.g., '1m','5m','1h','1d'
    limit: number of candles
    """
    exchange = ccxt.binance()
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    print(f"Crypto data for {symbol} fetched successfully!")
    return df

    # 🔽 Test Block
if __name__ == "__main__":
    # Test stock loader
    print("Testing Stock Data Loader...")
    df_stock = load_stock_dataset("AAPL", period="1mo", interval="1d")
    print(df_stock.head())      # show first 5 rows
    print(df_stock.info())      # show columns & datatypes

    # Test crypto loader
    print("\nTesting Crypto Data Loader...")
    df_crypto = load_crypto_dataset("BTC/USDT", timeframe="1h", limit=5)
    print(df_crypto.head())     # show first 5 rows
    print(df_crypto.info())     # show columns & datatypes
    