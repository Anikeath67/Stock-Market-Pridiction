
import sys
import yfinance as yf
import ccxt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import joblib


# =========================
# 1. Load Data
# =========================
def load_stock_dataset(ticker="AAPL", period="6mo", interval="1d"):
    print(f"📌 Downloading stock data for {ticker} ({period}, {interval})...")
    df = yf.download(ticker, period=period, interval=interval)
    df.reset_index(inplace=True)
    if df.empty:
        raise ValueError(f"No data downloaded. Check period/interval for ticker {ticker}")
    print(f"✅ Downloaded {df.shape[0]} rows of stock data.\n")
    return df



def load_crypto_dataset(symbol="BTC/USDT", limit=500, timeframe="1h"):
    print(f"📌 Fetching live crypto data for {symbol} ({timeframe})...")
    exchange = ccxt.binance()
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=["Timestamp", "Open", "High", "Low", "Close", "Volume"])
    df["Date"] = pd.to_datetime(df["Timestamp"], unit="ms")
    df.drop("Timestamp", axis=1, inplace=True)
    print(f"✅ Downloaded {df.shape[0]} rows of crypto data.\n")
    return df


# =========================
# 2. Preprocess Data
# =========================
def preprocess_data(df, target_col="Close"):
    print("📌 Preprocessing data...")
    df = df.dropna()

    X = df[["Open", "High", "Low", "Volume"]]
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print(f"✅ Data preprocessed. Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}\n")
    return X_train, X_test, y_train, y_test


# =========================
# 3. Train Model
# =========================
def train_model(X_train, y_train):
    print("📌 Training model...")
    model = LinearRegression()
    model.fit(X_train, y_train)
    print("✅ Model trained.\n")
    return model


# =========================
# 4. Evaluate Model
# =========================
def evaluate_model(model, X_test, y_test):
    print("📌 Evaluating model...")
    y_pred = model.predict(X_test)

    plt.figure(figsize=(10, 6))
    plt.plot(y_test.values, label="Actual Price", color="blue")
    plt.plot(y_pred, label="Predicted Price", color="orange")
    plt.title("Price Prediction (Stock/Crypto)")
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.legend()
    plt.show()

    print("✅ Evaluation complete.\n")
    return y_pred


# =========================
# 5. Save/Load Model
# =========================
def save_model(model, filename):
    joblib.dump(model, filename)
    print(f"✅ Model saved as {filename}\n")


def load_model(filename):
    model = joblib.load(filename)
    print(f"✅ Model loaded from {filename}\n")
    return model


# =========================
# Main Program
# =========================
def main():
    print("Choose Prediction Type:")
    print("1. Stock Market")
    print("2. Cryptocurrency")
    choice = input("Enter choice (1/2): ").strip()

    if choice == "1":
        ticker = input("Enter stock ticker (e.g., AAPL, MSFT): ").strip() or "AAPL"
        period = input("Enter period (1mo,3mo,6mo,1y,5y,max): ").strip() or "6mo"
        interval = input("Enter interval (1d,1h,15m): ").strip() or "1d"
        df = load_stock_dataset(ticker, period=period, interval=interval)

    elif choice == "2":
        symbol = input("Enter crypto symbol (e.g., BTC/USDT, ETH/USDT): ").strip() or "BTC/USDT"
        timeframe = input("Enter timeframe (1m,5m,15m,1h,4h,1d): ").strip() or "1h"
        limit = int(input("Enter number of data points: ").strip() or 500)
        df = load_crypto_dataset(symbol, limit=limit, timeframe=timeframe)

    else:
        print("❌ Invalid choice")
        sys.exit(1)

    # Preprocess
    X_train, X_test, y_train, y_test = preprocess_data(df)

    # Train
    model = train_model(X_train, y_train)

    # Save
    save_model(model, "price_model.pkl")

    # Load
    loaded_model = load_model("price_model.pkl")

    # Evaluate
    evaluate_model(loaded_model, X_test, y_test)



if __name__ == "__main__":
    main()
