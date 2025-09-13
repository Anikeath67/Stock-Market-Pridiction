import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def preprocess_data(df):
    # Handle MultiIndex columns (flatten them if needed)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ['_'.join([str(c) for c in col if c]).strip() for col in df.columns.values]
    else:
        df.columns = df.columns.str.strip()

    # Normalize column names (crypto often has lowercase)
    df.columns = [c.capitalize() for c in df.columns]

    # Drop missing values
    df = df.dropna()

    print("📌 Available columns after cleaning:", df.columns.tolist())

    # Automatically detect feature columns
    possible_features = ["Open", "High", "Low", "Volume"]
    features = [col for col in df.columns if any(f in col for f in possible_features)]
    target_candidates = [col for col in df.columns if "Close" in col]

    if not features or not target_candidates:
        raise ValueError("❌ Could not detect features or target column in dataset.")

    target = target_candidates[0]

    print(f"✅ Selected features: {features}")
    print(f"✅ Target column: {target}")

    # Features and target
    X = df[features]
    y = df[target]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Feature scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test


# -------------------- TEST BLOCK --------------------
if __name__ == "__main__":
    from load_data import load_stock_dataset, load_crypto_dataset

    print("🔹 Testing Stock Preprocessing...")
    df_stock = load_stock_dataset("AAPL", period="1mo", interval="1d")
    X_train, X_test, y_train, y_test = preprocess_data(df_stock)
    print("X_train shape:", X_train.shape)
    print("y_train shape:", y_train.shape)

    print("\n🔹 Testing Crypto Preprocessing...")
    df_crypto = load_crypto_dataset("BTC/USDT", timeframe="1h", limit=20)
    X_train, X_test, y_train, y_test = preprocess_data(df_crypto)
    print("X_train shape:", X_train.shape)
    print("y_train shape:", y_train.shape)
