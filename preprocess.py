import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def preprocess_data(df):
    # Handle MultiIndex columns (flatten them)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ['_'.join([str(c) for c in col if c]).strip() for col in df.columns.values]
    else:
        df.columns = df.columns.str.strip()

    # Drop missing values
    df = df.dropna()

    print("📌 Available columns after cleaning:", df.columns.tolist())

    # Automatically detect feature columns
    possible_features = ["Open", "High", "Low", "Volume"]
    features = [col for col in df.columns if any(f in col for f in possible_features)]
    target = [col for col in df.columns if "Close" in col][0]  # pick first close column

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
