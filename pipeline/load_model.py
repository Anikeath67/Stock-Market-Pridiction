import joblib

def load_model(filename="stock_model.pkl"):
    model = joblib.load(filename)
    print(f"Model loaded from {filename}")
    return model
