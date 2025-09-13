import joblib

def save_model(model, filename="stock_model.pkl"):
    joblib.dump(model, filename)
    print(f"Model saved as {filename}")
