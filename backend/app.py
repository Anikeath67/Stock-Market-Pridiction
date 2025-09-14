import os
from flask import Flask, request, jsonify
from flask_cors import CORS

# Import your pipeline modules
from pipeline.load_data import load_stock_dataset, load_crypto_dataset
from pipeline.preprocess import preprocess_data
from pipeline.train_model import train_model

app = Flask(__name__)
CORS(app)

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "✅ Stock & Crypto Prediction API is running!"})

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    dtype = data.get("type")  # 'stock' or 'crypto'

    try:
        # Load data
        if dtype == "stock":
            ticker = data.get("ticker", "AAPL")
            period = data.get("period", "6mo")
            interval = data.get("interval", "1d")
            df = load_stock_dataset(ticker, period, interval)

        elif dtype == "crypto":
            symbol = data.get("symbol", "BTC/USDT")
            timeframe = data.get("timeframe", "1h")
            limit = data.get("limit", 100)
            df = load_crypto_dataset(symbol, limit=limit, timeframe=timeframe)

        else:
            return jsonify({"error": "Invalid type (use 'stock' or 'crypto')"}), 400

        # Preprocess + Train
        X_train, X_test, y_train, y_test = preprocess_data(df)
        model = train_model(X_train, y_train)
        y_pred = model.predict(X_test)

        return jsonify({
            "actual": y_test.tolist(),
            "predicted": y_pred.tolist()
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Render will call this with gunicorn (no need for debug=True)
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)