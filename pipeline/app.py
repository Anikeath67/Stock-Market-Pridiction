from flask import Flask, request, jsonify
from flask_cors import CORS
from pipeline.load_data import load_stock_dataset, load_crypto_dataset
from pipeline.preprocess import preprocess_data
from pipeline.train_model import train_model

app = Flask(__name__)
CORS(app)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    dtype = data.get('type')  # 'stock' or 'crypto'

    try:
        if dtype == 'stock':
            ticker = data.get('ticker', 'AAPL')
            period = data.get('period', '6mo')
            interval = data.get('interval', '1d')
            df = load_stock_dataset(ticker, period, interval)

        elif dtype == 'crypto':
            symbol = data.get('symbol', 'BTC/USDT')
            timeframe = data.get('timeframe', '1h')
            limit = data.get('limit', 100)
            df = load_crypto_dataset(symbol, limit=limit, timeframe=timeframe)

        else:
            return jsonify({"error": "Invalid type"}), 400

        X_train, X_test, y_train, y_test = preprocess_data(df)
        model = train_model(X_train, y_train)
        y_pred = model.predict(X_test)

        return jsonify({
            "actual": y_test.tolist(),
            "predicted": y_pred.tolist()
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
