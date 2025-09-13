import matplotlib.pyplot as plt

def plot_results(y_test, y_pred):
    plt.figure(figsize=(10, 5))
    plt.plot(y_test.values, label="Actual Price", color="blue")
    plt.plot(y_pred, label="Predicted Price", color="red")
    plt.title("Stock Price Prediction")
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.legend()
    plt.show()
