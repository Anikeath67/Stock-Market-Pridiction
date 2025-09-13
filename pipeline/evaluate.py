from sklearn.metrics import mean_squared_error, r2_score

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print("Evaluation Results:")
    print("Mean Squared Error:", mse)
    print("R2 Score:", r2)
    return y_pred
