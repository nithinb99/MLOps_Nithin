import joblib

def predict_data(X):
    """
    """
    model = joblib.load("../models/breast_cancer.pkl")
    y_pred = model.predict(X)
    return y_pred