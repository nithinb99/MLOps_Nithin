import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data():
    """
    Load the Wine dataset and return the features and target values.
    """
    wine = load_breast_cancer()
    X = wine.data
    y = wine.target
    return X, y

def split_data(X, y):
    """
    """
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=12)
    return X_train, X_test, y_train, y_test