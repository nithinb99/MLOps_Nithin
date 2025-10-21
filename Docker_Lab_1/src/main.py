from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import joblib

if __name__ == '__main__':
    wine = load_breast_cancer()
    X, y = wine.data, wine.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = SVC(kernel='rbf', random_state=42)
    model.fit(X_train, y_train)

    joblib.dump(model, 'cancer_model.pkl')
    
    print("The model training was successful")