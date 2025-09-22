import os
import pickle

# Keep module import cheap: only small constants here
AIRFLOW_HOME = os.getenv("AIRFLOW_HOME", os.path.dirname(__file__))
DATA_DIR = os.path.join(AIRFLOW_HOME, "data")
MODEL_DIR = os.path.join(AIRFLOW_HOME, "model")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")

FEATURES = ["BALANCE", "PURCHASES", "CREDIT_LIMIT"]


def load_data(file_name="file.csv", chunksize=50000):
    """
    Loads CSV in chunks and concatenates them safely into a DataFrame.
    Use chunksize to control memory footprint.
    """
    import pandas as pd
    file_path = os.path.join(DATA_DIR, file_name)

    chunks = []
    for chunk in pd.read_csv(file_path, chunksize=chunksize):
        chunks.append(chunk)

    df = pd.concat(chunks, ignore_index=True)
    return pickle.dumps(df)


def data_preprocessing(serialized_data):
    """
    Preprocess: drop NA, scale FEATURES (MinMax), persist scaler.
    Returns serialized numpy array.
    """
    import pandas as pd
    from sklearn.preprocessing import MinMaxScaler

    df = pickle.loads(serialized_data)
    df = df.dropna()

    X = df[FEATURES]
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # Save scaler when the task runs (not at import)
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)

    return pickle.dumps(X_scaled)


def build_save_model(serialized_data, filename="gmm_model.pkl"):
    """
    Train GMM for k=1..10, compute BIC, save the best model.
    Returns serialized (bic_scores, best_n).
    """
    from sklearn.mixture import GaussianMixture

    X = pickle.loads(serialized_data)

    bic_scores = []
    best_model = None
    lowest_bic = float("inf")
    best_n = None

    for k in range(1, 11):
        gmm = GaussianMixture(n_components=k, covariance_type="full", random_state=42)
        gmm.fit(X)
        bic = gmm.bic(X)
        bic_scores.append(bic)
        if bic < lowest_bic:
            lowest_bic = bic
            best_n = k
            best_model = gmm

    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, filename)
    with open(model_path, "wb") as f:
        pickle.dump(best_model, f)

    return pickle.dumps((bic_scores, best_n))


def load_model_predict(filename="gmm_model.pkl", test_file="test.csv"):
    """
    Load best GMM + saved scaler, transform test FEATURES, predict clusters.
    Returns serialized dict with predictions & probabilities (lists).
    """
    import pandas as pd

    # Load model & scaler
    model_path = os.path.join(MODEL_DIR, filename)
    with open(model_path, "rb") as f:
        gmm = pickle.load(f)

    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)

    # Load & scale test data consistently with training
    test_path = os.path.join(DATA_DIR, test_file)
    df_test = pd.read_csv(test_path)
    X_test = scaler.transform(df_test[FEATURES])

    preds = gmm.predict(X_test)
    probs = gmm.predict_proba(X_test)

    return pickle.dumps(
        {
            "predictions": preds.tolist(),
            "probabilities": probs.tolist(),
        }
    )