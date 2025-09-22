import os
from datetime import datetime, timedelta

from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator

# ---- Light, cheap constants (no I/O) ----
AIRFLOW_HOME = os.getenv("AIRFLOW_HOME", ".")
DATA_DIR = os.path.join(AIRFLOW_HOME, "data")
ARTIFACTS_DIR = os.path.join(AIRFLOW_HOME, "artifacts")


# ---- Task callables with lazy imports (heavy stuff inside) ----
def _load_data_callable(file_name):
    # Import only when the task runs
    from Lab_1_Airflow import load_data
    return load_data(file_name)


def _preprocess_callable(serialized_df):
    from Lab_1_Airflow import data_preprocessing
    return data_preprocessing(serialized_df)


def _train_gmm_callable(serialized_data):
    from Lab_1_Airflow import build_save_model
    return build_save_model(serialized_data, "gmm_model.pkl")


def _plot_bic_scores_callable(serialized_bic_and_clusters, ti=None):
    import os, pickle
    # import matplotlib only when needed
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    bic_scores, best_n = pickle.loads(serialized_bic_and_clusters)
    output_path = os.path.join(ARTIFACTS_DIR, "bic_scores.png")

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(bic_scores) + 1), bic_scores, marker="o", label="BIC")
    plt.axvline(best_n, color="r", linestyle="--", label=f"Optimal: {best_n}")
    plt.xlabel("Number of Clusters")
    plt.ylabel("BIC (lower is better)")
    plt.title("GMM Model Selection via BIC")
    plt.legend()
    plt.savefig(output_path)
    plt.close()

    ti.xcom_push(key="bic_plot_path", value=f"file://{output_path}")
    ti.xcom_push(key="best_n", value=best_n)


def _save_predictions_callable(_, ti=None):
    import os, pickle
    import pandas as pd
    from Lab_1_Airflow import load_model_predict

    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    # get predictions + probabilities from the saved model
    serialized_results = load_model_predict("gmm_model.pkl", "test.csv")
    results = pickle.loads(serialized_results)
    preds = results["predictions"]
    probs = results["probabilities"]

    # build output CSV alongside original test.csv fields
    df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
    prob_df = pd.DataFrame(
        probs, columns=[f"cluster_prob_{i}" for i in range(len(probs[0]))]
    )
    df_out = pd.concat(
        [df.reset_index(drop=True), pd.Series(preds, name="pred_cluster"), prob_df],
        axis=1,
    )

    output_path = os.path.join(ARTIFACTS_DIR, "test_predictions.csv")
    df_out.to_csv(output_path, index=False)

    ti.xcom_push(key="predictions_path", value=f"file://{output_path}")
    ti.xcom_push(
        key="cluster_distribution", value=df_out["pred_cluster"].value_counts().to_dict()
    )


# ---- Default Args ----
default_args = {
    "owner": "admin",
    "start_date": datetime(2025, 1, 15),
    "retries": 0,
    "retry_delay": timedelta(minutes=5),
}

dag = DAG(
    "Airflow_GMM_Lab1",
    default_args=default_args,
    description="Airflow DAG for GMM clustering with BIC model selection",
    schedule=None,
    catchup=False,
)

# ---- Tasks ----
load_data_task = PythonOperator(
    task_id="load_data_task",
    python_callable=_load_data_callable,
    op_args=["file.csv"],
    dag=dag,
)

data_preprocessing_task = PythonOperator(
    task_id="data_preprocessing_task",
    python_callable=_preprocess_callable,
    op_args=[load_data_task.output],
    dag=dag,
)

build_save_model_task = PythonOperator(
    task_id="build_save_model_task",
    python_callable=_train_gmm_callable,
    op_args=[data_preprocessing_task.output],
    dag=dag,
)

plot_bic_scores_task = PythonOperator(
    task_id="plot_bic_scores_task",
    python_callable=_plot_bic_scores_callable,
    op_args=[build_save_model_task.output],
    dag=dag,
)

save_predictions_task = PythonOperator(
    task_id="save_predictions_task",
    python_callable=_save_predictions_callable,
    op_args=[build_save_model_task.output],  # dummy dep to ensure ordering
    dag=dag,
)

# ---- Dependencies ----
load_data_task >> data_preprocessing_task >> build_save_model_task >> plot_bic_scores_task >> save_predictions_task