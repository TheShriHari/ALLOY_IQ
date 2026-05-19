import mlflow
import mlflow.sklearn
from pathlib import Path

TRACKING_URI = "sqlite:///mlruns/mlflow.db"
EXPERIMENT_NAME = "alloyiq_property_prediction"

def setup_mlflow():
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

def log_training_run(params: dict, metrics: dict, model, artifacts: list[str]):
    """
    Log a complete training run with params, metrics, model binary, and artifact files.
    Call this at the end of every training session.
    """
    with mlflow.start_run():
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)    # {"ys_r2": 0.91, "hv_rmse": 38.2, ...}
        mlflow.sklearn.log_model(model, "stacking_ensemble")
        for artifact_path in artifacts:
            mlflow.log_artifact(artifact_path)
        run_id = mlflow.active_run().info.run_id
        print(f"MLflow run logged: {run_id}")
        return run_id
