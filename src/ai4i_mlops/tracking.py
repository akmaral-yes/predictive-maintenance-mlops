import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.pipeline import Pipeline

from ai4i_mlops.config import (
    EXPERIMENT_NAME,
    MLFLOW_ARTIFACT_ROOT,
    MLFLOW_TRACKING_URI,
    MODEL_NAME,
)


def configure_tracking() -> None:
    """Point MLflow at the repo-local SQLite store."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)


def get_or_create_experiment_id() -> str:
    """Id of the project experiment, creating it with a repo-rooted artifact location if needed."""
    configure_tracking()
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is not None:
        return experiment.experiment_id
    return mlflow.create_experiment(EXPERIMENT_NAME, artifact_location=MLFLOW_ARTIFACT_ROOT.as_uri())


def log_fitted_pipeline(model: Pipeline, X_train: pd.DataFrame) -> None:
    """Log a fitted pipeline in the active run under MODEL_NAME, with an input example for the schema."""
    # skops refuses tree internals by default; this model is produced here, so trust only that type.
    mlflow.sklearn.log_model(
        model,
        name=MODEL_NAME,
        input_example=X_train.head(5),
        skops_trusted_types=["sklearn.tree._tree.Tree"],
    )
