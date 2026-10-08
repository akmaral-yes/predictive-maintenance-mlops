import mlflow

from ai4i_mlops.config import EXPERIMENT_NAME, MLFLOW_ARTIFACT_ROOT, MLFLOW_TRACKING_URI


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
