import mlflow
import pytest

from ai4i_mlops import tracking


@pytest.fixture
def local_store(tmp_path, monkeypatch):
    previous_uri = mlflow.get_tracking_uri()
    monkeypatch.setattr(tracking, "MLFLOW_TRACKING_URI", f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}")
    monkeypatch.setattr(tracking, "MLFLOW_ARTIFACT_ROOT", tmp_path / "mlruns")
    yield tmp_path
    mlflow.set_tracking_uri(previous_uri)


def test_experiment_is_created_once_under_artifact_root(local_store):
    first = tracking.get_or_create_experiment_id()
    second = tracking.get_or_create_experiment_id()
    assert first == second

    experiment = mlflow.get_experiment(first)
    assert experiment.name == tracking.EXPERIMENT_NAME
    assert experiment.artifact_location == (local_store / "mlruns").as_uri()
