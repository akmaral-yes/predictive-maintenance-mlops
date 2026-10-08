import mlflow
import numpy as np
import pandas as pd
import pytest

from ai4i_mlops import registry, retraining, tracking
from ai4i_mlops.data import clean, split
from ai4i_mlops.feedback import candidates_from_split, review, write_feedback
from ai4i_mlops.retraining import combine_with_feedback


def _frame(index: list[int], label: int = 0) -> tuple[pd.DataFrame, pd.Series]:
    X = pd.DataFrame(
        {
            "Type": ["L"] * len(index),
            "Air temperature": 300.0,
            "Process temperature": 310.0,
            "Rotational speed": 1500,
            "Torque": 40.0,
            "Tool wear": 10,
        },
        index=index,
    )
    return X, pd.Series(label, index=index, name="Machine failure")


TRAIN = _frame([0, 1, 2, 3])
FUTURE_INDEX = pd.Index([10, 11, 12])
TEST_INDEX = pd.Index([20, 21])


def _combine(feedback_index: list[int]):
    return combine_with_feedback(*TRAIN, *_frame(feedback_index, label=1), FUTURE_INDEX, TEST_INDEX)


def test_a_b_c_d_train_and_feedback_are_combined_with_source_indices():
    X, y = _combine([10, 12])
    assert list(X.index) == list(y.index) == [0, 1, 2, 3, 10, 12]
    assert X.index.is_unique
    assert len(X) == len(TRAIN[0]) + 2
    assert list(y.loc[[10, 12]]) == [1, 1]


def test_e_feedback_overlapping_train_is_rejected():
    with pytest.raises(ValueError, match="overlaps the train split.*: 2"):
        _combine([10, 2])


def test_f_feedback_overlapping_test_is_rejected():
    with pytest.raises(ValueError, match="overlaps the fixed test split.*: 21"):
        _combine([10, 21])


def test_g_feedback_outside_future_is_rejected():
    with pytest.raises(ValueError, match="not from the future split.*: 99"):
        _combine([10, 99])


@pytest.fixture
def validated_path(tmp_path, monkeypatch):
    path = tmp_path / "validated_feedback.jsonl"
    monkeypatch.setattr(retraining, "VALIDATED_FEEDBACK_PATH", path)
    return path


def test_h_missing_validated_file_refuses_to_retrain(validated_path, capsys):
    assert retraining.main([]) == 2
    assert "refusing to retrain" in capsys.readouterr().out


def test_i_empty_validated_file_refuses_to_retrain(validated_path, capsys):
    validated_path.write_text("")
    assert retraining.main([]) == 2
    assert "no records" in capsys.readouterr().out


def _tiny_raw(n: int = 300) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "UID": range(1, n + 1),
            "Product ID": [f"P{i}" for i in range(n)],
            "Type": np.resize(["L", "M", "H"], n),
            "Air temperature": rng.normal(300, 2, n),
            "Process temperature": rng.normal(310, 1.5, n),
            "Rotational speed": rng.integers(1200, 2800, n),
            "Torque": rng.normal(40, 10, n),
            "Tool wear": rng.integers(0, 250, n),
            "Machine failure": (np.arange(n) % 5 == 0).astype(int),
            **{mode: 0 for mode in ["TWF", "HDF", "PWF", "OSF", "RNF"]},
        }
    )


def test_retraining_never_touches_the_registry(tmp_path, monkeypatch, validated_path, capsys):
    """One MLflow run in a throwaway store; any registry mutation fails the test."""
    raw = _tiny_raw()
    monkeypatch.setattr(retraining, "load_raw", lambda: raw)
    monkeypatch.setattr(tracking, "MLFLOW_TRACKING_URI", f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}")
    monkeypatch.setattr(tracking, "MLFLOW_ARTIFACT_ROOT", tmp_path / "mlruns")

    def forbidden(*args, **kwargs):
        raise AssertionError("retraining must not mutate the registry")

    for name in ("register_run_model", "set_alias", "promote_candidate", "rollback_champion"):
        monkeypatch.setattr(registry, name, forbidden)
    monkeypatch.setattr("mlflow.register_model", forbidden)
    monkeypatch.setattr("mlflow.MlflowClient.set_registered_model_alias", forbidden)
    monkeypatch.setattr("mlflow.MlflowClient.create_model_version", forbidden)

    X_future, y_future = split(*clean(raw))["future"]
    approved, _ = review(candidates_from_split(X_future, y_future), [])
    write_feedback(validated_path, approved)

    previous_uri = mlflow.get_tracking_uri()
    try:
        assert retraining.main([]) == 0
        client = mlflow.MlflowClient()
        assert client.search_registered_models() == []
        runs = client.search_runs([tracking.get_or_create_experiment_id()])
        assert len(runs) == 1
        assert runs[0].data.params["n_total_train_rows"] == str(180 + len(approved))
    finally:
        mlflow.set_tracking_uri(previous_uri)
    assert "evaluation set = unchanged fixed test split" in capsys.readouterr().out
