import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

from ai4i_mlops import api, tracking
from ai4i_mlops.api import PredictionRequest, ServingModel, to_frame
from ai4i_mlops.config import PREDICTION_THRESHOLD
from ai4i_mlops.data import CATEGORICAL, NUMERIC
from ai4i_mlops.registry import RegistryError

VALID = {
    "machine_type": "L",
    "air_temperature": 298.1,
    "process_temperature": 308.6,
    "rotational_speed": 1551,
    "torque": 42.8,
    "tool_wear": 0,
}


class FakeModel:
    """Stands in for the sklearn pipeline: a fixed failure probability for every row."""

    classes_ = np.array([0, 1])

    def __init__(self, probability: float = 0.2, fail: bool = False) -> None:
        self.probability = probability
        self.fail = fail

    def predict_proba(self, X):
        if self.fail:
            raise RuntimeError("boom")
        return np.array([[1 - self.probability, self.probability]] * len(X))


@pytest.fixture
def serve(monkeypatch):
    """Start the app (lifespan included) with a fake model; counts how often the loader runs."""
    calls = []

    def start(model: FakeModel | None = None, error: Exception | None = None) -> TestClient:
        def loader() -> ServingModel:
            calls.append(1)
            if error is not None:
                raise error
            return ServingModel(model or FakeModel(), "7", "run-abc")

        monkeypatch.setattr(api, "load_serving_model", loader)
        return TestClient(api.app)

    start.calls = calls
    return start


def _metrics(client: TestClient) -> dict:
    return client.get("/metrics").json()


def test_a_request_becomes_exact_model_frame():
    frame = to_frame(PredictionRequest(**VALID))
    assert list(frame.columns) == [*CATEGORICAL, *NUMERIC]
    assert frame.iloc[0].tolist() == ["L", 298.1, 308.6, 1551, 42.8, 0]
    assert len(frame) == 1


@pytest.mark.parametrize(
    ("probability", "predicted"),
    [(0.2, False), (PREDICTION_THRESHOLD, True), (0.9, True)],  # C below, D exactly at, E above threshold
)
def test_b_c_d_e_prediction_uses_threshold(serve, probability, predicted):
    with serve(FakeModel(probability)) as client:
        response = client.post("/predict", json=VALID)
    assert response.status_code == 200
    body = response.json()
    assert body == {
        "failure_probability": pytest.approx(probability),
        "predicted_failure": predicted,
        "threshold": PREDICTION_THRESHOLD,
        "model_version": "7",
    }


@pytest.mark.parametrize(
    "payload",
    [
        {**VALID, "machine_type": "X"},  # F invalid Type
        {k: v for k, v in VALID.items() if k != "torque"},  # G missing field
        {**VALID, "TWF": 1},  # H extra field
        {**VALID, "rotational_speed": -1},  # J
        {**VALID, "torque": -0.1},  # K
        {**VALID, "tool_wear": -5},  # L
    ],
)
def test_f_to_l_invalid_requests_are_422(serve, payload):
    with serve() as client:
        assert client.post("/predict", json=payload).status_code == 422


@pytest.mark.parametrize("value", ["NaN", "Infinity", "-Infinity"])
def test_i_nan_and_inf_are_rejected(serve, value):
    body = json.dumps(VALID).replace("298.1", value)  # Python's JSON parser accepts these literals
    with serve() as client:
        response = client.post("/predict", content=body, headers={"content-type": "application/json"})
    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"] == ["body", "air_temperature"]


def test_m_unavailable_model_is_503_everywhere(serve):
    with serve(error=RegistryError("No champion alias")) as client:
        health = client.get("/health")
        assert health.status_code == 503
        assert health.json() == {
            "status": "unavailable",
            "model_loaded": False,
            "reason": "model could not be loaded (RegistryError)",
        }
        assert client.post("/predict", json=VALID).status_code == 503


def test_n_healthy_model(serve):
    with serve() as client:
        response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_loaded": True, "model_version": "7", "source_run_id": "run-abc"}


def test_o_model_loads_once(serve):
    with serve() as client:
        for _ in range(3):
            client.post("/predict", json=VALID)
        client.get("/health")
    assert len(serve.calls) == 1


def test_p_t_successful_predictions_count_and_health_metrics_do_not(serve):
    with serve() as client:
        client.post("/predict", json=VALID)
        client.post("/predict", json=VALID)
        client.get("/health")
        client.get("/metrics")
        m = _metrics(client)
    assert m["prediction_request_count"] == 2
    assert m["prediction_error_count"] == 0
    assert m["average_prediction_latency_ms"] >= 0  # V


def test_q_u_inference_failure_counts_as_error(serve):
    with serve(FakeModel(fail=True)) as client:
        response = client.post("/predict", json=VALID)
        assert response.status_code == 500
        assert response.json() == {"detail": "inference failed"}  # no traceback leaked
        client.post("/predict", json=VALID)
        m = _metrics(client)
    assert m["prediction_request_count"] == 2
    assert m["prediction_error_count"] == 2
    assert m["prediction_error_rate"] == 1.0
    assert m["average_prediction_latency_ms"] >= 0


def test_u_error_rate_is_errors_over_requests(serve):
    with serve() as client:
        assert _metrics(client)["prediction_error_rate"] == 0.0  # no requests yet
    model = FakeModel()
    with serve(model) as client:
        for fail in (False, True, False, False):
            model.fail = fail
            client.post("/predict", json=VALID)
        m = _metrics(client)
    assert (m["prediction_request_count"], m["prediction_error_count"]) == (4, 1)
    assert m["prediction_error_rate"] == 0.25


def test_r_invalid_request_counts_only_as_invalid(serve):
    with serve() as client:
        client.post("/predict", json={**VALID, "machine_type": "X"})
        m = _metrics(client)
    assert m["invalid_request_count"] == 1
    assert m["prediction_request_count"] == 0
    assert m["prediction_error_count"] == 0
    assert m["unavailable_count"] == 0


def test_s_unavailable_counts_only_as_unavailable(serve):
    with serve(error=RuntimeError("store missing")) as client:
        client.post("/predict", json=VALID)
        m = _metrics(client)
    assert m["unavailable_count"] == 1
    assert m["prediction_request_count"] == 0
    assert m["prediction_error_count"] == 0
    assert m["invalid_request_count"] == 0


def test_metrics_endpoint_returns_all_fields(serve):
    with serve() as client:
        assert set(_metrics(client)) == {
            "prediction_request_count",
            "prediction_error_count",
            "prediction_error_rate",
            "invalid_request_count",
            "unavailable_count",
            "average_prediction_latency_ms",
        }


def _no_registry(*args, **kwargs):
    raise AssertionError("the registry must not be used when AI4I_MODEL_URI is set")


@pytest.fixture
def registry_db(tmp_path, monkeypatch):
    """Point the configured tracking store at a SQLite file in tmp_path (not created)."""
    path = tmp_path / "mlflow.db"
    monkeypatch.setattr(tracking, "MLFLOW_TRACKING_URI", f"sqlite:///{path.as_posix()}")
    monkeypatch.delenv(api.MODEL_URI_ENV, raising=False)
    return path


def test_missing_local_registry_fails_fast_without_registry_lookup(registry_db, monkeypatch):
    monkeypatch.setattr(api, "get_champion", _no_registry)
    monkeypatch.setattr(api.mlflow.sklearn, "load_model", _no_registry)
    with TestClient(api.app) as client:
        health = client.get("/health")
        assert health.status_code == 503
        assert health.json()["reason"] == "local MLflow registry is unavailable"
        assert client.post("/predict", json=VALID).status_code == 503
    assert not registry_db.exists()  # MLflow never touched the store


def test_w_x_model_uri_override_uses_model_info(tmp_path, monkeypatch, registry_db):
    (tmp_path / "model_info.json").write_text(
        json.dumps({"registered_model_name": "ai4i-failure-model", "model_version": "4", "source_run_id": "run-x"})
    )
    loaded = []
    monkeypatch.setenv(api.MODEL_URI_ENV, str(tmp_path))
    monkeypatch.setattr(api, "get_champion", _no_registry)
    monkeypatch.setattr(api.mlflow.sklearn, "load_model", lambda uri: loaded.append(uri) or FakeModel())

    serving = api.load_serving_model()
    assert loaded == [str(tmp_path)]
    assert (serving.model_version, serving.source_run_id) == ("4", "run-x")


def test_model_uri_without_metadata_still_serves(tmp_path, monkeypatch, registry_db):
    monkeypatch.setenv(api.MODEL_URI_ENV, str(tmp_path))
    monkeypatch.setattr(api, "get_champion", _no_registry)
    monkeypatch.setattr(api.mlflow.sklearn, "load_model", lambda uri: FakeModel())
    serving = api.load_serving_model()
    assert (serving.model_version, serving.source_run_id) == ("unknown", "unknown")


def test_default_loads_resolved_champion_version(monkeypatch, registry_db):
    registry_db.touch()  # the local registry exists, so the normal champion path runs
    loaded = []
    monkeypatch.setattr(api, "get_champion", lambda: (type("Version", (), {"version": 4})(), "run-champion"))
    monkeypatch.setattr(api.mlflow.sklearn, "load_model", lambda uri: loaded.append(uri) or FakeModel())
    serving = api.load_serving_model()
    assert loaded == ["models:/ai4i-failure-model/4"]
    assert (serving.model_version, serving.source_run_id) == ("4", "run-champion")
