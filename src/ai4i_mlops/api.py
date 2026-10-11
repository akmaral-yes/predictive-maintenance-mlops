"""Serving API for the approved model.

The model is loaded once per process at startup: from AI4I_MODEL_URI when set (an exported model
directory, no registry needed), otherwise the registry champion. The sklearn flavour is used
because predict_failure_proba needs predict_proba. If loading fails the app still starts, and
/health and /predict answer 503 until the process is restarted.
"""

import json
import logging
import os
import threading
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field
from sklearn.pipeline import Pipeline

from ai4i_mlops import tracking
from ai4i_mlops.config import PREDICTION_THRESHOLD, REGISTERED_MODEL_NAME
from ai4i_mlops.data import CATEGORICAL, NUMERIC
from ai4i_mlops.pipeline import predict_failure_proba
from ai4i_mlops.registry import MODEL_INFO_FILE, get_champion

MODEL_URI_ENV = "AI4I_MODEL_URI"
UNKNOWN = "unknown"

logger = logging.getLogger(__name__)

# API field -> model column, in the model's column order.
FIELD_TO_COLUMN = dict(
    zip(
        ("machine_type", "air_temperature", "process_temperature", "rotational_speed", "torque", "tool_wear"),
        (*CATEGORICAL, *NUMERIC),
        strict=True,
    )
)


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    machine_type: Literal["L", "M", "H"]
    air_temperature: float
    process_temperature: float
    rotational_speed: float = Field(ge=0)
    torque: float = Field(ge=0)
    tool_wear: float = Field(ge=0)


def to_frame(request: PredictionRequest) -> pd.DataFrame:
    """One-row frame with exactly the model's columns and order."""
    row = {column: getattr(request, field) for field, column in FIELD_TO_COLUMN.items()}
    return pd.DataFrame([row], columns=list(FIELD_TO_COLUMN.values()))


@dataclass(frozen=True)
class ServingModel:
    model: Pipeline
    model_version: str
    source_run_id: str


def _read_model_info(uri: str) -> dict[str, str]:
    """Metadata written by `registry export`; missing or unreadable metadata is not fatal."""
    path = Path(uri.removeprefix("file://")) / MODEL_INFO_FILE
    try:
        info = json.loads(path.read_text())
    except (OSError, ValueError):
        return {}
    return info if isinstance(info, dict) else {}


class ModelUnavailableError(Exception):
    """A known, non-sensitive reason why no model can be served."""


def load_serving_model() -> ServingModel:
    uri = os.environ.get(MODEL_URI_ENV)
    if uri:
        info = _read_model_info(uri)
        return ServingModel(
            mlflow.sklearn.load_model(uri),
            str(info.get("model_version", UNKNOWN)),
            str(info.get("source_run_id", UNKNOWN)),
        )
    if tracking.local_store_missing():
        # Without this, MLflow retries opening the absent SQLite store for ~100 s before failing.
        raise ModelUnavailableError("local MLflow registry is unavailable")
    champion, run_id = get_champion()
    # The version the alias resolved to, so the reported metadata matches the loaded model.
    model = mlflow.sklearn.load_model(f"models:/{REGISTERED_MODEL_NAME}/{champion.version}")
    return ServingModel(model, str(champion.version), run_id)


class ServiceMetrics:
    """Process-local counters; they reset when the process or container restarts."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._requests = 0
        self._errors = 0
        self._invalid = 0
        self._unavailable = 0
        self._latency_ms_total = 0.0

    def record_prediction(self, latency_ms: float, failed: bool) -> None:
        with self._lock:
            self._requests += 1
            self._errors += failed
            self._latency_ms_total += latency_ms

    def record_invalid(self) -> None:
        with self._lock:
            self._invalid += 1

    def record_unavailable(self) -> None:
        with self._lock:
            self._unavailable += 1

    def snapshot(self) -> dict[str, float | int]:
        with self._lock:
            return {
                "prediction_request_count": self._requests,
                "prediction_error_count": self._errors,
                "prediction_error_rate": self._errors / self._requests if self._requests else 0.0,
                "invalid_request_count": self._invalid,
                "unavailable_count": self._unavailable,
                "average_prediction_latency_ms": self._latency_ms_total / self._requests if self._requests else 0.0,
            }


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    app.state.metrics = ServiceMetrics()
    app.state.serving = None
    app.state.load_error = None
    try:
        app.state.serving = load_serving_model()
    except ModelUnavailableError as e:
        app.state.load_error = str(e)
        logger.error("model not loaded: %s", e)
    except Exception as e:  # noqa: BLE001 - any failure means 503s, not a crashed app; reason kept out of responses
        app.state.load_error = f"model could not be loaded ({type(e).__name__})"
        logger.error("model loading failed: %s: %s", type(e).__name__, e)
    yield


app = FastAPI(title="AI4I failure prediction", lifespan=lifespan)


@app.exception_handler(RequestValidationError)
async def count_invalid_predictions(request: Request, exc: RequestValidationError) -> JSONResponse:
    if request.method == "POST" and request.url.path == "/predict":
        request.app.state.metrics.record_invalid()
    # FastAPI's default 422 body, minus the echoed "input": NaN/inf inputs cannot be encoded as
    # JSON (the default would turn them into a 500) and feature values are not repeated back.
    errors = [{k: v for k, v in error.items() if k != "input"} for error in exc.errors()]
    return JSONResponse(status_code=422, content={"detail": jsonable_encoder(errors)})


@app.get("/health")
def health(request: Request) -> JSONResponse:
    serving: ServingModel | None = request.app.state.serving
    if serving is None:
        return JSONResponse(
            status_code=503,
            content={"status": "unavailable", "model_loaded": False, "reason": request.app.state.load_error},
        )
    return JSONResponse(
        {
            "status": "ok",
            "model_loaded": True,
            "model_version": serving.model_version,
            "source_run_id": serving.source_run_id,
        }
    )


@app.post("/predict")
def predict(payload: PredictionRequest, request: Request) -> dict[str, float | bool | str]:
    state = request.app.state
    if state.serving is None:
        state.metrics.record_unavailable()
        raise HTTPException(status_code=503, detail="model not loaded")

    start = time.perf_counter()
    try:
        probability = float(predict_failure_proba(state.serving.model, to_frame(payload)).iloc[0])
    except Exception as e:  # noqa: BLE001 - any inference failure is a counted 500 without details
        state.metrics.record_prediction((time.perf_counter() - start) * 1000, failed=True)
        logger.error("inference failed: %s", type(e).__name__)  # never log feature values
        raise HTTPException(status_code=500, detail="inference failed") from None
    state.metrics.record_prediction((time.perf_counter() - start) * 1000, failed=False)
    return {
        "failure_probability": probability,
        "predicted_failure": probability >= PREDICTION_THRESHOLD,
        "threshold": PREDICTION_THRESHOLD,
        "model_version": state.serving.model_version,
    }


@app.get("/metrics")
def metrics(request: Request) -> dict[str, float | int]:
    """Process-local service telemetry (reset on restart)."""
    return request.app.state.metrics.snapshot()
