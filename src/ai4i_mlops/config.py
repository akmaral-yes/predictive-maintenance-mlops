from pathlib import Path
from typing import Final

SEED: Final[int] = 42

RF_PARAMS: Final[dict[str, int | str]] = {
    "n_estimators": 300,
    "class_weight": "balanced",
    "n_jobs": -1,
    "random_state": SEED,
}

# Turns probabilities into 0/1 predictions for recall/precision/F1. Not a gate threshold.
PREDICTION_THRESHOLD: Final[float] = 0.5

# Gate policy. Demo values informed by the first baseline run (pooled test recall 0.657,
# AP 0.746), not statistically derived production thresholds.
# Minimum acceptable pooled failure recall.
MIN_RECALL: Final[float] = 0.60
# Maximum allowed drop in pooled Average Precision relative to the current champion.
AP_MARGIN: Final[float] = 0.05

# Demo drift-alert threshold on the two-sample KS statistic, not a production-calibrated value.
KS_THRESHOLD: Final[float] = 0.10
# Deterministic sensor offsets for the drift demonstration (column -> amount added).
SIMULATED_SHIFTS: Final[dict[str, float]] = {
    "Air temperature": 2.0,
    "Torque": 5.0,
}

ROOT: Final[Path] = Path(__file__).resolve().parents[2]

MLFLOW_TRACKING_URI: Final[str] = f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}"
MLFLOW_ARTIFACT_ROOT: Final[Path] = ROOT / "mlruns"
EXPERIMENT_NAME: Final[str] = "ai4i-predictive-maintenance"
# Name of the model logged inside each training run.
MODEL_NAME: Final[str] = "model"
# Stable name of the versioned model family in the MLflow Model Registry.
REGISTERED_MODEL_NAME: Final[str] = "ai4i-failure-model"
