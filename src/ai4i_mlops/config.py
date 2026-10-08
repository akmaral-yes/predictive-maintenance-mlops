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

ROOT: Final[Path] = Path(__file__).resolve().parents[2]

MLFLOW_TRACKING_URI: Final[str] = f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}"
MLFLOW_ARTIFACT_ROOT: Final[Path] = ROOT / "mlruns"
EXPERIMENT_NAME: Final[str] = "ai4i-predictive-maintenance"
MODEL_NAME: Final[str] = "model"
