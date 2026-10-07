from typing import Final

SEED: Final[int] = 42

RF_PARAMS: Final[dict[str, int | str]] = {
    "n_estimators": 300,
    "class_weight": "balanced",
    "n_jobs": -1,
    "random_state": SEED,
}
