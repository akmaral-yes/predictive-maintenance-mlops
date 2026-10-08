import mlflow

from ai4i_mlops.config import PREDICTION_THRESHOLD, RF_PARAMS, SEED
from ai4i_mlops.contexts import CONTEXTS
from ai4i_mlops.data import clean, load_raw, split
from ai4i_mlops.evaluation import evaluate_slices, format_slices
from ai4i_mlops.pipeline import build_pipeline, predict_failure_proba
from ai4i_mlops.tracking import get_or_create_experiment_id, log_fitted_pipeline


def train() -> tuple[str, dict[str, float | int | None]]:
    """Fit on the 60% train split, evaluate on the fixed test split, log one MLflow run."""
    X, y = clean(load_raw())
    splits = split(X, y)  # the "future" split is reserved for later phases and not touched here
    X_train, y_train = splits["train"]
    X_test, y_test = splits["test"]

    model = build_pipeline().fit(X_train, y_train)
    results = evaluate_slices(X_test, y_test, predict_failure_proba(model, X_test))

    with mlflow.start_run(experiment_id=get_or_create_experiment_id()) as run:
        mlflow.log_params(
            {**RF_PARAMS, "seed": SEED, "prediction_threshold": PREDICTION_THRESHOLD, "n_train_rows": len(X_train)}
        )
        mlflow.log_metrics({name: value for name, value in results.items() if value is not None})
        log_fitted_pipeline(model, X_train)
    return run.info.run_id, results


if __name__ == "__main__":
    run_id, results = train()
    print(f"run {run_id}")
    print(format_slices(results, CONTEXTS))
