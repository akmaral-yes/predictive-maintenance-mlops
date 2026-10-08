import argparse
from collections.abc import Iterable

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
)

from ai4i_mlops.config import EXPERIMENT_NAME, MODEL_NAME, PREDICTION_THRESHOLD
from ai4i_mlops.contexts import CONTEXTS, select_context
from ai4i_mlops.data import clean, load_raw, split
from ai4i_mlops.pipeline import predict_failure_proba
from ai4i_mlops.tracking import configure_tracking

METRICS = ("recall", "precision", "f1", "average_precision")


def compute_metrics(
    y_true: ArrayLike, y_proba: ArrayLike, threshold: float = PREDICTION_THRESHOLD
) -> dict[str, float | None]:
    """Recall, precision, F1 and AP. A metric that is undefined for the input is None."""
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba, dtype=float)
    if len(y_true) != len(y_proba):
        raise ValueError(f"y_true has {len(y_true)} rows but y_proba has {len(y_proba)}")
    if len(y_true) == 0:
        return dict.fromkeys(METRICS)

    y_pred = (y_proba >= threshold).astype(int)
    has_positive = bool((y_true == 1).any())
    return {
        "recall": float(recall_score(y_true, y_pred)) if has_positive else None,
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "average_precision": float(average_precision_score(y_true, y_proba)) if has_positive else None,
    }


def evaluate_slices(
    X: pd.DataFrame,
    y: pd.Series,
    y_proba: pd.Series,
    prefix: str = "test",
    threshold: float = PREDICTION_THRESHOLD,
) -> dict[str, float | int | None]:
    """Metrics and counts for every context, flattened as <prefix>_<context>_<name>."""
    if not len(X) == len(y) == len(y_proba):
        raise ValueError(f"Length mismatch: X={len(X)}, y={len(y)}, y_proba={len(y_proba)}")
    if not (X.index.equals(y.index) and y.index.equals(y_proba.index)):
        raise ValueError("X, y and y_proba must share the same index")
    if not y.index.is_unique:
        raise ValueError("Index must be unique so probabilities can be matched to labels")

    results: dict[str, float | int | None] = {}
    for context in CONTEXTS:
        _, y_slice = select_context(X, y, context)
        key = f"{prefix}_{context}"
        results[f"{key}_n_rows"] = len(y_slice)
        results[f"{key}_n_failures"] = int(y_slice.sum())
        metrics = compute_metrics(y_slice, y_proba.loc[y_slice.index], threshold)
        for name, value in metrics.items():
            results[f"{key}_{name}"] = value
    return results


def format_slices(results: dict[str, float | int | None], contexts: Iterable[str], prefix: str = "test") -> str:
    """Plain-text table of counts and metrics, one row per context."""

    def cell(value: float | None) -> str:
        return "n/a" if value is None else f"{value:.3f}"

    lines = [f"{'context':<8}{'rows':>7}{'failures':>10}{'recall':>9}{'precision':>11}{'f1':>8}{'avg prec':>10}"]
    for context in contexts:
        r = {name: results[f"{prefix}_{context}_{name}"] for name in ("n_rows", "n_failures", *METRICS)}
        lines.append(
            f"{context:<8}{r['n_rows']:>7}{r['n_failures']:>10}{cell(r['recall']):>9}"
            f"{cell(r['precision']):>11}{cell(r['f1']):>8}{cell(r['average_precision']):>10}"
        )
    return "\n".join(lines)


def _latest_run_id() -> str:
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    runs = [] if experiment is None else mlflow.search_runs(
        [experiment.experiment_id],
        filter_string="attributes.status = 'FINISHED'",
        order_by=["start_time DESC"],
        max_results=1,
        output_format="list",
    )
    if not runs:
        raise SystemExit(f"No finished runs in experiment {EXPERIMENT_NAME!r}; run ai4i_mlops.training first.")
    return runs[0].info.run_id


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a logged model on the fixed test split.")
    parser.add_argument("--context", choices=CONTEXTS, required=True)
    parser.add_argument("--run-id", help="MLflow run to load (default: latest finished run)")
    args = parser.parse_args()

    configure_tracking()
    run_id = args.run_id or _latest_run_id()
    model = mlflow.sklearn.load_model(f"runs:/{run_id}/{MODEL_NAME}")

    X, y = clean(load_raw())
    X_test, y_test = split(X, y)["test"]
    results = evaluate_slices(X_test, y_test, predict_failure_proba(model, X_test))

    print(f"run {run_id}")
    print(format_slices(results, [args.context]))


if __name__ == "__main__":
    main()
