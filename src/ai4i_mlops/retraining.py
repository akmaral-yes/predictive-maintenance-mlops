"""Retrain a candidate on original train rows plus reviewed (approved) feedback only.

Reads only the validated feedback file, checks every row's provenance against the Phase 1 split
indices, evaluates on the unchanged fixed test split and logs one MLflow run. It never registers,
aliases, promotes or rolls back: that is the explicit registry lifecycle.
"""

import argparse
from collections.abc import Sequence

import mlflow
import pandas as pd
from mlflow.exceptions import MlflowException

from ai4i_mlops.config import (
    PREDICTION_THRESHOLD,
    RF_PARAMS,
    SEED,
    VALIDATED_FEEDBACK_PATH,
)
from ai4i_mlops.contexts import CONTEXTS
from ai4i_mlops.data import clean, load_raw, split
from ai4i_mlops.evaluation import evaluate_slices, format_slices
from ai4i_mlops.feedback import FeedbackError, load_validated_feedback
from ai4i_mlops.pipeline import build_pipeline, predict_failure_proba
from ai4i_mlops.tracking import get_or_create_experiment_id, log_fitted_pipeline

EXIT_OK = 0
EXIT_ERROR = 2


def _describe(indices: pd.Index) -> str:
    shown = ", ".join(str(i) for i in sorted(indices)[:10])
    return f"{len(indices)} source_index value(s): {shown}{' ...' if len(indices) > 10 else ''}"


def combine_with_feedback(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_feedback: pd.DataFrame,
    y_feedback: pd.Series,
    future_index: pd.Index,
    test_index: pd.Index,
) -> tuple[pd.DataFrame, pd.Series]:
    """Original train rows plus feedback rows, keeping source indices. Provenance is checked by index only."""
    if not X_feedback.index.equals(y_feedback.index):
        raise ValueError("Feedback X and y are not aligned on source_index")
    feedback = X_feedback.index
    problems = []
    if len(in_train := feedback.intersection(X_train.index)):
        problems.append(f"feedback overlaps the train split: {_describe(in_train)}")
    if len(in_test := feedback.intersection(test_index)):
        problems.append(f"feedback overlaps the fixed test split: {_describe(in_test)}")
    if len(outside := feedback.difference(future_index)):
        problems.append(f"feedback is not from the future split: {_describe(outside)}")
    if problems:
        raise ValueError("; ".join(problems))

    X = pd.concat([X_train, X_feedback[X_train.columns]])
    y = pd.concat([y_train, y_feedback.rename(y_train.name)])
    if not X.index.is_unique:
        raise ValueError(f"Combined training index has duplicates: {_describe(X.index[X.index.duplicated()])}")
    return X, y


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Retrain on original train rows plus validated feedback.")
    parser.parse_args(argv)

    try:
        X_feedback, y_feedback = load_validated_feedback(VALIDATED_FEEDBACK_PATH)
        splits = split(*clean(load_raw()))
        X_train, y_train = splits["train"]
        X_future, _ = splits["future"]
        X_test, y_test = splits["test"]
        X_combined, y_combined = combine_with_feedback(
            X_train, y_train, X_feedback, y_feedback, X_future.index, X_test.index
        )
    except FeedbackError as e:
        print("error: refusing to retrain without valid reviewed feedback (no fallback to ordinary training)")
        for problem in e.problems:
            print(f"  {problem}")
        return EXIT_ERROR
    except (ValueError, OSError, KeyError) as e:
        print(f"error: {e}")
        return EXIT_ERROR

    model = build_pipeline().fit(X_combined, y_combined)
    results = evaluate_slices(X_test, y_test, predict_failure_proba(model, X_test))

    try:
        with mlflow.start_run(experiment_id=get_or_create_experiment_id(), tags={"training_mode": "retrain"}) as run:
            mlflow.log_params(
                {
                    **RF_PARAMS,
                    "seed": SEED,
                    "prediction_threshold": PREDICTION_THRESHOLD,
                    "n_train_rows": len(X_combined),
                    "training_mode": "retrain",
                    "n_original_train_rows": len(X_train),
                    "n_validated_feedback_rows": len(X_feedback),
                    "n_total_train_rows": len(X_combined),
                }
            )
            mlflow.log_metrics({name: value for name, value in results.items() if value is not None})
            log_fitted_pipeline(model, X_combined)
    except MlflowException as e:
        print(f"error: {e}")
        return EXIT_ERROR

    print(f"run {run.info.run_id}")
    print(f"original train rows      {len(X_train)}")
    print(f"validated feedback rows  {len(X_feedback)}")
    print(f"total train rows         {len(X_combined)}")
    print(f"fixed test rows          {len(X_test)}")
    print("evaluation set = unchanged fixed test split")
    print(format_slices(results, CONTEXTS))
    print("Not registered. Next: uv run python -m ai4i_mlops.registry register --run-id " + run.info.run_id)
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
