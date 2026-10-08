"""Monitor the registry champion: labelled performance by Type and pooled numeric-feature drift.

train split -> drift reference; future split -> current monitoring batch; the fixed test split
is the model-quality benchmark and is not used here. Drift means INVESTIGATE; nothing is
trained, registered, promoted or rolled back automatically.
"""

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import mlflow.sklearn
import pandas as pd
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
from scipy.stats import ks_2samp
from sklearn.pipeline import Pipeline

from ai4i_mlops.config import KS_THRESHOLD, REGISTERED_MODEL_NAME, SIMULATED_SHIFTS
from ai4i_mlops.contexts import CONTEXTS
from ai4i_mlops.data import NUMERIC, clean, load_raw, split
from ai4i_mlops.evaluation import evaluate_slices, format_slices
from ai4i_mlops.pipeline import predict_failure_proba
from ai4i_mlops.registry import RegistryError, get_champion
from ai4i_mlops.tracking import configure_tracking

PREFIX = "batch"
EXIT_OK = 0
EXIT_INVESTIGATE = 1
EXIT_ERROR = 2


@dataclass(frozen=True)
class FeatureDrift:
    feature: str
    ks_statistic: float
    p_value: float  # informational only; never part of the decision
    drifted: bool


def _observations(frame: pd.DataFrame, feature: str, label: str) -> pd.Series:
    values = pd.to_numeric(frame[feature], errors="coerce").dropna()
    if values.empty:
        raise ValueError(f"{label} column {feature!r} has no usable numeric observations")
    return values


def compute_feature_drift(
    reference: pd.DataFrame, current: pd.DataFrame, threshold: float = KS_THRESHOLD
) -> tuple[FeatureDrift, ...]:
    """Two-sample KS per numeric feature; a feature drifts when ks_statistic >= threshold."""
    # Only data.NUMERIC is tested. Type is a known subgroup used for performance diagnostics;
    # per-Type samples of a few hundred rows are too small for one fixed KS threshold to be a
    # reliable drift rule in this demo.
    for frame, label in ((reference, "reference"), (current, "current")):
        if frame.empty:
            raise ValueError(f"{label} data is empty")
        missing = [f for f in NUMERIC if f not in frame.columns]
        if missing:
            raise ValueError(f"{label} data is missing numeric features: {missing}")

    results = []
    for feature in NUMERIC:
        test = ks_2samp(_observations(reference, feature, "reference"), _observations(current, feature, "current"))
        statistic = float(test.statistic)
        results.append(FeatureDrift(feature, statistic, float(test.pvalue), statistic >= threshold))
    return tuple(results)


def apply_simulated_shift(X: pd.DataFrame, shifts: Mapping[str, float] = SIMULATED_SHIFTS) -> pd.DataFrame:
    """Copy of X with a constant sensor offset added to each configured column."""
    missing = [c for c in shifts if c not in X.columns]
    if missing:
        raise ValueError(f"Cannot shift missing columns: {missing}")
    shifted = X.copy()
    for column, amount in shifts.items():
        shifted[column] = shifted[column] + amount
    return shifted


def load_champion() -> tuple[str, str, Pipeline]:
    """Registered champion version, its source run id and the model of that exact version."""
    champion, run_id = get_champion()
    model = mlflow.sklearn.load_model(f"models:/{REGISTERED_MODEL_NAME}/{champion.version}")
    return str(champion.version), run_id, model


def load_run_params(run_id: str) -> dict[str, str]:
    """Params recorded on a training run (MLflow stores them as strings)."""
    configure_tracking()
    return dict(MlflowClient().get_run(run_id).data.params)


def in_sample_feedback_rows(params: Mapping[str, str]) -> int | None:
    """Future-batch rows a retrained champion was trained on; None if not retrained, unknown or zero."""
    if params.get("training_mode") != "retrain":
        return None
    try:
        rows = int(params["n_validated_feedback_rows"])
    except (KeyError, TypeError, ValueError):
        return None
    return rows if rows > 0 else None


def load_monitoring_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Train features as reference, future features and labels as the current batch."""
    splits = split(*clean(load_raw()))
    X_reference, _ = splits["train"]
    X_current, y_current = splits["future"]
    return X_reference, X_current, y_current


def format_monitoring_report(
    version: str,
    run_id: str,
    simulated: bool,
    n_reference: int,
    performance: dict[str, float | int | None],
    drift: tuple[FeatureDrift, ...],
    in_sample_rows: int | None = None,
) -> str:
    drifted = [d.feature for d in drift if d.drifted]
    mode = "normal"
    if simulated:
        offsets = ", ".join(f"{column} +{amount}" for column, amount in SIMULATED_SHIFTS.items())
        mode = f"simulated shift (deliberately shifted: {offsets})"
    lines = [
        "Champion",
        "--------",
        f"registered version  {version}",
        f"source run ID       {run_id}",
        "",
        "Batch",
        "-----",
        f"mode                {mode}",
        f"reference rows      {n_reference}  (train split)",
        f"current rows        {performance[f'{PREFIX}_all_n_rows']}  (future split)",
        "",
        "Performance diagnostics" + (" (on the SHIFTED batch)" if simulated else ""),
        "-----------------------",
    ]
    if in_sample_rows is not None:
        lines += [
            f"NOTE: this champion was trained on {in_sample_rows} rows of the future batch, so the",
            "performance figures below are partly in-sample and not an unbiased",
            "estimate. The drift check is unaffected.",
        ]
    lines += [
        format_slices(performance, CONTEXTS, prefix=PREFIX),
        (
            f"Note: Type H has only {performance[f'{PREFIX}_H_n_failures']} failures in this batch; "
            "its metrics are unstable and diagnostic only."
        ),
        "",
        f"Data drift (two-sample KS, alert when KS >= {KS_THRESHOLD:.2f}; p-value informational)",
        "----------",
        f"{'feature':<22}{'KS':>8}{'p-value':>12}  status",
    ]
    lines += [
        f"{d.feature:<22}{d.ks_statistic:>8.3f}{d.p_value:>12.2e}  {'DRIFT' if d.drifted else 'OK'}" for d in drift
    ]
    lines += ["", "Overall", "-------"]
    if drifted:
        lines += [
            "INVESTIGATE",
            f"drifted features: {', '.join(drifted)}",
            "No automatic action is taken: no model is trained, registered, promoted or rolled back.",
        ]
    else:
        lines.append("OK")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Monitor the registry champion on the future batch.")
    parser.add_argument(
        "--simulate-shift", action="store_true", help="add SIMULATED_SHIFTS to the future batch before monitoring"
    )
    args = parser.parse_args(argv)

    try:
        version, run_id, model = load_champion()
        in_sample_rows = in_sample_feedback_rows(load_run_params(run_id))
        X_reference, X_current, y_current = load_monitoring_data()
        if args.simulate_shift:
            X_current = apply_simulated_shift(X_current)
        performance = evaluate_slices(X_current, y_current, predict_failure_proba(model, X_current), prefix=PREFIX)
        drift = compute_feature_drift(X_reference, X_current)
    except (RegistryError, MlflowException, ValueError, OSError) as e:
        print(f"error: {e}")
        return EXIT_ERROR

    print(
        format_monitoring_report(
            version, run_id, args.simulate_shift, len(X_reference), performance, drift, in_sample_rows
        )
    )
    return EXIT_INVESTIGATE if any(d.drifted for d in drift) else EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
