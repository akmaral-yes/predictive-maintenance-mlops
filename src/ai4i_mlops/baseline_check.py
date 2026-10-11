"""CI model-quality check: a fresh train-split-only fit, gated against the committed baseline.

No MLflow tracking store, experiment, run or registry is touched, and nothing is read from the
network. The committed baseline is always the deterministic train-split-only fit evaluated on the
fixed test split; it is never replaced by the current champion's metrics.
"""

import argparse
import json
import math
import platform
from collections.abc import Sequence
from pathlib import Path

import sklearn

from ai4i_mlops.config import BASELINE_METRICS_PATH, SEED
from ai4i_mlops.data import clean, load_raw, split
from ai4i_mlops.gate import AP_KEY, RECALL_KEY, evaluate_gate, format_report
from ai4i_mlops.training import fit_and_evaluate

EXIT_PASS = 0
EXIT_GATE_FAILED = 1
EXIT_ERROR = 2


class BaselineError(Exception):
    """The baseline file is missing or unusable."""


def train_split_candidate() -> tuple[dict[str, float | int | None], dict[str, int]]:
    """Fit on the original 60% train split only (no feedback) and evaluate on the fixed test split."""
    splits = split(*clean(load_raw()))
    X_train, y_train = splits["train"]
    X_test, y_test = splits["test"]
    _, results = fit_and_evaluate(X_train, y_train, X_test, y_test)
    return results, {"n_train_rows": len(X_train), "n_test_rows": len(X_test)}


def load_baseline(path: Path) -> dict[str, float]:
    """The pooled metrics the gate compares against; provenance fields are ignored."""
    try:
        data = json.loads(path.read_text())
    except FileNotFoundError as e:
        raise BaselineError(f"baseline file {path} does not exist") from e
    except json.JSONDecodeError as e:
        raise BaselineError(f"baseline file {path} is not valid JSON ({e.msg})") from e
    if not isinstance(data, dict):
        raise BaselineError(f"baseline file {path} must contain a JSON object")
    metrics = {}
    for key in (RECALL_KEY, AP_KEY):
        value = data.get(key)
        if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
            raise BaselineError(f"baseline {key} is missing or not a finite number: {value!r}")
        metrics[key] = float(value)
    return metrics


def write_baseline(path: Path) -> dict[str, object]:
    results, counts = train_split_candidate()
    baseline = {
        RECALL_KEY: results[RECALL_KEY],
        AP_KEY: results[AP_KEY],
        "source": "fresh fit on the original 60% train split only, evaluated on the fixed 20% test split",
        "seed": SEED,
        **counts,
        "python_version": platform.python_version(),
        "sklearn_version": sklearn.__version__,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(baseline, indent=2) + "\n")
    return baseline


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Gate a fresh train-split-only model against the committed baseline.")
    parser.add_argument("--baseline", type=Path, default=BASELINE_METRICS_PATH, help="baseline metrics JSON")
    parser.add_argument("--write-baseline", action="store_true", help="refit and (re)write the baseline file")
    args = parser.parse_args(argv)

    if args.write_baseline:
        baseline = write_baseline(args.baseline)
        print(f"wrote {args.baseline}")
        print(json.dumps(baseline, indent=2))
        return EXIT_PASS

    try:
        baseline = load_baseline(args.baseline)
    except BaselineError as e:
        print(f"error: {e}")
        return EXIT_ERROR

    results, counts = train_split_candidate()
    candidate = {key: results[key] for key in (RECALL_KEY, AP_KEY)}
    gate_result = evaluate_gate(candidate, baseline)
    print(f"baseline  {args.baseline}")
    print(f"candidate fresh fit on {counts['n_train_rows']} train rows, evaluated on {counts['n_test_rows']} fixed test rows")
    print(format_report(results, baseline, gate_result))
    return EXIT_PASS if gate_result.passed else EXIT_GATE_FAILED


if __name__ == "__main__":
    raise SystemExit(main())
