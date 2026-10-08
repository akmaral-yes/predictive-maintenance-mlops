import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

from ai4i_mlops.config import AP_MARGIN, MIN_RECALL
from ai4i_mlops.tracking import configure_tracking

RECALL_KEY = "test_all_recall"
AP_KEY = "test_all_average_precision"
# Exact-boundary values must not be decided by binary floating-point representation.
TOLERANCE = 1e-9

Metrics = Mapping[str, float | int | None]
Status = Literal["pass", "fail", "skipped"]


@dataclass(frozen=True)
class GateCheck:
    name: str
    status: Status
    message: str


@dataclass(frozen=True)
class GateResult:
    passed: bool
    checks: tuple[GateCheck, ...]


def _meets(value: float, required: float) -> bool:
    return value >= required - TOLERANCE


def _recall_check(recall: float | None, min_recall: float) -> GateCheck:
    name = "recall_floor"
    if recall is None:
        return GateCheck(name, "fail", "candidate pooled recall is undefined")
    if _meets(recall, min_recall):
        return GateCheck(name, "pass", f"recall {recall:.3f} >= {min_recall:.3f}")
    return GateCheck(name, "fail", f"recall {recall:.3f} < {min_recall:.3f}")


def _ap_check(ap: float | None, baseline: Metrics | None, ap_margin: float) -> GateCheck:
    name = "ap_no_regression"
    if baseline is None:
        return GateCheck(name, "skipped", "no baseline/champion yet")
    baseline_ap = baseline.get(AP_KEY)
    if baseline_ap is None:
        raise ValueError(f"Baseline metrics were given but {AP_KEY} is missing or None")
    if ap is None:
        return GateCheck(name, "fail", "candidate pooled AP is undefined")
    required = baseline_ap - ap_margin
    if _meets(ap, required):
        return GateCheck(name, "pass", f"AP {ap:.3f} >= baseline {baseline_ap:.3f} - {ap_margin:.3f}")
    return GateCheck(name, "fail", f"AP {ap:.3f} < baseline {baseline_ap:.3f} - {ap_margin:.3f}")


def evaluate_gate(
    candidate_metrics: Metrics,
    baseline_metrics: Metrics | None,
    min_recall: float = MIN_RECALL,
    ap_margin: float = AP_MARGIN,
) -> GateResult:
    """Decide on pooled test recall and AP only. Every check runs; any "fail" fails the gate."""
    checks = (
        _recall_check(candidate_metrics.get(RECALL_KEY), min_recall),
        _ap_check(candidate_metrics.get(AP_KEY), baseline_metrics, ap_margin),
    )
    return GateResult(passed=all(c.status != "fail" for c in checks), checks=checks)


def load_run_metrics(run_id: str) -> dict[str, float]:
    """All metrics recorded for an MLflow run."""
    from mlflow import MlflowClient

    configure_tracking()
    return dict(MlflowClient().get_run(run_id).data.metrics)


def load_pooled_metrics(run_id: str) -> dict[str, float | None]:
    """The two pooled metrics the gate decides on; None when the run did not record one."""
    metrics = load_run_metrics(run_id)
    return {key: metrics.get(key) for key in (RECALL_KEY, AP_KEY)}


def format_report(
    candidate_metrics: Metrics,
    baseline_metrics: Metrics | None,
    result: GateResult,
    min_recall: float = MIN_RECALL,
    ap_margin: float = AP_MARGIN,
) -> str:
    def num(value: float | None) -> str:
        return "undefined" if value is None else f"{value:.3f}"

    lines = [
        "Evaluation gate (pooled fixed-test metrics)",
        f"  candidate recall    {num(candidate_metrics.get(RECALL_KEY))}   MIN_RECALL {min_recall:.3f}",
        f"  candidate AP        {num(candidate_metrics.get(AP_KEY))}",
    ]
    if baseline_metrics is not None:
        baseline_ap = baseline_metrics.get(AP_KEY)
        lines.append(f"  baseline AP         {num(baseline_ap)}")
        if baseline_ap is not None:
            lines.append(f"  minimum allowed AP  {num(baseline_ap - ap_margin)}   AP_MARGIN {ap_margin:.3f}")

    lines.append("Checks")
    lines += [f"  {c.status.upper():<8} {c.name}: {c.message}" for c in result.checks]

    lines.append("Per-Type diagnostics (never gate)")
    lines.append(f"  {'context':<8}{'recall':>10}{'AP':>11}{'failures':>10}")
    for context in ("L", "M", "H"):
        key = f"test_{context}"
        failures = candidate_metrics.get(f"{key}_n_failures")
        lines.append(
            f"  {context:<8}{num(candidate_metrics.get(f'{key}_recall')):>10}"
            f"{num(candidate_metrics.get(f'{key}_average_precision')):>11}"
            f"{'n/a' if failures is None else int(failures):>10}"
        )

    lines.append("PASS" if result.passed else "FAIL")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Gate a candidate run against a baseline run on pooled test metrics.")
    parser.add_argument("--candidate-run-id", required=True)
    parser.add_argument("--baseline-run-id", help="current champion run; omit only before a first champion exists")
    args = parser.parse_args(argv)

    candidate = load_pooled_metrics(args.candidate_run_id)
    baseline = load_pooled_metrics(args.baseline_run_id) if args.baseline_run_id else None
    result = evaluate_gate(candidate, baseline)

    # Full metrics only feed the diagnostics section; the decision above used pooled metrics.
    diagnostics = {**load_run_metrics(args.candidate_run_id), **candidate}
    print(f"candidate run {args.candidate_run_id}")
    print(f"baseline run  {args.baseline_run_id or 'none'}")
    print(format_report(diagnostics, baseline, result))
    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
