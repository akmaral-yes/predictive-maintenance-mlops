import pytest

from ai4i_mlops import gate
from ai4i_mlops.config import AP_MARGIN, MIN_RECALL
from ai4i_mlops.gate import AP_KEY, RECALL_KEY, GateResult, evaluate_gate

BASELINE = {RECALL_KEY: 0.657, AP_KEY: 0.746}


def _metrics(recall: float | None, ap: float | None, **extra: float) -> dict[str, float | None]:
    return {RECALL_KEY: recall, AP_KEY: ap, **extra}


def _statuses(result: GateResult) -> dict[str, str]:
    return {c.name: c.status for c in result.checks}


def test_a_passing_candidate():
    result = evaluate_gate(_metrics(0.66, 0.74), BASELINE, min_recall=0.60, ap_margin=0.05)
    assert result.passed
    assert _statuses(result) == {"recall_floor": "pass", "ap_no_regression": "pass"}


def test_b_recall_failure_only():
    result = evaluate_gate(_metrics(0.55, 0.74), BASELINE, min_recall=0.60, ap_margin=0.05)
    assert not result.passed
    assert _statuses(result) == {"recall_floor": "fail", "ap_no_regression": "pass"}


def test_c_ap_regression_only():
    # Required AP is 0.746 - 0.05 = 0.696.
    result = evaluate_gate(_metrics(0.70, 0.65), BASELINE, min_recall=0.60, ap_margin=0.05)
    assert not result.passed
    assert _statuses(result) == {"recall_floor": "pass", "ap_no_regression": "fail"}


def test_d_both_failures_are_recorded():
    result = evaluate_gate(_metrics(0.30, 0.40), BASELINE, min_recall=0.60, ap_margin=0.05)
    assert not result.passed
    assert _statuses(result) == {"recall_floor": "fail", "ap_no_regression": "fail"}


def test_e_recall_alone_is_unsafe():
    # Predicting failure everywhere: perfect recall, AP at the base rate.
    result = evaluate_gate(_metrics(1.0, 0.034), BASELINE, min_recall=0.60, ap_margin=0.05)
    assert not result.passed
    assert _statuses(result) == {"recall_floor": "pass", "ap_no_regression": "fail"}


def test_f_exact_boundaries_pass_and_just_below_fails():
    boundary_ap = BASELINE[AP_KEY] - AP_MARGIN
    assert evaluate_gate(_metrics(MIN_RECALL, boundary_ap), BASELINE).passed

    below_recall = evaluate_gate(_metrics(MIN_RECALL - 1e-6, boundary_ap), BASELINE)
    assert _statuses(below_recall) == {"recall_floor": "fail", "ap_no_regression": "pass"}

    below_ap = evaluate_gate(_metrics(MIN_RECALL, boundary_ap - 1e-6), BASELINE)
    assert _statuses(below_ap) == {"recall_floor": "pass", "ap_no_regression": "fail"}


@pytest.mark.parametrize("candidate", [_metrics(None, None), {}])
def test_g_undefined_candidate_metrics_fail(candidate):
    result = evaluate_gate(candidate, BASELINE)
    assert not result.passed
    for check in result.checks:
        assert check.status == "fail"
        assert "undefined" in check.message


@pytest.mark.parametrize(("recall", "passed"), [(0.66, True), (0.55, False)])
def test_h_no_baseline_recall_decides(recall, passed):
    result = evaluate_gate(_metrics(recall, 0.01), None)
    assert result.passed is passed
    ap_check = next(c for c in result.checks if c.name == "ap_no_regression")
    assert ap_check.status == "skipped"
    assert "no baseline" in ap_check.message


def test_i_invalid_baseline_ap_raises():
    with pytest.raises(ValueError, match=AP_KEY):
        evaluate_gate(_metrics(0.66, 0.74), {RECALL_KEY: 0.657, AP_KEY: None})


def test_j_type_metrics_never_gate():
    good = {"test_L_recall": 1.0, "test_M_recall": 1.0, "test_H_recall": 1.0, "test_H_average_precision": 1.0}
    bad = {"test_L_recall": 0.0, "test_M_recall": 0.0, "test_H_recall": 0.0, "test_H_average_precision": 0.0}
    result_good = evaluate_gate(_metrics(0.66, 0.74, **good), BASELINE)
    result_bad = evaluate_gate(_metrics(0.66, 0.74, **bad), BASELINE)
    assert result_good == result_bad


def test_report_lists_checks_and_type_diagnostics():
    candidate = _metrics(0.66, 0.74, test_H_recall=0.25, test_H_n_failures=4.0)
    report = gate.format_report(candidate, BASELINE, evaluate_gate(candidate, BASELINE))
    assert "Per-Type diagnostics (never gate)" in report
    assert "PASS     recall_floor" in report
    assert report.splitlines()[-1] == "PASS"


@pytest.mark.parametrize(("candidate", "code", "final"), [(_metrics(0.66, 0.74), 0, "PASS"), (_metrics(1.0, 0.034), 1, "FAIL")])
def test_cli_exit_codes(monkeypatch, capsys, candidate, code, final):
    by_run = {"cand": candidate, "base": BASELINE}
    monkeypatch.setattr(gate, "load_pooled_metrics", lambda run_id: by_run[run_id])
    monkeypatch.setattr(gate, "load_run_metrics", lambda run_id: by_run[run_id])

    assert gate.main(["--candidate-run-id", "cand", "--baseline-run-id", "base"]) == code
    assert capsys.readouterr().out.splitlines()[-1] == final
