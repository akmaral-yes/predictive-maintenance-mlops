import json

import mlflow
import pytest

from ai4i_mlops import baseline_check, tracking
from ai4i_mlops.config import AP_MARGIN, MIN_RECALL
from ai4i_mlops.gate import AP_KEY, RECALL_KEY

BASELINE = {RECALL_KEY: 0.657, AP_KEY: 0.746, "source": "train split only", "seed": 42}


@pytest.fixture
def candidate(monkeypatch):
    """Replace the expensive fit with fixed metrics; MLflow tracking calls fail the test."""
    metrics = {RECALL_KEY: 0.657, AP_KEY: 0.746}

    def no_tracking(*args, **kwargs):
        raise AssertionError("baseline_check must not touch MLflow tracking")

    monkeypatch.setattr(baseline_check, "train_split_candidate", lambda: (dict(metrics), {"n_train_rows": 6000, "n_test_rows": 2000}))
    for name in ("set_tracking_uri", "start_run", "create_experiment", "set_experiment"):
        monkeypatch.setattr(mlflow, name, no_tracking)
    monkeypatch.setattr(tracking, "configure_tracking", no_tracking)
    return metrics


@pytest.fixture
def baseline_file(tmp_path):
    path = tmp_path / "baseline_metrics.json"
    path.write_text(json.dumps(BASELINE))
    return path


def test_a_matching_candidate_passes(candidate, baseline_file, capsys):
    assert baseline_check.main(["--baseline", str(baseline_file)]) == 0
    assert capsys.readouterr().out.splitlines()[-1] == "PASS"


def test_b_recall_below_floor_fails(candidate, baseline_file, capsys):
    candidate[RECALL_KEY] = MIN_RECALL - 0.01
    assert baseline_check.main(["--baseline", str(baseline_file)]) == 1
    assert capsys.readouterr().out.splitlines()[-1] == "FAIL"


def test_c_ap_regression_fails(candidate, baseline_file):
    candidate[AP_KEY] = BASELINE[AP_KEY] - AP_MARGIN - 0.01
    assert baseline_check.main(["--baseline", str(baseline_file)]) == 1


def test_d_missing_baseline_is_operational_error(candidate, tmp_path, capsys):
    assert baseline_check.main(["--baseline", str(tmp_path / "absent.json")]) == 2
    assert "does not exist" in capsys.readouterr().out


def test_e_malformed_json_is_operational_error(candidate, tmp_path):
    path = tmp_path / "bad.json"
    path.write_text("{not json")
    assert baseline_check.main(["--baseline", str(path)]) == 2


@pytest.mark.parametrize("key", [RECALL_KEY, AP_KEY])
def test_f_missing_metric_key_is_operational_error(candidate, tmp_path, key):
    path = tmp_path / "baseline.json"
    path.write_text(json.dumps({k: v for k, v in BASELINE.items() if k != key}))
    assert baseline_check.main(["--baseline", str(path)]) == 2


@pytest.mark.parametrize("value", ["0.746", None, True, [0.7]])
def test_g_wrong_metric_type_is_operational_error(candidate, tmp_path, value):
    path = tmp_path / "baseline.json"
    path.write_text(json.dumps({**BASELINE, AP_KEY: value}))
    assert baseline_check.main(["--baseline", str(path)]) == 2


def test_h_i_write_baseline_writes_metrics_and_provenance(candidate, tmp_path):
    path = tmp_path / "config" / "baseline_metrics.json"
    assert baseline_check.main(["--baseline", str(path), "--write-baseline"]) == 0
    written = json.loads(path.read_text())
    assert (written[RECALL_KEY], written[AP_KEY]) == (0.657, 0.746)
    assert {"source", "seed", "n_train_rows", "n_test_rows", "python_version", "sklearn_version"} <= set(written)
    assert baseline_check.load_baseline(path) == {RECALL_KEY: 0.657, AP_KEY: 0.746}


def test_j_check_creates_no_tracking_store(candidate, baseline_file, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # a default MLflow store would appear in the working directory
    assert baseline_check.main(["--baseline", str(baseline_file)]) == 0
    assert not (tmp_path / "mlruns").exists() and not (tmp_path / "mlflow.db").exists()
