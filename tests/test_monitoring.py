from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier

from ai4i_mlops import monitoring
from ai4i_mlops.config import KS_THRESHOLD, SIMULATED_SHIFTS
from ai4i_mlops.data import NUMERIC
from ai4i_mlops.monitoring import apply_simulated_shift, compute_feature_drift
from ai4i_mlops.registry import RegistryError


def _batch(n: int = 400, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "Type": rng.choice(["L", "M", "H"], size=n, p=[0.6, 0.3, 0.1]),
            "Air temperature": rng.normal(300.0, 2.0, n),
            "Process temperature": rng.normal(310.0, 1.5, n),
            "Rotational speed": rng.normal(1540.0, 180.0, n).round(),
            "Torque": rng.normal(40.0, 10.0, n),
            "Tool wear": rng.integers(0, 250, n),
        }
    )


def _by_feature(drift):
    return {d.feature: d for d in drift}


def test_a_identical_distributions_do_not_drift():
    X = _batch()
    drift = compute_feature_drift(X, X.copy())
    assert all(d.ks_statistic == pytest.approx(0.0) for d in drift)
    assert not any(d.drifted for d in drift)


def test_b_clearly_shifted_feature_crosses_threshold():
    reference = _batch()
    current = reference.assign(Torque=reference["Torque"] + 30.0)
    drift = _by_feature(compute_feature_drift(reference, current))
    assert drift["Torque"].drifted
    assert drift["Torque"].ks_statistic >= KS_THRESHOLD


def test_c_constant_offset_on_one_feature_is_flagged_alone():
    reference = _batch()
    current = reference.copy()
    current["Process temperature"] += 1.5  # one standard deviation of the fixture
    drift = _by_feature(compute_feature_drift(reference, current))
    assert drift["Process temperature"].drifted
    assert [f for f, d in drift.items() if d.drifted] == ["Process temperature"]


def test_d_apply_simulated_shift_returns_shifted_copy():
    X = _batch(50)
    before = X.copy()
    shifted = apply_simulated_shift(X)

    assert shifted is not X
    pd.testing.assert_frame_equal(X, before)
    assert (shifted["Air temperature"] - X["Air temperature"]).tolist() == pytest.approx([2.0] * len(X))
    assert (shifted["Torque"] - X["Torque"]).tolist() == pytest.approx([5.0] * len(X))
    unchanged = [c for c in X.columns if c not in SIMULATED_SHIFTS]
    pd.testing.assert_frame_equal(shifted[unchanged], X[unchanged])


def test_e_simulated_shift_columns_are_numeric_features():
    assert set(SIMULATED_SHIFTS) <= set(NUMERIC)


def test_f_missing_numeric_feature_raises():
    X = _batch()
    with pytest.raises(ValueError, match="missing numeric features.*Torque"):
        compute_feature_drift(X, X.drop(columns="Torque"))


@pytest.mark.parametrize("empty", ["reference", "current"])
def test_g_empty_data_raises(empty):
    X = _batch()
    frames = {"reference": X, "current": X.copy()}
    frames[empty] = X.iloc[0:0]
    with pytest.raises(ValueError, match=f"{empty} data is empty"):
        compute_feature_drift(frames["reference"], frames["current"])


def test_feature_without_numeric_values_raises():
    X = _batch()
    with pytest.raises(ValueError, match="no usable numeric observations"):
        compute_feature_drift(X, X.assign(Torque=np.nan))


def test_inputs_are_not_mutated():
    reference, current = _batch(seed=1), _batch(seed=2)
    before = reference.copy(), current.copy()
    compute_feature_drift(reference, current)
    pd.testing.assert_frame_equal(reference, before[0])
    pd.testing.assert_frame_equal(current, before[1])


def test_h_type_is_never_drift_tested():
    reference = _batch()
    current = reference.assign(Type="H")  # an extreme Type change on its own
    drift = compute_feature_drift(reference, current)
    assert [d.feature for d in drift] == NUMERIC
    assert not any(d.drifted for d in drift)


def _fixed_ks(monkeypatch, statistic: float, pvalue: float) -> None:
    monkeypatch.setattr(monitoring, "ks_2samp", lambda a, b: SimpleNamespace(statistic=statistic, pvalue=pvalue))


def test_i_statistic_equal_to_threshold_is_drift(monkeypatch):
    _fixed_ks(monkeypatch, KS_THRESHOLD, 0.5)
    X = _batch(20)
    assert all(d.drifted for d in compute_feature_drift(X, X))


@pytest.mark.parametrize(("statistic", "pvalue", "drifted"), [(0.05, 1e-30, False), (0.5, 0.99, True)])
def test_j_p_value_does_not_influence_decision(monkeypatch, statistic, pvalue, drifted):
    _fixed_ks(monkeypatch, statistic, pvalue)
    X = _batch(20)
    assert all(d.drifted is drifted for d in compute_feature_drift(X, X))


@pytest.fixture
def fake_champion_and_data(monkeypatch):
    """Tiny in-memory champion and batches; no registry, database or CSV is touched."""
    X = _batch()
    y = pd.Series((np.arange(len(X)) % 10 == 0).astype(int), index=X.index)
    model = DummyClassifier(strategy="prior").fit(X, y)
    monkeypatch.setattr(monitoring, "load_champion", lambda: ("7", "run-abc", model))
    monkeypatch.setattr(monitoring, "load_monitoring_data", lambda: (X, X.copy(), y))
    params: dict[str, str] = {}
    monkeypatch.setattr(monitoring, "load_run_params", lambda run_id: params)
    return params


def test_cli_clean_batch_is_ok(fake_champion_and_data, capsys):
    assert monitoring.main([]) == 0
    out = capsys.readouterr().out
    assert out.splitlines()[-1] == "OK"
    assert "RETRAIN" not in out


def test_cli_shifted_batch_is_investigate(fake_champion_and_data, capsys):
    assert monitoring.main(["--simulate-shift"]) == 1
    out = capsys.readouterr().out
    assert "INVESTIGATE" in out
    assert "drifted features: Air temperature, Torque" in out
    assert "SHIFTED batch" in out
    assert "RETRAIN" not in out


def test_cli_without_champion_is_operational_error(monkeypatch, capsys):
    def no_champion():
        raise RegistryError("No champion alias")

    monkeypatch.setattr(monitoring, "get_champion", no_champion)
    assert monitoring.main([]) == 2
    out = capsys.readouterr().out
    assert "No champion" in out
    assert "RETRAIN" not in out


NOTE = "NOTE: this champion was trained on"


@pytest.mark.parametrize(
    ("params", "rows"),
    [
        ({"training_mode": "retrain", "n_validated_feedback_rows": "1995"}, 1995),  # A
        ({"n_validated_feedback_rows": "1995"}, None),  # B: ordinary training
        ({}, None),  # C: older run without either param
        ({"training_mode": "retrain"}, None),  # D: count missing
        ({"training_mode": "retrain", "n_validated_feedback_rows": "lots"}, None),  # E: malformed
        ({"training_mode": "retrain", "n_validated_feedback_rows": "0"}, None),
    ],
)
def test_in_sample_feedback_rows(params, rows):
    assert monitoring.in_sample_feedback_rows(params) == rows


def test_a_retrained_champion_gets_the_note(fake_champion_and_data, capsys):
    fake_champion_and_data.update(training_mode="retrain", n_validated_feedback_rows="1995")
    assert monitoring.main([]) == 0
    out = capsys.readouterr().out
    assert f"{NOTE} 1995 rows of the future batch" in out
    assert out.index(NOTE) < out.index("context    rows")  # directly above the performance table


@pytest.mark.parametrize(
    "params",
    [{"training_mode": "train"}, {}, {"training_mode": "retrain"}, {"training_mode": "retrain", "n_validated_feedback_rows": "x"}],
)
def test_b_c_d_e_no_note_and_monitoring_still_succeeds(fake_champion_and_data, capsys, params):
    fake_champion_and_data.update(params)
    assert monitoring.main([]) == 0
    assert NOTE not in capsys.readouterr().out


@pytest.mark.parametrize("simulate", [[], ["--simulate-shift"]])
def test_f_g_note_changes_neither_exit_code_nor_drift(fake_champion_and_data, capsys, simulate):
    def drift_section(out: str) -> str:
        return out[out.index("Data drift"):]

    code_without = monitoring.main(simulate)
    out_without = capsys.readouterr().out
    fake_champion_and_data.update(training_mode="retrain", n_validated_feedback_rows="1995")
    code_with = monitoring.main(simulate)
    out_with = capsys.readouterr().out

    assert NOTE in out_with and NOTE not in out_without
    assert code_with == code_without
    assert drift_section(out_with) == drift_section(out_without)
