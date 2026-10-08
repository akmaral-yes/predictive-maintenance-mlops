import pandas as pd
import pytest

from ai4i_mlops.evaluation import compute_metrics, evaluate_slices

Y_TRUE = [1, 1, 1, 0, 0, 0]
Y_PROBA = [0.9, 0.6, 0.3, 0.7, 0.2, 0.1]
CASE_A = {
    "recall": 2 / 3,
    "precision": 2 / 3,
    "f1": 2 / 3,
    "average_precision": (1 + 2 / 3 + 3 / 4) / 3,
}


def _slices_input(types: list[str]) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    index = [100 + i for i in range(len(types))]
    X = pd.DataFrame({"Type": types}, index=index)
    return X, pd.Series(Y_TRUE, index=index), pd.Series(Y_PROBA, index=index)


def test_case_a_metrics():
    assert compute_metrics(Y_TRUE, Y_PROBA, threshold=0.5) == pytest.approx(CASE_A)


def test_case_b_slices():
    r = evaluate_slices(*_slices_input(["L", "L", "L", "H", "H", "H"]), threshold=0.5)

    assert (r["test_L_n_rows"], r["test_L_n_failures"]) == (3, 3)
    assert r["test_L_recall"] == pytest.approx(2 / 3)
    assert r["test_L_precision"] == pytest.approx(1.0)
    assert r["test_L_f1"] == pytest.approx(0.8)
    assert r["test_L_average_precision"] == pytest.approx(1.0)

    assert (r["test_H_n_rows"], r["test_H_n_failures"]) == (3, 0)
    assert r["test_H_recall"] is None
    assert r["test_H_average_precision"] is None
    assert r["test_H_precision"] == 0.0
    assert r["test_H_f1"] == 0.0

    assert (r["test_M_n_rows"], r["test_M_n_failures"]) == (0, 0)
    assert type(r["test_M_n_rows"]) is int and type(r["test_M_n_failures"]) is int
    for name in CASE_A:
        assert r[f"test_M_{name}"] is None

    assert {name: r[f"test_all_{name}"] for name in CASE_A} == pytest.approx(CASE_A)


def test_case_c_recall_alone_is_not_enough():
    # Predicting failure everywhere gives perfect recall but AP equal to the base rate.
    y_true = [1] * 5 + [0] * 95
    m = compute_metrics(y_true, [1.0] * 100, threshold=0.5)
    assert m["recall"] == 1.0
    assert m["average_precision"] < 0.1


def test_empty_input_gives_none():
    assert compute_metrics([], []) == dict.fromkeys(CASE_A)


def test_unequal_lengths_raise():
    with pytest.raises(ValueError, match="rows"):
        compute_metrics([1, 0], [0.5])


def test_threshold_is_inclusive():
    assert compute_metrics([1, 0], [0.5, 0.4], threshold=0.5)["recall"] == 1.0


def test_misaligned_proba_index_is_rejected():
    X, y, y_proba = _slices_input(["L", "L", "L", "H", "H", "H"])
    with pytest.raises(ValueError, match="same index"):
        evaluate_slices(X, y, y_proba.reset_index(drop=True))


def test_length_mismatch_is_rejected():
    X, y, y_proba = _slices_input(["L", "L", "L", "H", "H", "H"])
    with pytest.raises(ValueError, match="Length mismatch"):
        evaluate_slices(X, y, y_proba.iloc[:-1])


def test_slicing_keeps_probabilities_with_their_labels():
    # Interleaved Types: each slice must pick up its own rows' probabilities, not the first n.
    X, y, y_proba = _slices_input(["L", "H", "L", "H", "L", "H"])
    r = evaluate_slices(X, y, y_proba, threshold=0.5)
    # L rows: labels [1, 1, 0], probabilities [0.9, 0.3, 0.2]
    assert r["test_L_recall"] == pytest.approx(compute_metrics([1, 1, 0], [0.9, 0.3, 0.2])["recall"])
    assert r["test_L_recall"] == pytest.approx(0.5)
    # H rows: labels [1, 0, 0], probabilities [0.6, 0.7, 0.1]
    assert r["test_H_precision"] == pytest.approx(0.5)
    assert r["test_H_average_precision"] == pytest.approx(0.5)
