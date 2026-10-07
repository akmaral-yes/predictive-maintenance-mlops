import pandas as pd
import pytest

from ai4i_mlops.contexts import CONTEXTS, select_context


@pytest.fixture
def xy() -> tuple[pd.DataFrame, pd.Series]:
    # Non-default index so that resetting it would be visible.
    index = [10, 11, 12, 13, 14]
    X = pd.DataFrame(
        {"Type": ["L", "M", "L", "H", "M"], "Torque": [40.0, 41.0, 42.0, 43.0, 44.0]},
        index=index,
    )
    y = pd.Series([0, 1, 0, 1, 0], index=index, name="Machine failure")
    return X, y


def test_contexts_are_explicit():
    assert CONTEXTS == ("all", "L", "M", "H")


def test_all_returns_everything(xy):
    X, y = xy
    Xs, ys = select_context(X, y, "all")
    pd.testing.assert_frame_equal(Xs, X)
    pd.testing.assert_series_equal(ys, y)


@pytest.mark.parametrize(
    ("context", "expected_index"),
    [("L", [10, 12]), ("M", [11, 14]), ("H", [13])],
)
def test_type_context_filters_rows(xy, context, expected_index):
    X, y = xy
    Xs, ys = select_context(X, y, context)
    assert (Xs["Type"] == context).all()
    assert list(Xs.index) == list(ys.index) == expected_index


def test_slices_stay_aligned_with_original_index(xy):
    X, y = xy
    Xs, ys = select_context(X, y, "M")
    assert list(Xs.index) == list(ys.index) == [11, 14]
    assert list(ys) == [1, 0]
    assert list(Xs["Torque"]) == [41.0, 44.0]


def test_valid_context_without_rows_is_empty(xy):
    X, y = xy
    no_h = X["Type"] != "H"
    Xs, ys = select_context(X[no_h], y[no_h], "H")
    assert Xs.empty and ys.empty
    assert list(Xs.columns) == list(X.columns)


def test_inputs_are_not_mutated(xy):
    X, y = xy
    X_before, y_before = X.copy(), y.copy()
    for context in CONTEXTS:
        select_context(X, y, context)
    pd.testing.assert_frame_equal(X, X_before)
    pd.testing.assert_series_equal(y, y_before)


def test_unknown_context_raises_with_valid_list(xy):
    X, y = xy
    with pytest.raises(ValueError) as exc:
        select_context(X, y, "X")
    for context in CONTEXTS:
        assert context in str(exc.value)
