import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.validation import check_is_fitted

from ai4i_mlops.data import CATEGORICAL, NUMERIC
from ai4i_mlops.pipeline import build_pipeline, predict_failure_proba


@pytest.fixture
def xy() -> tuple[pd.DataFrame, pd.Series]:
    n = 12
    X = pd.DataFrame(
        {
            "Type": ["L", "M", "H"] * 4,
            "Air temperature": [298.0 + i * 0.1 for i in range(n)],
            "Process temperature": [308.0 + i * 0.1 for i in range(n)],
            "Rotational speed": [1400 + 10 * i for i in range(n)],
            "Torque": [40.0 + i for i in range(n)],
            "Tool wear": [5 * i for i in range(n)],
        }
    )
    y = pd.Series([0, 1] * 6)
    return X, y


def test_build_pipeline_is_unfitted_and_fresh():
    pipe = build_pipeline()
    assert isinstance(pipe, Pipeline)
    with pytest.raises(NotFittedError):
        check_is_fitted(pipe)
    assert build_pipeline() is not pipe


def test_fit_and_predict_proba(xy):
    X, y = xy
    pipe = build_pipeline().fit(X, y)
    proba = pipe.predict_proba(X)
    assert proba.shape == (len(X), 2)

    failure = predict_failure_proba(pipe, X)
    assert list(failure.index) == list(X.index)
    assert failure.between(0, 1).all()


def test_type_is_one_hot_encoded(xy):
    X, y = xy
    pipe = build_pipeline().fit(X, y)
    encoder = pipe.named_steps["preprocess"].named_transformers_["categorical"]
    assert isinstance(encoder, OneHotEncoder)
    assert encoder.feature_names_in_.tolist() == CATEGORICAL
    assert sorted(encoder.categories_[0]) == ["H", "L", "M"]
    assert pipe.named_steps["preprocess"].transformers_[1][2] == NUMERIC


def test_unseen_type_does_not_raise(xy):
    X, y = xy
    pipe = build_pipeline().fit(X, y)
    unseen = X.head(2).assign(Type="Z")
    assert len(predict_failure_proba(pipe, unseen)) == 2
