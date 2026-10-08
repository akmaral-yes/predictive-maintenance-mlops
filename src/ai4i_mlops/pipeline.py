import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from ai4i_mlops.config import RF_PARAMS
from ai4i_mlops.data import CATEGORICAL, NUMERIC


def build_pipeline() -> Pipeline:
    """Return a new, unfitted pooled pipeline: one-hot Type, raw numerics, RandomForest."""
    preprocess = ColumnTransformer(
        [
            ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
            ("numeric", "passthrough", NUMERIC),
        ]
    )
    return Pipeline([("preprocess", preprocess), ("model", RandomForestClassifier(**RF_PARAMS))])


def predict_failure_proba(model: Pipeline, X: pd.DataFrame) -> pd.Series:
    """Probability of the positive (failure) class, indexed like X."""
    positive = list(model.classes_).index(1)
    return pd.Series(model.predict_proba(X)[:, positive], index=X.index, name="failure_proba")
