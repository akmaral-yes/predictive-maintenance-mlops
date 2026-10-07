from typing import Final

import pandas as pd

# Evaluation/monitoring slices of the one pooled model. Never a reason to train per-Type models.
CONTEXTS: Final[tuple[str, ...]] = ("all", "L", "M", "H")


def select_context(X: pd.DataFrame, y: pd.Series, context: str) -> tuple[pd.DataFrame, pd.Series]:
    """Return the rows of X and y for one context, keeping the original index."""
    if context not in CONTEXTS:
        raise ValueError(f"Unknown context {context!r}; valid contexts: {', '.join(CONTEXTS)}")
    if context == "all":
        return X, y
    mask = X["Type"] == context
    return X.loc[mask], y.loc[mask]
