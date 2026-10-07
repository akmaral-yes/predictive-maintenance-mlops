from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "raw" / "ai4i2020.csv"

TARGET = "Machine failure"
IDS = ["UID", "Product ID"]
LEAKAGE = ["TWF", "HDF", "PWF", "OSF", "RNF"]
NUMERIC = ["Air temperature", "Process temperature", "Rotational speed", "Torque", "Tool wear"]
CATEGORICAL = ["Type"]


def load_raw(path: Path = DATA_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Return features (numeric + Type) and a 1-D 0/1 target. IDs and leakage columns are dropped."""
    needed = set(NUMERIC + CATEGORICAL + [TARGET] + LEAKAGE + IDS)
    missing = needed - set(df.columns)
    if missing:
        raise KeyError(f"Columns not found in CSV: {sorted(missing)}")
    X = df[CATEGORICAL + NUMERIC].copy()
    y = df[TARGET].astype(int)
    return X, y


def split(X, y, seed: int = 42) -> dict[str, tuple[pd.DataFrame, pd.Series]]:
    """60/20/20 split into train / future (simulated new labels) / test, stratified on Type x failure."""
    strata = X["Type"] + "_" + y.astype(str)
    X_train, X_rest, y_train, y_rest = train_test_split(
        X, y, test_size=0.4, stratify=strata, random_state=seed
    )
    X_future, X_test, y_future, y_test = train_test_split(
        X_rest, y_rest, test_size=0.5, stratify=strata.loc[X_rest.index], random_state=seed
    )
    return {
        "train": (X_train, y_train),
        "future": (X_future, y_future),
        "test": (X_test, y_test),
    }


def report_counts(splits) -> None:
    for name, (X, y) in splits.items():
        print(f"\n{name}: {len(y)} rows, {y.mean():.3%} failures")
        print(y.groupby(X["Type"]).agg(rows="size", failures="sum"))


if __name__ == "__main__":
    X, y = clean(load_raw())
    report_counts(split(X, y))