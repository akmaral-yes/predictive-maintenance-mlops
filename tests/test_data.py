from ai4i_mlops.data import IDS, LEAKAGE, clean, load_raw, split


def _prepared():
    X, y = clean(load_raw())
    return X, y, split(X, y)


def test_dataset_matches_known_counts():
    df = load_raw()
    assert len(df) == 10_000
    assert df["Machine failure"].sum() == 339
    h = df[df["Type"] == "H"]
    assert len(h) == 1003
    assert h["Machine failure"].sum() == 21


def test_leakage_and_ids_absent():
    X, _, _ = _prepared()
    assert not set(LEAKAGE + IDS) & set(X.columns)
    assert "Type" in X.columns


def test_target_is_binary_1d():
    _, y, _ = _prepared()
    assert y.ndim == 1
    assert set(y.unique()) <= {0, 1}


def test_split_sizes_and_no_overlap():
    X, _, splits = _prepared()
    n = len(X)
    expected = {"train": 0.6, "future": 0.2, "test": 0.2}
    for name, (Xs, _) in splits.items():
        assert abs(len(Xs) - expected[name] * n) <= 2
    idx = [set(Xs.index) for Xs, _ in splits.values()]
    assert not (idx[0] & idx[1]) and not (idx[0] & idx[2]) and not (idx[1] & idx[2])


def test_every_split_has_all_types():
    _, _, splits = _prepared()
    for Xs, _ in splits.values():
        assert set(Xs["Type"]) == {"L", "M", "H"}


def test_every_split_has_type_h_failures():
    _, _, splits = _prepared()
    for Xs, ys in splits.values():
        assert ((Xs["Type"] == "H") & (ys == 1)).sum() >= 1