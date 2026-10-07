import pandas as pd
from ucimlrepo import fetch_ucirepo

d = fetch_ucirepo(id=601)
print("features:", d.data.features.columns.tolist())
print("targets: ", d.data.targets.columns.tolist())
print("ids:     ", None if d.data.ids is None else d.data.ids.columns.tolist())

parts = [x for x in (d.data.ids, d.data.features, d.data.targets) if x is not None]
df = pd.concat(parts, axis=1)
print(df.shape, "failure rate:", df["Machine failure"].mean())

# the number the plan wants confirmed first
print(df.groupby("Type")["Machine failure"].agg(rows="size", failures="sum"))

df.to_csv("data/raw/ai4i2020.csv", index=False)