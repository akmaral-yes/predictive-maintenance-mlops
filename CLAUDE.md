# predictive-maintenance-mlops

Gated MLOps lifecycle on the AI4I 2020 dataset: one global failure model, an
evaluation gate, an MLflow registry with rollback, a simulated-drift check,
reviewed retraining, and CI. Portfolio project; keep it small and explainable.

## Setup and commands
- Python 3.13, managed with uv. Run everything through uv.
- Package lives in src/ai4i_mlops/ (installed via hatchling). Run modules with
  `uv run python -m ai4i_mlops.<module>`.
- Checks: `uv run ruff check .` and `uv run pytest`. Both must pass before a commit.
- MLflow state (mlflow.db, mlruns/) is local and gitignored.

## Data contract
- Source file: data/raw/ai4i2020.csv. Never read from the network after the one-off
  fetch in scripts/fetch_once.py. Tests and CI never need the network.
- Target: `Machine failure` (~3.4% positive, 339 of 10,000 rows).
- Features: Air temperature, Process temperature, Rotational speed, Torque,
  Tool wear, plus Type (L, M, H), one-hot encoded inside the sklearn Pipeline.
- Dropped columns: UID, Product ID (identifiers) and TWF, HDF, PWF, OSF, RNF.
  The five mode columns encode the target (label leakage) and must never reach X.
- Type H has only 21 failures in total, so Type is an evaluation and monitoring
  slice only.

## Modeling rules
- One pooled model. Never train separate per-Type models. Never gate per Type.
- Imbalance: report recall, precision, F1 and average precision (AP). Never
  report accuracy. RandomForest uses class_weight="balanced" and seed 42.
- Splits are 60/20/20 (train / future / test), stratified on Type x failure.
  The test split is fixed and is never trained on.
- Settings (seed, threshold, model params, MLflow location) live in config.py.

## Lifecycle rules
- Gate: promote only if pooled test recall >= R_MIN and candidate AP >=
  champion AP - AP_MARGIN. The no-regression check is on AP, not recall (a model
  that predicts failure everywhere has recall 1.0). Thresholds are set in
  Phase 4 from real numbers; do not invent them before then.
- Registry: candidate and champion are MLflow aliases. Rollback is an alias move.
- CI never uses MLflow. It trains a fresh candidate and compares it to
  config/baseline_metrics.json.
- Drift (simulated sensor shift on a held-out batch) means investigate, never
  automatic retrain. The L/M/H difference is a subgroup difference, not drift.
- Feedback is a signal, not ground truth: nothing enters training unless the
  review step validated it.
- No time series, no paid APIs, no cloud.

## Phases
1 data contract (done) ·
2 config + contexts (done) ·
3 pipeline + metrics + training + MLflow (done) ·
4 gate ·
5 registry + rollback ·
6 monitoring + drift ·
7 feedback + retrain ·
8 API + CI + Docker

## Working style
- Do only the phase I ask for. Do not scaffold later phases or empty files.
- Small modules, type hints, no dead code, tests for pure logic.
- If a name, number or file differs from what is written here, stop and ask
  instead of guessing.
