"""Log a deliberately weak candidate (prior-only DummyClassifier) to demonstrate that the gate rejects it."""

import mlflow
import mlflow.sklearn
from sklearn.dummy import DummyClassifier

from ai4i_mlops.config import MODEL_NAME, PREDICTION_THRESHOLD, SEED
from ai4i_mlops.contexts import CONTEXTS
from ai4i_mlops.data import clean, load_raw, split
from ai4i_mlops.evaluation import evaluate_slices, format_slices
from ai4i_mlops.pipeline import build_pipeline, predict_failure_proba
from ai4i_mlops.tracking import get_or_create_experiment_id

X, y = clean(load_raw())
splits = split(X, y)  # the "future" split stays unused, as in training
X_train, y_train = splits["train"]
X_test, y_test = splits["test"]

# Same preprocessing as the real model; only the classifier step is swapped.
model = build_pipeline().set_params(model=DummyClassifier(strategy="prior")).fit(X_train, y_train)
results = evaluate_slices(X_test, y_test, predict_failure_proba(model, X_test))

with mlflow.start_run(experiment_id=get_or_create_experiment_id(), tags={"gate_demo": "weak_candidate"}) as run:
    mlflow.log_params(
        {
            "classifier": "DummyClassifier",
            "strategy": "prior",
            "seed": SEED,
            "prediction_threshold": PREDICTION_THRESHOLD,
            "n_train_rows": len(X_train),
        }
    )
    mlflow.log_metrics({name: value for name, value in results.items() if value is not None})
    mlflow.sklearn.log_model(model, name=MODEL_NAME, input_example=X_train.head(5))

print(f"weak candidate run {run.info.run_id}")
print(f"pooled recall {results['test_all_recall']}  pooled AP {results['test_all_average_precision']}")
print(format_slices(results, CONTEXTS))
