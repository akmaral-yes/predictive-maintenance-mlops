import mlflow
import mlflow.sklearn
import pytest
from sklearn.dummy import DummyClassifier

from ai4i_mlops import registry, tracking
from ai4i_mlops.config import MODEL_NAME, REGISTERED_MODEL_NAME
from ai4i_mlops.gate import AP_KEY, RECALL_KEY
from ai4i_mlops.registry import CANDIDATE, CHAMPION, GATE_TAG, RegistryError

GOOD = (0.66, 0.74)


@pytest.fixture(autouse=True)
def local_store(tmp_path, monkeypatch):
    """Point every MLflow call at a throwaway store so the real mlflow.db is never touched."""
    previous_uri = mlflow.get_tracking_uri()
    monkeypatch.setattr(tracking, "MLFLOW_TRACKING_URI", f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}")
    monkeypatch.setattr(tracking, "MLFLOW_ARTIFACT_ROOT", tmp_path / "mlruns")
    yield
    mlflow.set_tracking_uri(previous_uri)


def _log_run(recall: float, ap: float) -> str:
    """A training run with pooled metrics and a tiny logged model."""
    with mlflow.start_run(experiment_id=tracking.get_or_create_experiment_id()) as run:
        mlflow.log_metrics({RECALL_KEY: recall, AP_KEY: ap})
        model = DummyClassifier(strategy="prior").fit([[0], [1]], [0, 1])
        mlflow.sklearn.log_model(model, name=MODEL_NAME, pip_requirements=["scikit-learn"])
    return run.info.run_id


def _register(recall: float, ap: float) -> str:
    return str(registry.register_run_model(_log_run(recall, ap)).version)


def _champion(recall: float = GOOD[0], ap: float = GOOD[1]) -> str:
    """Bootstrap a champion through the normal register -> promote path."""
    version = _register(recall, ap)
    assert registry.promote_candidate()
    return version


def _alias(alias: str) -> str | None:
    version = registry.get_alias_version(alias)
    return None if version is None else str(version.version)


def _gate_tag(version: str) -> str | None:
    return registry.get_version(version).tags.get(GATE_TAG)


def test_a_b_registration_creates_version_and_sets_candidate():
    run_id = _log_run(*GOOD)
    version = registry.register_run_model(run_id)
    assert registry.get_version(version.version).run_id == run_id
    assert _alias(CANDIDATE) == str(version.version)
    assert _alias(CHAMPION) is None


def test_c_successful_promotion_moves_champion_and_consumes_candidate():
    _champion()
    b = _register(0.70, 0.75)
    assert registry.promote_candidate()
    assert _alias(CHAMPION) == b
    assert _alias(CANDIDATE) is None
    assert _gate_tag(b) == "PASS"


def test_d_failed_promotion_leaves_champion_and_candidate():
    a = _champion()
    b = _register(0.70, 0.40)
    assert not registry.promote_candidate()
    assert _alias(CHAMPION) == a
    assert _alias(CANDIDATE) == b
    assert _gate_tag(b) == "FAIL"


def test_e_candidate_already_champion_is_an_error():
    a = _champion()
    registry.set_alias(CANDIDATE, a)
    with pytest.raises(RegistryError, match="already the champion"):
        registry.promote_candidate()


def test_f_no_candidate_is_an_error():
    with pytest.raises(RegistryError, match="No candidate"):
        registry.promote_candidate()


def test_g_bootstrap_candidate_becomes_first_champion():
    a = _register(*GOOD)
    assert registry.promote_candidate()
    assert _alias(CHAMPION) == a
    assert _alias(CANDIDATE) is None
    assert _gate_tag(a) == "PASS"


def test_h_bootstrap_candidate_failing_recall_creates_no_champion():
    a = _register(0.30, 0.74)
    assert not registry.promote_candidate()
    assert _alias(CHAMPION) is None
    assert _alias(CANDIDATE) == a
    assert _gate_tag(a) == "FAIL"


def test_i_rollback_moves_champion_and_keeps_versions():
    a = _champion()
    b = _register(*GOOD)
    assert registry.promote_candidate()
    registry.rollback_champion(a)
    assert _alias(CHAMPION) == a
    assert registry.get_version(a) and registry.get_version(b)
    assert _gate_tag(b) == "PASS"


def test_rollback_to_failed_version_is_rejected():
    a = _champion()
    b = _register(0.70, 0.40)
    assert not registry.promote_candidate()
    with pytest.raises(RegistryError, match="gate_result=FAIL"):
        registry.rollback_champion(b)
    assert _alias(CHAMPION) == a


def test_rollback_to_ungated_version_is_rejected():
    a = _champion()
    b = _register(*GOOD)
    with pytest.raises(RegistryError, match="no gate result"):
        registry.rollback_champion(b)
    assert _alias(CHAMPION) == a


def test_registering_same_run_twice_creates_no_second_version():
    run_id = _log_run(*GOOD)
    first = registry.register_run_model(run_id)
    with pytest.raises(RegistryError, match=f"already registered as version {first.version}"):
        registry.register_run_model(run_id)
    assert len(mlflow.MlflowClient().search_model_versions(f"name = '{REGISTERED_MODEL_NAME}'")) == 1
    assert registry.main(["register", "--run-id", run_id]) == registry.EXIT_ERROR


def test_j_rollback_to_unknown_version_is_an_error():
    _champion()
    with pytest.raises(RegistryError, match="does not exist"):
        registry.rollback_champion("99")


def test_rollback_without_champion_is_an_error():
    a = _register(*GOOD)
    with pytest.raises(RegistryError, match="No champion"):
        registry.rollback_champion(a)


def test_k_rollback_to_current_champion_is_an_error():
    a = _champion()
    with pytest.raises(RegistryError, match="already the champion"):
        registry.rollback_champion(a)


def test_l_unregistered_newer_run_has_no_influence(capsys):
    a = _champion(*GOOD)
    b = _register(*GOOD)
    # Newest run, never registered. If its AP 0.99 were read as the champion baseline, B would fail.
    newer = _log_run(0.95, 0.99)
    capsys.readouterr()

    assert registry.promote_candidate()
    out = capsys.readouterr().out
    assert newer not in out
    assert registry.get_version(a).run_id in out
    assert _alias(CHAMPION) == b


def test_m_cli_exit_codes():
    good_run, weak_run = _log_run(*GOOD), _log_run(0.70, 0.30)
    assert registry.main(["promote"]) == registry.EXIT_ERROR  # no candidate
    assert registry.main(["register", "--run-id", "no-such-run"]) == registry.EXIT_ERROR
    assert registry.main(["register", "--run-id", good_run]) == registry.EXIT_OK
    assert registry.main(["promote"]) == registry.EXIT_OK
    assert registry.main(["register", "--run-id", weak_run]) == registry.EXIT_OK
    assert registry.main(["promote"]) == registry.EXIT_GATE_FAILED
    assert registry.main(["rollback", "--to-version", "99"]) == registry.EXIT_ERROR
    assert registry.main(["status"]) == registry.EXIT_OK
    assert (registry.EXIT_OK, registry.EXIT_GATE_FAILED, registry.EXIT_ERROR) == (0, 1, 2)


def test_registered_model_name_is_used():
    _register(*GOOD)
    assert registry.get_alias_version(CANDIDATE).name == REGISTERED_MODEL_NAME


def test_get_champion_returns_version_and_source_run():
    with pytest.raises(RegistryError, match="No champion"):
        registry.get_champion()
    run_id = _log_run(*GOOD)
    registry.register_run_model(run_id)
    assert registry.promote_candidate()
    version, source_run = registry.get_champion()
    assert (str(version.version), source_run) == ("1", run_id)
