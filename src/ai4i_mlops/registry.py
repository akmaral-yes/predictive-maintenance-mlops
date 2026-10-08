"""Registry versions and aliases: register a run's model as candidate, promote through the gate, roll back.

training run -> logged model -> registered model version; aliases point at versions:
  candidate = version proposed for promotion, champion = version currently approved.
"""

import argparse
from collections.abc import Sequence

import mlflow
from mlflow import MlflowClient
from mlflow.entities.model_registry import ModelVersion
from mlflow.exceptions import MlflowException

from ai4i_mlops.config import MODEL_NAME, REGISTERED_MODEL_NAME
from ai4i_mlops.gate import (
    evaluate_gate,
    format_report,
    load_pooled_metrics,
    load_run_metrics,
)
from ai4i_mlops.tracking import configure_tracking

CANDIDATE = "candidate"
CHAMPION = "champion"
GATE_TAG = "gate_result"

EXIT_OK = 0
EXIT_GATE_FAILED = 1
EXIT_ERROR = 2


class RegistryError(Exception):
    """The command could not run as asked (missing alias, unknown version, ...)."""


def _client() -> MlflowClient:
    configure_tracking()
    return MlflowClient()


def _aliases() -> dict[str, str]:
    try:
        aliases = _client().get_registered_model(REGISTERED_MODEL_NAME).aliases
    except MlflowException as e:
        if e.error_code == "RESOURCE_DOES_NOT_EXIST":
            return {}
        raise
    # The SQLite store returns version numbers as int; keep them as str throughout.
    return {alias: str(version) for alias, version in aliases.items()}


def get_version(version: str) -> ModelVersion:
    try:
        return _client().get_model_version(REGISTERED_MODEL_NAME, version)
    except MlflowException as e:
        raise RegistryError(f"Version {version} of {REGISTERED_MODEL_NAME!r} does not exist") from e


def get_alias_version(alias: str) -> ModelVersion | None:
    version = _aliases().get(alias)
    return None if version is None else get_version(version)


def set_alias(alias: str, version: str) -> None:
    _client().set_registered_model_alias(REGISTERED_MODEL_NAME, alias, version)


def source_run_id(version: ModelVersion) -> str:
    """The training run that produced this registered version; its metrics are the version's metrics."""
    if not version.run_id:
        raise RegistryError(f"Version {version.version} has no source run")
    return version.run_id


def get_champion() -> tuple[ModelVersion, str]:
    """The approved version and the training run that produced it."""
    champion = get_alias_version(CHAMPION)
    if champion is None:
        raise RegistryError(f"No {CHAMPION} alias on {REGISTERED_MODEL_NAME!r}; promote a candidate first")
    return champion, source_run_id(champion)


def register_run_model(run_id: str) -> ModelVersion:
    """Register the model logged by this training run as a new version and point candidate at it."""
    client = _client()
    try:
        experiment_id = client.get_run(run_id).info.experiment_id
    except MlflowException as e:
        raise RegistryError(f"Unknown run {run_id}") from e
    # No registered model yet means no versions, so nothing to compare against.
    for existing in client.search_model_versions(f"name = '{REGISTERED_MODEL_NAME}'"):
        if existing.run_id == run_id:
            raise RegistryError(
                f"Run {run_id} is already registered as version {existing.version}; "
                "train a new run to propose a new model"
            )
    logged = client.search_logged_models(
        experiment_ids=[experiment_id],
        filter_string=f"source_run_id = '{run_id}' AND name = '{MODEL_NAME}'",
    )
    if len(logged) != 1:
        raise RegistryError(f"Run {run_id} has {len(logged)} logged models named {MODEL_NAME!r}, expected 1")

    version = mlflow.register_model(f"models:/{logged[0].model_id}", REGISTERED_MODEL_NAME)
    set_alias(CANDIDATE, version.version)
    return version


def _describe(version: ModelVersion | None) -> str:
    return "none" if version is None else f"version {version.version} (run {version.run_id})"


def promote_candidate() -> bool:
    """Gate the candidate version against the champion version; on PASS move champion and consume candidate."""
    candidate = get_alias_version(CANDIDATE)
    if candidate is None:
        raise RegistryError("No candidate alias; register a run first")
    champion = get_alias_version(CHAMPION)
    if champion is not None and str(champion.version) == str(candidate.version):
        raise RegistryError(f"Candidate version {candidate.version} is already the champion")

    candidate_run = source_run_id(candidate)
    candidate_metrics = load_pooled_metrics(candidate_run)
    champion_metrics = None if champion is None else load_pooled_metrics(source_run_id(champion))
    result = evaluate_gate(candidate_metrics, champion_metrics)

    print(f"candidate: {_describe(candidate)}")
    print(f"champion:  {_describe(champion)}")
    diagnostics = {**load_run_metrics(candidate_run), **candidate_metrics}
    print(format_report(diagnostics, champion_metrics, result))

    client = _client()
    client.set_model_version_tag(REGISTERED_MODEL_NAME, candidate.version, GATE_TAG, "PASS" if result.passed else "FAIL")
    old = "none" if champion is None else champion.version
    if not result.passed:
        print(f"champion unchanged: {old}; candidate stays on version {candidate.version} for inspection")
        return False

    set_alias(CHAMPION, candidate.version)
    client.delete_registered_model_alias(REGISTERED_MODEL_NAME, CANDIDATE)
    print(f"champion: {old} -> {candidate.version}")
    if champion is not None:
        print("to roll back:")
        print(f"uv run python -m ai4i_mlops.registry rollback --to-version {old}")
    return True


def rollback_champion(target_version: str) -> None:
    """Move champion to an existing version that passed the gate. Nothing is retrained, re-registered or deleted."""
    champion = get_alias_version(CHAMPION)
    if champion is None:
        raise RegistryError("No champion to roll back")
    target = get_version(target_version)
    if str(target.version) == str(champion.version):
        raise RegistryError(f"Version {target.version} is already the champion")
    gate_result = target.tags.get(GATE_TAG)
    if gate_result != "PASS":
        found = "no gate result" if gate_result is None else f"{GATE_TAG}={gate_result}"
        raise RegistryError(
            f"Version {target.version} has {found}; rollback accepts only versions with {GATE_TAG}=PASS"
        )
    set_alias(CHAMPION, target.version)
    print(f"champion: {champion.version} -> {target.version}")


def print_status() -> None:
    print(f"registered model: {REGISTERED_MODEL_NAME}")
    for alias in (CANDIDATE, CHAMPION):
        version = get_alias_version(alias)
        if version is None:
            print(f"{alias + ':':<11}none")
        else:
            gate_result = version.tags.get(GATE_TAG, "not gated")
            print(f"{alias + ':':<11}version {version.version}  run {version.run_id}  {GATE_TAG}={gate_result}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Model Registry: register, promote, roll back, status.")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("register", help="register a run's logged model and set candidate").add_argument(
        "--run-id", required=True
    )
    commands.add_parser("promote", help="gate candidate against champion; promote on PASS")
    commands.add_parser("rollback", help="move champion to an existing version").add_argument(
        "--to-version", required=True, type=int
    )
    commands.add_parser("status", help="show candidate and champion")
    args = parser.parse_args(argv)

    try:
        if args.command == "register":
            previous = get_alias_version(CANDIDATE)
            version = register_run_model(args.run_id)
            print(f"registered version {version.version} of {REGISTERED_MODEL_NAME} from run {args.run_id}")
            print(f"candidate: {'none' if previous is None else previous.version} -> {version.version}")
        elif args.command == "promote":
            if not promote_candidate():
                return EXIT_GATE_FAILED
        elif args.command == "rollback":
            rollback_champion(str(args.to_version))
        else:
            print_status()
    except (RegistryError, MlflowException, ValueError) as e:
        print(f"error: {e}")
        return EXIT_ERROR
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
