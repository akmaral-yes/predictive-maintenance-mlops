"""Feedback records and the review boundary.

Feedback is a signal, not ground truth. Candidates (all pending) are the immutable record of what
arrived; review writes a separate file containing only approved records, and that file is the
only feedback retraining may read. Any malformed record stops the operation: nothing is dropped,
and a failed review removes any earlier validated file.
"""

import argparse
import json
import math
import os
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import pandas as pd

from ai4i_mlops.config import FEEDBACK_CANDIDATES_PATH, VALIDATED_FEEDBACK_PATH
from ai4i_mlops.data import CATEGORICAL, NUMERIC, TARGET

FEATURES = CATEGORICAL + NUMERIC
TYPES = ("L", "M", "H")
STATUSES = ("pending", "approved", "rejected")
FIELDS = ("feedback_id", "source_index", *FEATURES, "label", "review_status")

EXIT_OK = 0
EXIT_ERROR = 2


class FeedbackError(Exception):
    """Feedback could not be used as asked; carries every problem found."""

    def __init__(self, problems: Sequence[str]) -> None:
        self.problems = list(problems)
        super().__init__("; ".join(self.problems))


@dataclass(frozen=True)
class FeedbackRecord:
    feedback_id: str
    source_index: int
    features: dict[str, str | float | int]
    label: int
    review_status: str


def feedback_id_for(source_index: int) -> str:
    return f"fb-{source_index:05d}"


def _is_number(value: object) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(value)


def _record_problems(raw: dict) -> list[str]:
    problems = []
    if not isinstance(raw.get("feedback_id"), str) or not raw["feedback_id"]:
        problems.append("feedback_id missing or not a non-empty string")
    if type(raw.get("source_index")) is not int:
        problems.append("source_index missing or not an integer")
    missing = [f for f in FEATURES if f not in raw]
    if missing:
        problems.append(f"missing features {missing}")
    unexpected = sorted(set(raw) - set(FIELDS))
    if unexpected:
        problems.append(f"unexpected fields {unexpected}")
    if "Type" in raw and raw["Type"] not in TYPES:
        problems.append(f"Type {raw['Type']!r} not one of {TYPES}")
    bad_numeric = [f for f in NUMERIC if f in raw and not _is_number(raw[f])]
    if bad_numeric:
        problems.append(f"non-numeric values for {bad_numeric}")
    if type(raw.get("label")) is not int or raw["label"] not in (0, 1):
        problems.append(f"label {raw.get('label')!r} is not 0 or 1")
    if raw.get("review_status") not in STATUSES:
        problems.append(f"review_status {raw.get('review_status')!r} not one of {STATUSES}")
    return problems


def parse_records(lines: Iterable[str]) -> list[FeedbackRecord]:
    """Validate every line; raise FeedbackError listing all problems if any record is invalid."""
    records, problems = [], []
    seen_ids: set[str] = set()
    seen_indices: set[int] = set()
    for line_no, line in enumerate(lines, start=1):
        if not line.strip():
            problems.append(f"line {line_no}: empty line")
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as e:
            problems.append(f"line {line_no}: not valid JSON ({e.msg})")
            continue
        if not isinstance(raw, dict):
            problems.append(f"line {line_no}: not a JSON object")
            continue
        where = f"line {line_no} (feedback_id={raw.get('feedback_id')!r}, source_index={raw.get('source_index')!r})"
        record_problems = _record_problems(raw)
        if raw.get("feedback_id") in seen_ids:
            record_problems.append("duplicate feedback_id")
        if raw.get("source_index") in seen_indices:
            record_problems.append("duplicate source_index")
        if record_problems:
            problems += [f"{where}: {p}" for p in record_problems]
            continue
        seen_ids.add(raw["feedback_id"])
        seen_indices.add(raw["source_index"])
        records.append(
            FeedbackRecord(
                feedback_id=raw["feedback_id"],
                source_index=raw["source_index"],
                features={f: raw[f] for f in FEATURES},
                label=raw["label"],
                review_status=raw["review_status"],
            )
        )
    if problems:
        raise FeedbackError(problems)
    return records


def read_feedback(path: Path) -> list[FeedbackRecord]:
    if not path.exists():
        raise FeedbackError([f"{path} does not exist"])
    records = parse_records(path.read_text().splitlines())
    if not records:
        raise FeedbackError([f"{path} contains no records"])
    return records


def write_feedback(path: Path, records: Sequence[FeedbackRecord]) -> None:
    """Write all records or nothing: a temporary file replaces the target only when complete."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as f:
        for r in records:
            row = {"feedback_id": r.feedback_id, "source_index": r.source_index, **r.features}
            f.write(json.dumps({**row, "label": r.label, "review_status": r.review_status}) + "\n")
    os.replace(tmp, path)


def candidates_from_split(X_future: pd.DataFrame, y_future: pd.Series) -> list[FeedbackRecord]:
    """One pending candidate per future-split row, ordered by original dataset index."""
    records = []
    for source_index in sorted(X_future.index):
        row = X_future.loc[source_index]
        features = {f: row[f].item() if hasattr(row[f], "item") else row[f] for f in FEATURES}
        records.append(
            FeedbackRecord(
                feedback_id=feedback_id_for(int(source_index)),
                source_index=int(source_index),
                features=features,
                label=int(y_future.loc[source_index]),
                review_status="pending",
            )
        )
    return records


def review(candidates: Sequence[FeedbackRecord], reject_ids: Iterable[str]) -> tuple[list[FeedbackRecord], list[str]]:
    """Approve every pending candidate except the rejected ids. Returns (approved records, rejected ids)."""
    reject = set(reject_ids)
    unknown = sorted(reject - {c.feedback_id for c in candidates})
    if unknown:
        raise FeedbackError([f"unknown feedback_id(s) to reject: {unknown}"])
    not_pending = [c.feedback_id for c in candidates if c.review_status != "pending"]
    if not_pending:
        raise FeedbackError([f"candidate file must contain only pending records; not pending: {not_pending}"])
    approved = [replace(c, review_status="approved") for c in candidates if c.feedback_id not in reject]
    rejected = [c.feedback_id for c in candidates if c.feedback_id in reject]
    return approved, rejected


def load_validated_feedback(path: Path) -> tuple[pd.DataFrame, pd.Series]:
    """Approved feedback as (X, y) indexed by source_index. Anything else in the file is an error."""
    records = read_feedback(path)
    not_approved = [f"{r.feedback_id} ({r.review_status})" for r in records if r.review_status != "approved"]
    if not_approved:
        raise FeedbackError([f"validated feedback contains non-approved records: {not_approved}"])
    index = pd.Index([r.source_index for r in records])
    X = pd.DataFrame([r.features for r in records], index=index, columns=FEATURES)
    y = pd.Series([r.label for r in records], index=index, name=TARGET, dtype=int)
    return X, y


def _remove_validated_feedback() -> None:
    """After a failed review no validated file may remain, so stale approvals cannot reach retraining."""
    VALIDATED_FEEDBACK_PATH.with_suffix(VALIDATED_FEEDBACK_PATH.suffix + ".tmp").unlink(missing_ok=True)
    if VALIDATED_FEEDBACK_PATH.exists():
        VALIDATED_FEEDBACK_PATH.unlink()
        print("stale validated feedback was removed")
    else:
        print("no validated feedback file existed to remove")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Review feedback candidates into validated feedback.")
    parser.add_argument("--approve-all", action="store_true", required=True, help="approve every candidate not rejected")
    parser.add_argument("--reject", action="append", default=[], metavar="ID", help="feedback_id to reject (repeatable)")
    try:
        args = parser.parse_args(argv)
    except SystemExit as e:
        if e.code:  # a usage error is a failed review too; --help (code 0) is not
            _remove_validated_feedback()
        raise

    try:
        candidates = read_feedback(FEEDBACK_CANDIDATES_PATH)
        approved, rejected = review(candidates, args.reject)
        write_feedback(VALIDATED_FEEDBACK_PATH, approved)
    except (FeedbackError, OSError) as e:
        print("error: feedback review failed")
        for problem in getattr(e, "problems", [str(e)]):
            print(f"  {problem}")
        _remove_validated_feedback()
        return EXIT_ERROR

    print(f"total candidates  {len(candidates)}")
    print(f"approved          {len(approved)}")
    print(f"rejected          {len(rejected)}  {', '.join(rejected)}")
    print(f"validated file    {VALIDATED_FEEDBACK_PATH}")
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
