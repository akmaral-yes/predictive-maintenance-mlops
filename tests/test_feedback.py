import json

import pandas as pd
import pytest

from ai4i_mlops import feedback
from ai4i_mlops.data import NUMERIC
from ai4i_mlops.feedback import (
    FeedbackError,
    candidates_from_split,
    load_validated_feedback,
    parse_records,
    read_feedback,
    review,
    write_feedback,
)


def _future() -> tuple[pd.DataFrame, pd.Series]:
    index = [42, 7, 105, 300]  # original dataset indices, deliberately unsorted
    X = pd.DataFrame(
        {
            "Type": ["L", "M", "H", "L"],
            "Air temperature": [298.1, 299.2, 300.3, 301.4],
            "Process temperature": [308.5, 309.6, 310.7, 311.8],
            "Rotational speed": [1500, 1600, 1700, 1400],
            "Torque": [40.5, 35.25, 50.0, 60.75],
            "Tool wear": [0, 10, 200, 150],
        },
        index=index,
    )
    return X, pd.Series([0, 1, 0, 1], index=index, name="Machine failure")


def _row(**overrides) -> dict:
    row = {
        "feedback_id": "fb-00001",
        "source_index": 1,
        "Type": "L",
        "Air temperature": 298.0,
        "Process temperature": 308.0,
        "Rotational speed": 1500,
        "Torque": 40.0,
        "Tool wear": 5,
        "label": 0,
        "review_status": "pending",
    }
    return {**row, **overrides}


def _lines(*rows: dict) -> list[str]:
    return [json.dumps(r) for r in rows]


@pytest.fixture
def paths(tmp_path, monkeypatch):
    candidates, validated = tmp_path / "feedback_candidates.jsonl", tmp_path / "validated_feedback.jsonl"
    monkeypatch.setattr(feedback, "FEEDBACK_CANDIDATES_PATH", candidates)
    monkeypatch.setattr(feedback, "VALIDATED_FEEDBACK_PATH", validated)
    write_feedback(candidates, candidates_from_split(*_future()))
    return candidates, validated


def test_a_generated_pending_candidates_parse(paths):
    candidates, _ = paths
    records = read_feedback(candidates)
    assert [r.source_index for r in records] == [7, 42, 105, 300]
    assert [r.feedback_id for r in records] == ["fb-00007", "fb-00042", "fb-00105", "fb-00300"]
    assert {r.review_status for r in records} == {"pending"}
    assert records[0].features["Torque"] == 35.25 and records[0].label == 1


def test_b_c_d_e_review_keeps_candidates_and_writes_only_approved(paths):
    candidates, validated = paths
    before = candidates.read_bytes()

    assert feedback.main(["--approve-all", "--reject", "fb-00042", "--reject", "fb-00300"]) == 0

    assert candidates.read_bytes() == before  # B: the candidate file is never modified
    written = read_feedback(validated)
    assert [r.feedback_id for r in written] == ["fb-00007", "fb-00105"]  # C, D
    assert {r.review_status for r in written} == {"approved"}  # E: nothing pending reaches it


def test_f_duplicate_feedback_id_is_rejected():
    with pytest.raises(FeedbackError, match="duplicate feedback_id"):
        parse_records(_lines(_row(), _row(source_index=2)))


def test_g_duplicate_source_index_is_rejected():
    with pytest.raises(FeedbackError, match="duplicate source_index"):
        parse_records(_lines(_row(), _row(feedback_id="fb-00002")))


def test_h_missing_feature_is_rejected():
    row = _row()
    del row["Torque"]
    with pytest.raises(FeedbackError, match=r"fb-00001.*missing features \['Torque'\]"):
        parse_records(_lines(row))


@pytest.mark.parametrize("label", [2, -1, "1", 0.5, True, None])
def test_i_invalid_label_is_rejected(label):
    with pytest.raises(FeedbackError, match="is not 0 or 1"):
        parse_records(_lines(_row(label=label)))


def test_j_invalid_review_status_is_rejected():
    with pytest.raises(FeedbackError, match="review_status 'maybe'"):
        parse_records(_lines(_row(review_status="maybe")))


def test_unexpected_field_and_bad_values_are_rejected():
    with pytest.raises(FeedbackError) as e:
        parse_records(_lines(_row(TWF=1, Type="X", Torque="high")))
    assert len(e.value.problems) == 3


def test_k_unknown_reject_id_is_an_error(paths, capsys):
    _, validated = paths
    assert feedback.main(["--approve-all", "--reject", "fb-99999"]) == 2
    assert "fb-99999" in capsys.readouterr().out
    assert not validated.exists()


def test_l_one_malformed_candidate_fails_the_whole_review(paths, capsys):
    candidates, validated = paths
    with candidates.open("a") as f:
        f.write(json.dumps(_row(feedback_id="fb-bad", source_index=999, label=7)) + "\n")
        f.write("{not json\n")
    assert feedback.main(["--approve-all"]) == 2
    out = capsys.readouterr().out
    assert "fb-bad" in out and "source_index=999" in out and "not valid JSON" in out
    assert not validated.exists()  # no partial file, no silently dropped rows


def test_review_requires_pending_candidates():
    records = parse_records(_lines(_row(review_status="approved")))
    with pytest.raises(FeedbackError, match="only pending"):
        review(records, [])


def test_m_n_loader_returns_approved_rows_indexed_by_source_index(paths):
    _, validated = paths
    assert feedback.main(["--approve-all", "--reject", "fb-00105"]) == 0
    X, y = load_validated_feedback(validated)
    assert list(X.index) == list(y.index) == [7, 42, 300]
    assert list(X.columns) == ["Type", *NUMERIC]
    assert y.loc[7] == 1 and X.loc[300, "Tool wear"] == 150


@pytest.mark.parametrize("status", ["pending", "rejected"])
def test_loader_refuses_non_approved_rows(tmp_path, status):
    path = tmp_path / "validated.jsonl"
    path.write_text("\n".join(_lines(_row(review_status="approved"), _row(feedback_id="x", source_index=2, review_status=status))) + "\n")
    with pytest.raises(FeedbackError, match="non-approved"):
        load_validated_feedback(path)


def test_loader_refuses_missing_and_empty_files(tmp_path):
    with pytest.raises(FeedbackError, match="does not exist"):
        load_validated_feedback(tmp_path / "absent.jsonl")
    empty = tmp_path / "empty.jsonl"
    empty.write_text("")
    with pytest.raises(FeedbackError, match="no records"):
        load_validated_feedback(empty)


@pytest.fixture
def stale(paths):
    """An older validated file from an earlier successful review."""
    candidates, validated = paths
    assert feedback.main(["--approve-all"]) == 0
    assert validated.exists()
    return candidates, validated


def test_failed_review_removes_stale_validated_file(stale, capsys):
    candidates, validated = stale
    capsys.readouterr()
    candidates.write_text(candidates.read_text() + "{not json\n")  # A, D: malformed candidate input
    malformed = candidates.read_bytes()

    assert feedback.main(["--approve-all"]) == 2
    assert not validated.exists()
    assert "stale validated feedback was removed" in capsys.readouterr().out
    assert candidates.read_bytes() == malformed  # F: review never writes the candidate file


def test_failed_review_without_validated_file_reports_nothing_to_remove(paths, capsys):
    _, validated = paths
    assert feedback.main(["--approve-all", "--reject", "fb-99999"]) == 2  # B
    assert not validated.exists()
    assert "no validated feedback file existed to remove" in capsys.readouterr().out


def test_unknown_reject_id_removes_stale_validated_file(stale, capsys):
    _, validated = stale
    assert feedback.main(["--approve-all", "--reject", "fb-99999"]) == 2  # C
    assert not validated.exists()
    assert "stale validated feedback was removed" in capsys.readouterr().out


def test_missing_candidate_file_and_usage_error_remove_stale_file(stale):
    candidates, validated = stale
    assert feedback.main(["--approve-all", "--reject", "fb-00007"]) == 0
    with pytest.raises(SystemExit) as e:
        feedback.main([])  # --approve-all missing: usage error
    assert e.value.code == 2 and not validated.exists()

    assert feedback.main(["--approve-all"]) == 0
    candidates.rename(candidates.with_suffix(".moved"))
    assert feedback.main(["--approve-all"]) == 2
    assert not validated.exists()


def test_successful_review_after_failure_recreates_validated_file(stale):
    candidates, validated = stale
    before = candidates.read_bytes()
    assert feedback.main(["--approve-all", "--reject", "fb-99999"]) == 2
    assert not validated.exists()

    assert feedback.main(["--approve-all", "--reject", "fb-00042"]) == 0  # E
    assert [r.feedback_id for r in read_feedback(validated)] == ["fb-00007", "fb-00105", "fb-00300"]
    assert candidates.read_bytes() == before  # F
