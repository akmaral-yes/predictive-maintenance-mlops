"""Simulate newly labelled production feedback: one pending candidate per future-split row.

Overwrites the generated candidate file deterministically. Train and fixed-test rows are never used.
"""

from ai4i_mlops.config import FEEDBACK_CANDIDATES_PATH
from ai4i_mlops.data import clean, load_raw, split
from ai4i_mlops.feedback import (
    EXIT_ERROR,
    EXIT_OK,
    FeedbackError,
    candidates_from_split,
    read_feedback,
    write_feedback,
)


def main() -> int:
    try:
        X_future, y_future = split(*clean(load_raw()))["future"]
        write_feedback(FEEDBACK_CANDIDATES_PATH, candidates_from_split(X_future, y_future))
        candidates = read_feedback(FEEDBACK_CANDIDATES_PATH)  # re-read so the written file is validated
    except (FeedbackError, OSError, KeyError) as e:
        print(f"error: {e}")
        return EXIT_ERROR
    print(f"wrote {len(candidates)} pending feedback candidates to {FEEDBACK_CANDIDATES_PATH}")
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
