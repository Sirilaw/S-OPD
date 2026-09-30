import sys
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from rescore_mathverse_mcq import repair_dataframe  # noqa: E402


def test_repair_dataframe_is_deterministic_and_preserves_legacy_fields():
    data = pd.DataFrame(
        [
            {
                "question_type": "multi-choice",
                "answer": "B",
                "prediction": "Reasoning. Final answer: B",
                "extract": "demo A\ndemo C\nB",
                "log_extract": "legacy extraction",
                "extract_source": "judge",
                "extract_judge_responses": [["raw extraction response"]],
                "score": False,
                "log_score": "legacy score",
                "score_source": "judge",
                "score_judge_responses": [["Judgement: 0"]],
            },
            {
                "question_type": "multi-choice",
                "answer": "C",
                "prediction": "Either A or C",
                "extract": "C",
                "log_extract": "legacy extraction",
                "extract_source": "judge",
                "extract_judge_responses": [["C"]],
                "score": True,
                "log_score": "legacy score",
                "score_source": "judge",
                "score_judge_responses": [["1"]],
            },
            {
                "question_type": "free-form",
                "answer": "42",
                "prediction": "42",
                "extract": "42",
                "log_extract": "legacy extraction",
                "extract_source": "judge",
                "extract_judge_responses": [["42"]],
                "score": True,
                "log_score": "legacy score",
                "score_source": "judge",
                "score_judge_responses": [["1"]],
            },
        ]
    )

    repaired, summary = repair_dataframe(data)

    assert repaired.loc[0, "extract"] == "B"
    assert bool(repaired.loc[0, "score"]) is True
    assert repaired.loc[0, "legacy_extract"] == "demo A\ndemo C\nB"
    assert bool(repaired.loc[0, "legacy_score"]) is False
    assert repaired.loc[0, "extract_judge_responses"] == [["raw extraction response"]]
    assert repaired.loc[0, "score_judge_responses"] == [["Judgement: 0"]]

    assert repaired.loc[1, "extract"] == "C"
    assert repaired.loc[1, "extract_source"] == "judge"
    assert bool(repaired.loc[1, "mcqfix_applied"]) is False
    assert repaired.loc[2, "score_source"] == "judge"

    assert summary["deterministically_rescored_rows"] == 1
    assert summary["ambiguous_single_choice_rows_retained"] == 1
    assert summary["false_negatives_fixed"] == 1
    assert summary["false_positives_fixed"] == 0
    assert summary["legacy_correct"] == 2
    assert summary["repaired_correct"] == 3
