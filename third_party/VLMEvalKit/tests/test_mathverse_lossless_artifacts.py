import sys
from pathlib import Path

import pandas as pd
import pytest

VLMEVALKIT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(VLMEVALKIT_ROOT))

from vlmeval.dataset.image_vqa import MathVerse  # noqa: E402
from vlmeval.dataset.utils.mathverse import (  # noqa: E402
    MathVerse_auxeval_extract,
    MathVerse_auxeval_score,
    extract_mathverse_mcq_answer,
    is_mathverse_single_choice,
    normalize_mathverse_mcq_answer,
    parse_mathverse_judgement,
)
from vlmeval.smp.file import dump, get_intermediate_file_path, load  # noqa: E402


class StubJudge:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.temperatures = []

    def generate(self, prompt, temperature):
        del prompt
        self.temperatures.append(temperature)
        return next(self.responses)


@pytest.mark.parametrize(
    ("response", "expected"),
    [
        ("1", 1),
        ("Judgement: 0", 0),
        ("**Judgment: 1**\nExplanation: equivalent answers.", 1),
        ('{"judgement": 0, "reason": "different"}', 0),
        ("Judement is 1", 1),
        ("Judgement: 0\nCorrection: Judgement: 1", 1),
        ("The answers appear consistent.", None),
    ],
)
def test_parse_mathverse_judgement(response, expected):
    assert parse_mathverse_judgement(response) == expected


@pytest.mark.parametrize(
    ("response", "expected"),
    [
        ("A", "A"),
        ("B: 12 cm", "B"),
        ("The correct answer is **C**.", "C"),
        ("Therefore, the correct answer is:\n\n**A: $3/5$**", "A"),
        (r"work\n\boxed{\text{D}}", "D"),
        ("Answer: B\nCorrection: Final Answer: A", "A"),
        ("demo: A\ndemo: C\nD", "D"),
        ("A, C", None),
        ("This argument discusses option-like variables a and b.", None),
    ],
)
def test_extract_mathverse_mcq_answer(response, expected):
    assert extract_mathverse_mcq_answer(response) == expected


@pytest.mark.parametrize(
    ("answer", "expected"),
    [("A", "A"), ("(B)", "B"), (" [c] ", "C"), ("C\nD", None)],
)
def test_normalize_mathverse_mcq_answer(answer, expected):
    assert normalize_mathverse_mcq_answer(answer) == expected


def test_mathverse_single_choice_requires_valid_single_label():
    assert is_mathverse_single_choice({"question_type": "multi-choice", "answer": "D"})
    assert not is_mathverse_single_choice({"question_type": "multi-choice", "answer": "C\nD"})
    assert not is_mathverse_single_choice({"question_type": "free-form", "answer": "D"})


def test_mathverse_mcq_extract_bypasses_judge():
    judge = StubJudge([])
    line = {
        "question_type": "multi-choice",
        "answer": "B",
        "prediction": "After checking the diagram, final answer: B.",
    }

    result = MathVerse_auxeval_extract(judge, line)

    assert result["extract"] == "B"
    assert result["extract_source"] == "rule_mcq_prediction"
    assert result["extract_judge_responses"] == []
    assert judge.temperatures == []


def test_mathverse_mcq_score_bypasses_judge_and_normalizes_gold():
    judge = StubJudge([])
    line = {
        "question_type": "multi-choice",
        "answer": "(C)",
        "extract": "C",
    }

    result = MathVerse_auxeval_score(judge, line)

    assert result["score"] is True
    assert result["score_source"] == "rule_mcq_exact_match"
    assert result["score_judge_responses"] == []
    assert judge.temperatures == []


def test_mathverse_ambiguous_mcq_prediction_uses_judge_fallback():
    leaked_demos = "(-2,1)\nD\nDomain: (-3,3]\nnull\n22.3\nf(x)=-x^2-2x+1\nD"
    judge = StubJudge([leaked_demos])
    line = {
        "question_type": "multi-choice",
        "answer": "D",
        "prediction": "Either A or C.",
    }

    result = MathVerse_auxeval_extract(judge, line)

    assert result["extract"] == "D"
    assert result["extract_source"] == "judge_mcq"
    assert result["extract_judge_responses"] == [leaked_demos]
    assert judge.temperatures == [0.0]


def test_mathverse_score_keeps_raw_responses_and_uses_zero_temperature():
    judge = StubJudge(["The answers match.", "Judgement: 1\nExplanation: same value."])
    line = {
        "question_for_eval": "What is x?",
        "answer": "1/2",
        "extract": "0.5",
        "prediction": "work followed by 0.5",
    }

    result = MathVerse_auxeval_score(judge, line)

    assert result["score"] is True
    assert result["score_source"] == "judge"
    assert result["score_judge_responses"] == [
        "The answers match.",
        "Judgement: 1\nExplanation: same value.",
    ]
    assert judge.temperatures == [0.0, 0.0]


def test_mathverse_extract_keeps_raw_response():
    judge = StubJudge(["Extracted Answer: 42"])
    result = MathVerse_auxeval_extract(judge, {"prediction": "Therefore x=42."})

    assert result["extract"] == "42"
    assert result["extract_source"] == "judge"
    assert result["extract_judge_responses"] == ["Extracted Answer: 42"]
    assert judge.temperatures == [0.0]


def test_mathverse_reports_parser_failures(monkeypatch, tmp_path):
    monkeypatch.setenv("PRED_FORMAT", "json")
    prediction_file = tmp_path / "model_MathVerse_MINI.json"
    dump(pd.DataFrame({"index": [1, 2], "prediction": ["a", "b"]}), str(prediction_file))
    score_file = get_intermediate_file_path(
        str(prediction_file), "_judge_score_mcqfix_v2"
    )
    dump(
        pd.DataFrame(
            {
                "score_source": ["judge", "judge_parse_failure"],
                "extract_source": ["judge", "judge"],
            }
        ),
        score_file,
    )

    report = MathVerse.report_judge_err(
        prediction_file,
        total_samples=2,
        judge_model="judge",
    )

    assert report == {"failed": 1, "total": 2}


def test_dataframe_json_round_trip_preserves_long_text_and_nested_judge_outputs(tmp_path):
    long_prediction = "推理步骤\n" * 20_000
    frame = pd.DataFrame(
        {
            "index": [1],
            "prediction": [long_prediction],
            "extract": ["42"],
            "score": [True],
            "score_judge_responses": [["Judgement: 1\nExplanation: correct."]],
        }
    )
    output = tmp_path / "predictions.json"

    dump(frame, str(output))
    restored = load(str(output))

    assert isinstance(restored, pd.DataFrame)
    pd.testing.assert_frame_equal(restored, frame)
    assert restored.loc[0, "prediction"] == long_prediction
