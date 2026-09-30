import sys
from pathlib import Path

VLMEVALKIT_ROOT = Path(__file__).resolve().parents[2] / "third_party" / "VLMEvalKit"
sys.path.insert(0, str(VLMEVALKIT_ROOT))

from vlmeval.dataset.utils.mathverse import (  # noqa: E402
    FAIL_MSG,
    MathVerse_auxeval_extract,
    MathVerse_auxeval_score,
    map_mathverse_response_to_option,
)

QUESTION = """Find the angle.\nChoices:\nA:40°\nB:60°\nC:120°\nD:140°"""


class _Judge:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = 0

    def generate(self, prompt, temperature=0.0):
        assert QUESTION in prompt
        assert temperature == 0.0
        self.calls += 1
        return next(self.responses)


def _line():
    return {
        "question_type": "multi-choice",
        "answer": "D",
        "prediction": "The supplementary angle is 140 degrees.",
        "question_for_eval": QUESTION,
    }


def test_maps_semantic_judge_answer_to_unique_option():
    assert map_mathverse_response_to_option("140°", QUESTION) == "D"
    assert map_mathverse_response_to_option("Final answer: 140 degrees", QUESTION) == "D"
    assert map_mathverse_response_to_option("75°", QUESTION) is None

    judge = _Judge(["140°"])
    extracted = MathVerse_auxeval_extract(judge, _line())

    assert judge.calls == 1
    assert extracted["extract"] == "D"
    assert extracted["extract_source"] == "judge_mcq"
    assert extracted["extract_judge_responses"] == ["140°"]

    scored = MathVerse_auxeval_score(judge, {**_line(), **extracted})
    assert scored["score"] is True
    assert scored["score_source"] == "rule_mcq_exact_match"


def test_stable_non_choice_answer_is_incorrect_not_parse_failure():
    judge = _Judge(["75°"] * 5)
    extracted = MathVerse_auxeval_extract(judge, _line())

    assert judge.calls == 5
    assert extracted["extract"] == "75°"
    assert extracted["extract_source"] == "judge_mcq_invalid_choice_majority"
    assert "All 5 retries failed" not in extracted["log_extract"]

    scored = MathVerse_auxeval_score(judge, {**_line(), **extracted})
    assert scored["score"] is False
    assert scored["score_source"] == "rule_mcq_invalid_choice"
    assert scored["score_judge_responses"] == []


def test_inconsistent_unparseable_answers_remain_a_real_parse_failure():
    judge = _Judge(["x", "y", "z", "w", "q"])
    extracted = MathVerse_auxeval_extract(judge, _line())

    assert extracted["extract"] == ""
    assert extracted["extract_source"] == "judge_parse_failure"
    assert "All 5 retries failed" in extracted["log_extract"]


def test_stable_api_errors_remain_a_real_failure():
    judge = _Judge([FAIL_MSG] * 5)
    extracted = MathVerse_auxeval_extract(judge, _line())

    assert extracted["extract"] == ""
    assert extracted["extract_source"] == "judge_parse_failure"
    assert "All 5 retries failed" in extracted["log_extract"]
