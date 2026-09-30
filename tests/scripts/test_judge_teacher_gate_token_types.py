from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))
SCRIPT = SCRIPTS_DIR / "judge_teacher_gate_token_types.py"
SPEC = importlib.util.spec_from_file_location("teacher_gate_llm_judge", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_api_config_fields_are_strings():
    assert isinstance(MODULE.BASE_URL, str)
    assert isinstance(MODULE.API_KEY, str)
    assert isinstance(MODULE.MODEL_NAME, str)


def test_marked_context_highlights_only_target_span():
    context = MODULE.marked_context("triangle A is above line BC", 14, 19, 100)

    assert "[[TARGET: above]]" in context


def test_validate_judgments_accepts_complete_known_categories():
    payload = {
        "judgments": [
            {
                "id": "a",
                "category": "visual_relationship",
                "confidence": 0.9,
                "rationale": "spatial relation",
            },
            {
                "id": "b",
                "category": "reasoning_or_logic_point",
                "confidence": 0.8,
                "rationale": "logical transition",
            },
        ]
    }

    result = MODULE.validate_judgments(payload, ["a", "b"])

    assert [row["id"] for row in result] == ["a", "b"]


def test_validate_judgments_rejects_missing_or_unknown_labels():
    payload = {
        "judgments": [
            {"id": "a", "category": "unknown", "confidence": 0.9, "rationale": "x"}
        ]
    }

    with pytest.raises(ValueError, match="Unknown category"):
        MODULE.validate_judgments(payload, ["a"])


def test_wilson_interval_contains_observed_fraction():
    low, high = MODULE.wilson_interval(30, 100)

    assert low < 0.3 < high


def test_heuristic_categories_map_to_llm_taxonomy():
    mapping = MODULE.HEURISTIC_TO_JUDGE_CATEGORY

    assert mapping["number"] == "calculation_or_quantitative"
    assert mapping["math_operator_or_symbol"] == "calculation_or_quantitative"
    assert mapping["function_word"] == "discourse_or_function"
    assert mapping["punctuation"] == "nonsemantic_or_structure"
