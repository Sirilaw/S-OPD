from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "analyze_teacher_gate_token_types.py"
SPEC = importlib.util.spec_from_file_location("teacher_gate_token_types", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_requested_semantic_categories_are_separated():
    assert MODULE.lexical_category("triangle") == "visual_element_or_attribute"
    assert MODULE.lexical_category("above") == "visual_relationship"
    assert MODULE.lexical_category("therefore") == "reasoning_or_logic_point"
    assert MODULE.lexical_category("boxed") == "answer_or_conclusion"
    assert MODULE.lexical_category("42") == "number"


def test_geometry_label_inherits_visual_element_role():
    assert MODULE.lexical_category("A", previous="point") == "visual_element_or_attribute"
    assert MODULE.lexical_category("D", previous="option") == "other_content_word"


def test_token_offsets_map_to_mutually_exclusive_units():
    units = MODULE.text_units("triangle A is above line BC.")
    offsets = [(0, 8), (8, 10), (10, 13), (13, 19), (19, 24), (24, 27), (27, 28)]

    aligned = MODULE.align_units_to_tokens(units, offsets)

    assert aligned[0] == ("triangle", "visual_element_or_attribute")
    assert aligned[1] == ("a", "visual_element_or_attribute")
    assert aligned[3] == ("above", "visual_relationship")
    assert aligned[-1] == (".", "punctuation")


class _FakeTokenizer:
    is_fast = True

    def __call__(self, text, add_special_tokens, return_offsets_mapping):
        assert not add_special_tokens
        assert return_offsets_mapping
        return {"offset_mapping": [(0, len(text))]}


def test_alignment_accepts_short_hidden_special_token_suffix():
    offsets, _, aligned, adjustment = MODULE.align_text(
        _FakeTokenizer(), "triangle", expected_tokens=3
    )

    assert offsets == [(0, 8), (8, 8), (8, 8)]
    assert adjustment == 2
    assert aligned[-2:] == [
        ("<eos>", "structure_or_control"),
        ("<eos>", "structure_or_control"),
    ]


class _TwoPieceTokenizer:
    is_fast = True

    def __call__(self, text, add_special_tokens, return_offsets_mapping):
        return {"offset_mapping": [(0, 4), (4, len(text))]}


def test_alignment_trims_one_retokenized_piece_at_truncation_boundary():
    offsets, _, aligned, adjustment = MODULE.align_text(
        _TwoPieceTokenizer(), "triangle", expected_tokens=1
    )

    assert offsets == [(0, 4)]
    assert len(aligned) == 1
    assert adjustment == -1
