from scripts.probe_three_section_vllm import (
    THREE_SECTION_INSTRUCTION,
    analyze_response,
    append_three_section_instruction,
    parse_indices,
)


def test_analyze_response_accepts_only_complete_ordered_nonempty_sections():
    response = """<grounding>visible facts</grounding>
<reasoning>reason from facts</reasoning>
<answer>42</answer>"""
    result = analyze_response(response)
    assert result["all_tags_once"]
    assert result["ordered"]
    assert result["nonempty_sections"]
    assert result["exact_three_section_format"]


def test_analyze_response_rejects_extra_text_and_empty_or_reordered_sections():
    extra = "prefix <grounding>x</grounding><reasoning>y</reasoning><answer>z</answer>"
    empty = "<grounding>x</grounding><reasoning></reasoning><answer>z</answer>"
    reordered = "<reasoning>y</reasoning><grounding>x</grounding><answer>z</answer>"
    assert not analyze_response(extra)["exact_three_section_format"]
    assert not analyze_response(empty)["exact_three_section_format"]
    assert not analyze_response(reordered)["exact_three_section_format"]


def test_instruction_is_appended_once_to_last_user_message():
    messages = [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "reply"},
        {"role": "user", "content": [{"type": "text", "text": "question"}]},
    ]
    append_three_section_instruction(messages)
    append_three_section_instruction(messages)
    text_parts = [part["text"] for part in messages[-1]["content"] if part.get("type") == "text"]
    assert sum(text.count(THREE_SECTION_INSTRUCTION) for text in text_parts) == 1
    assert THREE_SECTION_INSTRUCTION in messages[-1]["content"][-1]["text"]


def test_parse_indices_is_deterministic_and_validates_bounds():
    assert parse_indices(None, row_count=10, sample_count=3, selection="first", seed=0) == [0, 1, 2]
    assert parse_indices(None, row_count=10, sample_count=3, selection="random", seed=7) == parse_indices(
        None, row_count=10, sample_count=3, selection="random", seed=7
    )
