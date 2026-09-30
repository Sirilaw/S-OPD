from PIL import Image

from evaluation.geometry3k_oracle.canonical_601_eval import (
    insert_oracle_facts,
    structure_image_messages,
)


def test_oracle_insertion_preserves_the_canonical_suffix():
    question = "<image>Find x."
    suffix = " You FIRST think, then put the final answer in \\boxed{}."
    prompt = [{"role": "user", "content": question + suffix}]

    updated = insert_oracle_facts(prompt, question, ["AB is parallel to CD."])

    assert prompt[0]["content"] == question + suffix
    assert updated[0]["content"] == (
        question
        + "\n\nDiagram observations:\n- AB is parallel to CD."
        + suffix
    )


def test_empty_oracle_facts_leave_prompt_unchanged():
    prompt = [{"role": "user", "content": "<image>Find x."}]

    updated = insert_oracle_facts(prompt, "<image>Find x.", [])

    assert updated == prompt
    assert updated is not prompt


def test_structured_image_prompt_matches_dataset_conversion():
    image = Image.new("RGB", (2, 2), "white")
    prompt = [{"role": "user", "content": "<image>Find x."}]

    updated = structure_image_messages(prompt, image)

    assert prompt[0]["content"] == "<image>Find x."
    assert updated[0]["content"][0]["type"] == "image"
    assert updated[0]["content"][0]["image"] is image
    assert updated[0]["content"][1] == {"type": "text", "text": "Find x."}
