from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image


SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "analyze_mathverse_visual_delta_vllm.py"
SPEC = importlib.util.spec_from_file_location("mathverse_delta", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_random_mask_is_deterministic_and_keeps_size():
    image = Image.new("RGB", (31, 29), "white")
    first = MODULE.random_mask_image(image, mask_ratio=0.6, patch_size=14, seed=123)
    second = MODULE.random_mask_image(image, mask_ratio=0.6, patch_size=14, seed=123)
    assert first.size == image.size
    assert np.array_equal(np.asarray(first), np.asarray(second))
    assert set(np.unique(np.asarray(first))).issubset({0, 255})


def test_patch_shuffle_is_deterministic_preserves_pixels_and_changes_layout():
    # Each 14x14 block has a unique constant RGB value, making both the pixel
    # multiset and spatial permutation easy to verify.
    array = np.zeros((28, 42, 3), dtype=np.uint8)
    value = 10
    for top in range(0, 28, 14):
        for left in range(0, 42, 14):
            array[top : top + 14, left : left + 14] = (value, value + 1, value + 2)
            value += 20
    image = Image.fromarray(array, mode="RGB")
    first = MODULE.patch_shuffle_image(image, block_size=14, seed=123)
    second = MODULE.patch_shuffle_image(image, block_size=14, seed=123)
    first_array = np.asarray(first)
    assert first.size == image.size
    assert np.array_equal(first_array, np.asarray(second))
    assert not np.array_equal(first_array, array)
    assert np.array_equal(
        np.sort(first_array.reshape(-1, 3), axis=0),
        np.sort(array.reshape(-1, 3), axis=0),
    )


def test_additive_gaussian_noise_is_deterministic_clipped_and_channel_independent():
    image = Image.new("RGB", (128, 128), (128, 128, 128))
    first = MODULE.additive_gaussian_noise_image(image, std=0.2, seed=123)
    second = MODULE.additive_gaussian_noise_image(image, std=0.2, seed=123)
    values = np.asarray(first)
    assert first.size == image.size
    assert np.array_equal(values, np.asarray(second))
    assert values.min() >= 0 and values.max() <= 255
    assert not np.array_equal(values[..., 0], values[..., 1])
    assert 40 < values[..., 0].std() < 60


def test_prepare_stratifies_and_deduplicates_images(tmp_path):
    source_image = tmp_path / "source.png"
    Image.new("RGB", (28, 28), "white").save(source_image)
    rows = []
    for version in MODULE.PROBLEM_VERSIONS:
        for index in range(2):
            rows.append(
                {
                    "sample_index": f"{version}-{index}",
                    "problem_index": index,
                    "problem_version": version,
                    "query_cot": f"Solve {version} {index}",
                    "answer": "1",
                    "image": str(source_image),
                }
            )
    dataset = tmp_path / "testmini.json"
    dataset.write_text(json.dumps(rows), encoding="utf-8")
    output = tmp_path / "output"
    MODULE.prepare(
        argparse.Namespace(
            dataset=dataset,
            images_dir=None,
            output_dir=output,
            samples_per_class=1,
            question_field="auto",
            seed=7,
            overwrite=False,
        )
    )
    manifest = MODULE.read_jsonl(output / "manifest.jsonl")
    assert len(manifest) == 5
    assert {row["problem_version"] for row in manifest} == set(MODULE.PROBLEM_VERSIONS)
    assert all(row["question_field"] == "query_cot" for row in manifest)
    assert len(list((output / "images").glob("*.png"))) == 1


def test_token_type_basic_classes():
    specials = {"<|im_end|>"}
    assert MODULE.token_type("<|im_end|>", "<|im_end|>", specials) == "special_or_control"
    assert MODULE.token_type("x", "  ", specials) == "whitespace"
    assert MODULE.token_type("x", "\n", specials) == "newline"
    assert MODULE.token_type("x", "42", specials) == "number_or_numeric_piece"
    assert MODULE.token_type("x", "\\frac", specials) == "latex_command"
    assert MODULE.token_type("x", "=", specials) == "math_operator_or_symbol"
    assert MODULE.token_type("x", "angle", specials) == "alphabetic_word_piece"


def test_response_prompt_logprobs_uses_expanded_output_suffix():
    token_ids = [100, 151655, 151655, 200, 201]
    prompt_logprobs = [None]
    for token_id in token_ids[1:]:
        prompt_logprobs.append({token_id: SimpleNamespace(logprob=-token_id / 1000)})
    output = SimpleNamespace(prompt_token_ids=token_ids, prompt_logprobs=prompt_logprobs)
    values = MODULE.response_prompt_logprobs(output, [200, 201])
    assert values == [-0.2, -0.201]
