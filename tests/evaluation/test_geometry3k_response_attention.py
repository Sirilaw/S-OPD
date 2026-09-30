from pathlib import Path

import numpy as np
import pytest

from evaluation.geometry3k_attention_rollout.visualize_response_attention import (
    load_generic_example,
    load_saved,
    parse_layers,
    resolve_checkpoint,
)


def test_parse_layers():
    assert parse_layers("all", 4) == [0, 1, 2, 3]
    assert parse_layers("last", 4) == [3]
    assert parse_layers("0,2-3", 4) == [0, 2, 3]
    with pytest.raises(ValueError):
        parse_layers("4", 4)
    with pytest.raises(ValueError):
        parse_layers("3-2", 4)


def test_resolve_checkpoint_prefers_largest_numeric_step(tmp_path: Path):
    for step in (9, 100, 20):
        checkpoint = (
            tmp_path / "checkpoints" / f"global_step_{step}" / "actor" / "huggingface"
        )
        checkpoint.mkdir(parents=True)
        (checkpoint / "config.json").write_text("{}", encoding="utf-8")
    assert "global_step_100" in str(resolve_checkpoint(tmp_path))


def test_resolve_direct_hf_checkpoint(tmp_path: Path):
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    assert resolve_checkpoint(tmp_path) == tmp_path.resolve()


def test_load_generic_example(tmp_path: Path):
    image = tmp_path / "busy_street.jpg"
    image.write_bytes(b"not decoded by this helper")
    example = load_generic_example(image, " Describe every visible detail. ", None)
    assert example["sample_id"] == "busy_street"
    assert example["prompt"] == "Describe every visible detail."
    assert example["image_path"] == image.resolve()


def test_load_saved_can_reaggregate_one_layer(tmp_path: Path):
    attention = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)
    np.savez_compressed(
        tmp_path / "vanilla_attention.npz",
        raw_map=np.zeros((2, 2), dtype=np.float32),
        conditional_map=np.zeros((2, 2), dtype=np.float32),
        attention=attention,
        visual_mass_by_layer_token=attention.sum(axis=-1),
        layers=np.asarray([0, 1]),
        response_ids=np.asarray([1, 2, 3]),
    )
    metadata = {"vanilla": {"visual_grid": [2, 2], "layers": [0, 1]}}
    result = load_saved(tmp_path, "vanilla", metadata, "0")
    assert result["layers"] == [0]
    np.testing.assert_array_equal(result["raw_map"], attention[0].mean(axis=0).reshape(2, 2))
