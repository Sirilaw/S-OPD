import sys
import types
import json

# Keep answer-unit tests runnable without installing VLMEvalKit.
vlmeval = types.ModuleType("vlmeval")
dataset = types.ModuleType("vlmeval.dataset")
image_base = types.ModuleType("vlmeval.dataset.image_base")
image_base.ImageBaseDataset = object
smp = types.ModuleType("vlmeval.smp")
smp.dump = lambda *args: None
smp.load = lambda *args: None
sys.modules.setdefault("vlmeval", vlmeval)
sys.modules.setdefault("vlmeval.dataset", dataset)
sys.modules.setdefault("vlmeval.dataset.image_base", image_base)
sys.modules.setdefault("vlmeval.smp", smp)

from PIL import Image

from local_math_dataset import (
    _materialize_images,
    _random_patch_mask_images,
    _json_safe_record,
    answers_equal,
    answers_equal_with_choices,
    extract_answer,
    has_complete_boxed_answer,
    resolve_choice_answer,
)


def test_boxed_and_numeric_answers():
    assert extract_answer(r"work... \boxed{\frac{1}{2}}") == r"\frac{1}{2}"
    assert answers_equal(r"Therefore \boxed{\frac{1}{2}}", "0.5")
    assert answers_equal("Answer: 50%", "0.5")


def test_multiple_choice_and_alternatives():
    assert answers_equal("The answer is (b)", "B")
    assert answers_equal(r"\boxed{12}", "11 || 12")


def test_choice_value_resolution_and_boxed_format_validation():
    choices = '["5", "12", "13", "26"]'
    assert resolve_choice_answer(r"\boxed{A}", choices) == "5"
    assert resolve_choice_answer(r"\boxed{A. \\ 5}", choices) == "5"
    assert answers_equal_with_choices(r"\boxed{A. 5}", "5.0", choices)
    assert has_complete_boxed_answer(r"reasoning... \boxed{A}")
    assert not has_complete_boxed_answer("Answer: A")
    assert not has_complete_boxed_answer(r"truncated \boxed{A")


def test_embedded_parquet_image_bytes(tmp_path):
    png = b"\x89PNG\r\n\x1a\n" + b"test-payload"
    paths = _materialize_images([{"bytes": png, "path": None}], tmp_path, tmp_path / "cache", 3)
    assert paths[0].endswith(".png")
    assert (tmp_path / "cache" / paths[0].rsplit("/", 1)[-1]).read_bytes() == png


def test_random_patch_mask_ablation_is_deterministic(tmp_path):
    source = tmp_path / "source.png"
    Image.new("RGB", (28, 28), color="red").save(source)

    first = _random_patch_mask_images([str(source)], tmp_path / "cache", 7, 0.6, 14, 123)
    second = _random_patch_mask_images([str(source)], tmp_path / "cache", 7, 0.6, 14, 123)

    assert first == second
    assert Image.open(first[0]).tobytes() == Image.open(second[0]).tobytes()


def test_random_patch_mask_ratio_extremes(tmp_path):
    source = tmp_path / "source.png"
    Image.new("RGB", (17, 15), color="red").save(source)

    original = _random_patch_mask_images([str(source)], tmp_path / "cache", 0, 0.0, 14, 0)
    black = _random_patch_mask_images([str(source)], tmp_path / "cache", 0, 1.0, 14, 0)

    assert original == [str(source)]
    assert Image.open(black[0]).getbbox() is None


def test_dataframe_prediction_payload_does_not_keep_raw_image_bytes(tmp_path):
    source = tmp_path / "source.png"
    Image.new("RGB", (8, 8), color="red").save(source)
    raw = {"bytes": source.read_bytes(), "path": None}

    # Mirrors LocalMathDataset's construction of the JSON-facing record.
    image_column = "images"
    prompt_column = "prompt"
    row = {
        "index": 0,
        prompt_column: [{"role": "user", "content": "q"}],
        "answer": "a",
        image_column: [raw],
    }
    record = _json_safe_record(row, (image_column, prompt_column))
    record["question"] = "q"
    record["image_path"] = str(source)

    json.dumps(record)
    assert image_column not in record
    assert prompt_column not in record
