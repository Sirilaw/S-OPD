# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Convert the original ViRL39K release to the parquet format used by OPD.

The official release has a single train split. This script therefore writes
only ``train.parquet`` and does not create a validation/test split.
"""

import argparse
import json
import os
import zipfile
from pathlib import Path
from typing import Any

import datasets

DATA_SOURCE = "TIGER-Lab/ViRL39K"
DEFAULT_DATASET_PATH = "/data2/siyuan/datasets/ViRL39K-original"
DEFAULT_SAVE_DIR = "/data2/siyuan/datasets/ViRL39K-opd-helpful-system"
DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."
INSTRUCTION_FOLLOWING = (
    r"You FIRST think about the reasoning process as an internal monologue and then provide the final answer. "
    r"The reasoning process MUST BE enclosed within <think> </think> tags. "
    r"The final answer MUST BE put in \boxed{}."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--local_dataset_path",
        default=DEFAULT_DATASET_PATH,
        help="Original ViRL39K directory, parquet file, or a dataset saved with save_to_disk().",
    )
    parser.add_argument(
        "--local_save_dir",
        default=DEFAULT_SAVE_DIR,
        help="Directory in which train.parquet is written.",
    )
    parser.add_argument("--local_dir", default=None, help="Deprecated alias for --local_save_dir.")
    parser.add_argument(
        "--images_dir",
        default=None,
        help="Optional image root. By default paths are resolved relative to the original dataset directory.",
    )
    parser.add_argument(
        "--images_archive",
        default=None,
        help="Optional images.zip path. By default, <dataset_dir>/images.zip is used when present.",
    )
    parser.add_argument(
        "--cache_dir",
        default=None,
        help="Hugging Face cache directory. Defaults to <local_save_dir>/.cache.",
    )
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument(
        "--image_storage",
        choices=("embedded", "path"),
        default="embedded",
        help="Embed image bytes for portable parquet, or retain absolute paths for smaller/faster output.",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Process only the first N rows. Intended for smoke tests.",
    )
    parser.add_argument(
        "--system_prompt",
        default=DEFAULT_SYSTEM_PROMPT,
        help="System message prepended to every prompt. Pass an empty string to omit it.",
    )
    return parser.parse_args()


def find_source_parquet(dataset_path: Path) -> Path:
    if dataset_path.is_file():
        if dataset_path.suffix != ".parquet":
            raise ValueError(f"Expected a parquet file, got: {dataset_path}")
        return dataset_path

    for name in ("39Krelease.parquet", "train.parquet"):
        candidate = dataset_path / name
        if candidate.is_file():
            return candidate

    parquet_files = sorted(dataset_path.rglob("*.parquet"))
    if len(parquet_files) == 1:
        return parquet_files[0]
    if not parquet_files:
        raise FileNotFoundError(f"No parquet file found under {dataset_path}")
    candidates = "\n  ".join(str(path) for path in parquet_files)
    raise RuntimeError(
        "Multiple parquet files were found. Pass the desired file directly with "
        f"--local_dataset_path:\n  {candidates}"
    )


def load_source_dataset(dataset_path: Path, cache_dir: Path) -> tuple[datasets.Dataset, Path]:
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"ViRL39K source does not exist: {dataset_path}\n"
            "Check that /data_122 is mounted, or pass the correct path with --local_dataset_path."
        )

    if dataset_path.is_dir() and (dataset_path / "dataset_info.json").is_file():
        loaded = datasets.load_from_disk(str(dataset_path))
        if isinstance(loaded, datasets.DatasetDict):
            loaded = loaded["train"]
        return loaded, dataset_path

    parquet_path = find_source_parquet(dataset_path)
    loaded = datasets.load_dataset(
        "parquet",
        data_files=str(parquet_path),
        split="train",
        cache_dir=str(cache_dir),
    )
    return loaded, parquet_path.parent


def normalize_image_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (json.JSONDecodeError, TypeError):
            pass
    if isinstance(value, (list, tuple)):
        return list(value)
    if hasattr(value, "tolist"):
        converted = value.tolist()
        return converted if isinstance(converted, list) else [converted]
    return [value]


def resolve_image_path(relative_path: str, dataset_root: Path, images_root: Path | None) -> Path:
    image_path = Path(os.path.expanduser(relative_path))
    if image_path.is_absolute():
        candidates = [image_path]
    else:
        candidates = [dataset_root / image_path]
        if images_root is not None:
            candidates.extend((images_root / image_path, images_root / image_path.name))

    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    tried = "\n  ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Could not resolve image {relative_path!r}. Tried:\n  {tried}")


def prepare_image(
    value: Any,
    dataset_root: Path,
    images_root: Path | None,
    image_storage: str,
    images_archive: zipfile.ZipFile | None,
) -> Any:
    # PIL images and already embedded Hugging Face Image dictionaries are ready.
    if not isinstance(value, (str, dict)):
        return value
    if isinstance(value, dict) and value.get("bytes") is not None:
        return value

    original_path = value.get("path") if isinstance(value, dict) else value
    if not original_path:
        raise ValueError(f"Unsupported image entry: {value!r}")
    try:
        path = resolve_image_path(str(original_path), dataset_root, images_root)
    except FileNotFoundError as path_error:
        if images_archive is None:
            raise
        archive_name = str(original_path).replace("\\", "/").lstrip("./")
        try:
            image_bytes = images_archive.read(archive_name)
        except KeyError as archive_error:
            raise FileNotFoundError(
                f"Could not resolve image {original_path!r} as an extracted file or as "
                f"{archive_name!r} inside {images_archive.filename}"
            ) from archive_error
        if image_storage == "path":
            raise ValueError(
                "--image_storage path cannot reference files inside images.zip. "
                "Use the default --image_storage embedded or extract images.zip first."
            ) from path_error
        return {"path": Path(archive_name).name, "bytes": image_bytes}
    else:
        if image_storage == "path":
            return {"path": str(path), "bytes": None}
        with path.open("rb") as image_file:
            return {"path": path.name, "bytes": image_file.read()}


def strip_outer_boxed(answer: str) -> str:
    """Remove one outer ``\\boxed{}`` wrapper for rule-based answer grading."""
    answer = answer.strip()
    if answer.startswith("$") and answer.endswith("$"):
        answer = answer[1:-1].strip()
    prefix = r"\boxed{"
    if not answer.startswith(prefix):
        return answer

    depth = 1
    for position in range(len(prefix), len(answer)):
        if answer[position] == "{":
            depth += 1
        elif answer[position] == "}":
            depth -= 1
            if depth == 0:
                if not answer[position + 1 :].strip():
                    return answer[len(prefix) : position]
                break
    return answer


def add_missing_image_placeholders(question: str, image_count: int) -> str:
    placeholder_count = question.count("<image>")
    if placeholder_count == image_count:
        return question
    if placeholder_count == 0 and image_count:
        return "".join("<image>" for _ in range(image_count)) + " " + question
    raise ValueError(
        f"Question has {placeholder_count} <image> placeholders but {image_count} images: {question[:160]!r}"
    )


def optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def convert_example(
    example: dict[str, Any],
    index: int,
    dataset_root: Path,
    images_root: Path | None,
    image_storage: str,
    images_archive: zipfile.ZipFile | None,
    system_prompt: str,
) -> dict[str, Any]:
    question_key = "question" if "question" in example else "problem"
    image_key = "image" if "image" in example else "images"
    if question_key not in example or "answer" not in example or image_key not in example:
        raise KeyError(
            "Expected question/problem, answer, and image/images columns; "
            f"available columns are: {sorted(example)}"
        )

    question = str(example[question_key]).strip()
    answer = str(example["answer"]).strip()
    raw_images = normalize_image_list(example[image_key])
    question = add_missing_image_placeholders(question, len(raw_images))
    images = [
        prepare_image(image, dataset_root, images_root, image_storage, images_archive)
        for image in raw_images
    ]
    category = str(example.get("category") or "vision-language reasoning")

    prompt = [
        {
            "role": "user",
            "content": f"{question} {INSTRUCTION_FOLLOWING}",
        }
    ]
    if system_prompt:
        prompt.insert(0, {"role": "system", "content": system_prompt})

    return {
        "data_source": DATA_SOURCE,
        "prompt": prompt,
        "images": images,
        "ability": category,
        "reward_model": {
            "style": "rule",
            "ground_truth": strip_outer_boxed(answer),
        },
        "extra_info": {
            "split": "train",
            "index": index,
            "answer": answer,
            "question": question,
            "qid": str(example.get("qid") or index),
            "category": category,
            "source": str(example.get("source") or ""),
            "pass_rate_32b_trained": optional_float(example.get("PassRate_32BTrained")),
            "pass_rate_7b_base": optional_float(
                example.get("PassRate_7BBase", example.get("PassRate_7BUntrained"))
            ),
        },
    }


def main() -> None:
    args = parse_args()
    dataset_path = Path(os.path.expanduser(args.local_dataset_path)).resolve()
    save_dir_arg = args.local_dir if args.local_dir is not None else args.local_save_dir
    if args.local_dir is not None:
        print("Warning: --local_dir is deprecated; use --local_save_dir.")
    save_dir = Path(os.path.expanduser(save_dir_arg)).resolve()
    images_root = Path(os.path.expanduser(args.images_dir)).resolve() if args.images_dir else None
    cache_dir = (
        Path(os.path.expanduser(args.cache_dir)).resolve()
        if args.cache_dir
        else save_dir / ".cache"
    )
    cache_dir.mkdir(parents=True, exist_ok=True)

    source_dataset, dataset_root = load_source_dataset(dataset_path, cache_dir)
    images_archive_path = (
        Path(os.path.expanduser(args.images_archive)).resolve()
        if args.images_archive
        else dataset_root / "images.zip"
    )
    if not images_archive_path.is_file():
        images_archive_path = None
    if args.max_samples is not None:
        if args.max_samples <= 0:
            raise ValueError("--max_samples must be positive")
        source_dataset = source_dataset.select(range(min(args.max_samples, len(source_dataset))))
    if len(source_dataset) == 0:
        raise ValueError("The source dataset is empty")

    print(f"Loading {len(source_dataset):,} ViRL39K examples from {dataset_path}")
    print(f"Images will be stored as: {args.image_storage}")
    if images_archive_path is not None:
        print(f"Reading unextracted images from {images_archive_path}")

    def generate_examples():
        archive = zipfile.ZipFile(images_archive_path) if images_archive_path is not None else None
        try:
            for index, example in enumerate(source_dataset):
                if index and index % 1000 == 0:
                    print(f"Converted {index:,}/{len(source_dataset):,} examples")
                yield convert_example(
                    example,
                    index,
                    dataset_root,
                    images_root,
                    args.image_storage,
                    archive,
                    args.system_prompt,
                )
        finally:
            if archive is not None:
                archive.close()

    processed = datasets.Dataset.from_generator(generate_examples, cache_dir=str(cache_dir))
    processed = processed.cast_column("images", datasets.Sequence(datasets.Image()))

    save_dir.mkdir(parents=True, exist_ok=True)
    train_path = save_dir / "train.parquet"
    processed.to_parquet(str(train_path))
    print(f"Wrote {len(processed):,} train examples to {train_path}")

    if args.hdfs_dir is not None:
        from verl.utils.hdfs_io import copy, makedirs

        makedirs(args.hdfs_dir)
        copy(src=str(save_dir), dst=args.hdfs_dir)


if __name__ == "__main__":
    main()
