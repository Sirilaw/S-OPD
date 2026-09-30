#!/usr/bin/env python3
"""Pre-filter a verl multimodal RL dataset with the model processor."""

import argparse
import copy
import os
import re
import traceback
from io import BytesIO
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from qwen_vl_utils.vision_process import smart_resize

from verl.utils import hf_processor, hf_tokenizer
from verl.utils.dataset.vision_utils import process_image
from verl.utils.dataset.rl_dataset import RLHFDataset


FILTER_INDEX_COLUMN = "__filter_original_index"
PROMPT_LENGTH_COLUMN = "__filter_prompt_length"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Filter prompts using the same multimodal processor path as verl, "
            "then save a standalone parquet in the original row order."
        )
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--max-prompt-length", type=int, default=4096)
    parser.add_argument("--num-workers", type=int, default=32)
    parser.add_argument("--shuffle-seed", type=int, default=42)
    parser.add_argument(
        "--length-mode",
        choices=("qwen-image-metadata", "full"),
        default="qwen-image-metadata",
        help=(
            "Use Qwen's exact image-grid formula without materializing pixel tensors, "
            "or run the complete image processor for every row."
        ),
    )
    parser.add_argument(
        "--validation-samples",
        type=int,
        default=16,
        help="Full-processor checks for qwen-image-metadata mode.",
    )
    parser.add_argument("--system-prompt")
    parser.add_argument(
        "--normalize-thinking-tags",
        action="store_true",
        help=(
            "Replace <thinking>/</thinking> with <think>/</think> in prompt "
            "and teacher_prompt messages before computing prompt lengths."
        ),
    )
    parser.add_argument("--use-shm", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def validate_system_prompt(dataframe, expected: str) -> None:
    invalid_indices = []
    for index, messages in enumerate(dataframe["prompt"]):
        if (
            not messages
            or messages[0].get("role") != "system"
            or messages[0].get("content") != expected
        ):
            invalid_indices.append(index)
            if len(invalid_indices) == 10:
                break
    if invalid_indices:
        raise ValueError(
            f"System prompt mismatch at row(s) {invalid_indices}; expected {expected!r}."
        )


def normalize_thinking_tags(dataframe):
    """Normalize reasoning tags without changing the VERL message schema."""

    message_columns = [
        column for column in ("prompt", "teacher_prompt") if column in dataframe.column_names
    ]

    def normalize_row(example: dict) -> dict:
        updates = {}
        for column in message_columns:
            messages = copy.deepcopy(example[column])
            if isinstance(messages, dict):
                messages = [messages]
            for message in messages or []:
                content = message.get("content")
                if isinstance(content, str):
                    message["content"] = content.replace(
                        "<thinking>", "<think>"
                    ).replace("</thinking>", "</think>")
            updates[column] = messages
        return updates

    return dataframe.map(
        normalize_row,
        desc="Normalizing <thinking> tags to <think>",
    )


def build_messages_for_template(doc: dict, prompt_key: str, image_key: str) -> list[dict]:
    raw = copy.deepcopy(doc[prompt_key])
    messages = [raw] if isinstance(raw, dict) else raw
    if not isinstance(messages, list):
        raise TypeError(f"prompt must be dict or list[dict], got {type(raw)}")

    images = doc.get(image_key) or []
    if not isinstance(images, (list, tuple)):
        images = [images]

    image_offset = 0
    for message in messages:
        if not images or not isinstance(message["content"], str):
            continue
        content_list = []
        segments = [segment for segment in re.split("(<image>)", message["content"]) if segment]
        for segment in segments:
            if segment == "<image>":
                if image_offset >= len(images):
                    raise ValueError("Prompt contains more <image> placeholders than images.")
                content_list.append({"type": "image"})
                image_offset += 1
            else:
                content_list.append({"type": "text", "text": segment})
        message["content"] = content_list

    if image_offset != len(images):
        raise ValueError(
            f"Prompt contains {image_offset} <image> placeholders for {len(images)} images."
        )
    return messages


def original_image_size(image: dict | Image.Image) -> tuple[int, int]:
    if isinstance(image, Image.Image):
        width, height = image.size
        return height, width
    if not isinstance(image, dict):
        raise TypeError(f"Unsupported image type: {type(image)}")
    if isinstance(image.get("image"), Image.Image):
        width, height = image["image"].size
        return height, width
    if image.get("bytes") is not None:
        with Image.open(BytesIO(image["bytes"])) as pil_image:
            width, height = pil_image.size
        return height, width
    image_path = image.get("path") or image.get("image")
    if image_path is None:
        raise ValueError("Image has neither bytes nor a path.")
    if isinstance(image_path, str) and image_path.startswith("file://"):
        image_path = image_path[7:]
    with Image.open(image_path) as pil_image:
        width, height = pil_image.size
    return height, width


def qwen_resized_image_size(image: dict | Image.Image, image_patch_size: int) -> tuple[int, int]:
    # datasets.Image decodes parquet image structs into PIL.Image. verl's
    # process_image returns PIL inputs directly without qwen_vl_utils resizing,
    # so the model image processor must see the original dimensions here too.
    if isinstance(image, Image.Image):
        return original_image_size(image)

    patch_factor = image_patch_size * 2
    if "resized_height" in image and "resized_width" in image:
        return smart_resize(
            image["resized_height"],
            image["resized_width"],
            factor=patch_factor,
        )

    height, width = original_image_size(image)
    min_pixels = image.get("min_pixels")
    max_pixels = image.get("max_pixels")
    return smart_resize(
        height,
        width,
        factor=patch_factor,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )


def qwen_metadata_prompt_length(
    doc: dict,
    processor,
    prompt_key: str,
    image_key: str,
    image_patch_size: int,
) -> int:
    images = doc.get(image_key) or []
    if not isinstance(images, (list, tuple)):
        images = [images]
    messages = build_messages_for_template(doc, prompt_key, image_key)
    raw_prompt = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )

    image_processor = processor.image_processor
    merge_length = image_processor.merge_size**2
    for image in images:
        resized_height, resized_width = qwen_resized_image_size(image, image_patch_size)
        image_patches = image_processor.get_number_of_image_patches(
            resized_height,
            resized_width,
            images_kwargs={},
        )
        image_tokens = image_patches // merge_length
        raw_prompt = raw_prompt.replace(
            processor.image_token,
            "<|placeholder|>" * image_tokens,
            1,
        )
    raw_prompt = raw_prompt.replace("<|placeholder|>", processor.image_token)
    return len(processor.tokenizer(text=[raw_prompt])["input_ids"][0])


def full_processor_prompt_length(
    doc: dict,
    dataset: RLHFDataset,
    processor,
    image_key: str,
    image_patch_size: int,
) -> int:
    images = doc.get(image_key) or []
    if not isinstance(images, (list, tuple)):
        images = [images]
    messages = dataset._build_messages(doc)
    raw_prompt = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )
    processed_images = [
        process_image(image, image_patch_size=image_patch_size) for image in images
    ]
    return len(
        processor(text=[raw_prompt], images=processed_images)["input_ids"][0]
    )


def validation_indices(lengths: np.ndarray, threshold: int, count: int) -> list[int]:
    count = min(max(count, 0), len(lengths))
    if count == 0:
        return []
    evenly_spaced_count = count // 2
    boundary_count = count - evenly_spaced_count
    evenly_spaced = np.linspace(
        0, len(lengths) - 1, num=evenly_spaced_count, dtype=np.int64
    )
    boundary = np.argsort(np.abs(lengths - threshold), kind="stable")[:boundary_count]
    return sorted(set(np.concatenate((evenly_spaced, boundary)).tolist()))


def main() -> None:
    args = parse_args()
    if args.max_prompt_length <= 0:
        raise ValueError("--max-prompt-length must be positive.")
    if args.num_workers <= 0:
        raise ValueError("--num-workers must be positive.")
    if args.validation_samples < 0:
        raise ValueError("--validation-samples must be non-negative.")
    if not args.input.is_file():
        raise FileNotFoundError(args.input)
    if args.input.resolve() == args.output.resolve():
        raise ValueError("--output must not overwrite --input.")

    model_kwargs = {"local_files_only": True} if args.local_files_only else {}
    tokenizer = hf_tokenizer(str(args.model), **model_kwargs)
    processor = hf_processor(str(args.model), use_fast=True, **model_kwargs)
    if processor is None:
        raise RuntimeError(f"{args.model!r} did not produce a multimodal processor.")

    config = OmegaConf.create(
        {
            "prompt_key": "prompt",
            "image_key": "images",
            "video_key": "videos",
            "image_patch_size": 14,
            "max_prompt_length": args.max_prompt_length,
            "filter_overlong_prompts": False,
            "filter_overlong_prompts_workers": args.num_workers,
            "use_shm": args.use_shm,
            "shuffle": False,
            "seed": args.shuffle_seed,
            "return_raw_chat": True,
        }
    )
    dataset = RLHFDataset(
        data_files=str(args.input),
        tokenizer=tokenizer,
        processor=processor,
        config=config,
    )
    dataframe = dataset.dataframe
    input_rows = len(dataframe)

    if args.normalize_thinking_tags:
        dataframe = normalize_thinking_tags(dataframe)

    if args.system_prompt is not None:
        validate_system_prompt(dataframe, args.system_prompt)

    if args.length_mode == "full":
        # Hugging Face assigns each process a contiguous logical shard. Shuffle
        # before expensive pixel preprocessing, then restore the original order.
        dataframe = dataframe.add_column(
            FILTER_INDEX_COLUMN, np.arange(input_rows, dtype=np.int64)
        )
        dataframe = dataframe.shuffle(seed=args.shuffle_seed)
        dataset.filter_overlong_prompts = True
        dataset.num_workers = args.num_workers
        filtered = dataset.maybe_filter_out_long_prompts(dataframe)
        filtered = filtered.sort(FILTER_INDEX_COLUMN)
        filtered = filtered.remove_columns(FILTER_INDEX_COLUMN)
        lengths = None
    else:
        def get_length(doc: dict) -> dict:
            try:
                length = qwen_metadata_prompt_length(
                    doc,
                    processor=processor,
                    prompt_key="prompt",
                    image_key="images",
                    image_patch_size=14,
                )
            except Exception:
                print("Error calculating prompt length; marking row as overlong.")
                traceback.print_exc()
                length = args.max_prompt_length + 1
            return {PROMPT_LENGTH_COLUMN: length}

        length_dataset = dataframe.map(
            get_length,
            remove_columns=dataframe.column_names,
            num_proc=args.num_workers if args.num_workers > 1 else None,
            batch_size=128,
            writer_batch_size=128,
            desc="Calculating exact Qwen image-token prompt lengths",
        )
        lengths = np.asarray(length_dataset[PROMPT_LENGTH_COLUMN], dtype=np.int64)

        check_indices = validation_indices(
            lengths,
            threshold=args.max_prompt_length,
            count=args.validation_samples,
        )
        print(f"Validating metadata lengths with full processor on {len(check_indices)} rows...")
        for index in check_indices:
            exact_length = full_processor_prompt_length(
                dataframe[int(index)],
                dataset=dataset,
                processor=processor,
                image_key="images",
                image_patch_size=14,
            )
            if exact_length != int(lengths[index]):
                raise RuntimeError(
                    f"Prompt length mismatch at row {index}: "
                    f"metadata={lengths[index]}, full_processor={exact_length}."
                )

        keep_indices = np.flatnonzero(lengths <= args.max_prompt_length).tolist()
        filtered = dataframe.select(keep_indices)

    if args.system_prompt is not None:
        validate_system_prompt(filtered, args.system_prompt)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = args.output.with_name(f".{args.output.name}.tmp-{os.getpid()}")
    try:
        filtered.to_parquet(str(temporary_output))
        os.replace(temporary_output, args.output)
    finally:
        if temporary_output.exists():
            temporary_output.unlink()

    output_rows = len(filtered)
    print(f"input rows: {input_rows}")
    print(f"output rows: {output_rows}")
    print(f"filtered rows: {input_rows - output_rows}")
    if lengths is not None:
        retained_lengths = lengths[lengths <= args.max_prompt_length]
        removed_lengths = lengths[lengths > args.max_prompt_length]
        print(f"maximum retained prompt length: {retained_lengths.max()}")
        if len(removed_lengths):
            print(f"minimum removed prompt length: {removed_lengths.min()}")
    print(f"output: {args.output}")


if __name__ == "__main__":
    main()
