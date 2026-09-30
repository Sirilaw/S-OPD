"""Local visual-math benchmark adapter for VLMEvalKit.

The adapter intentionally does not download data.  It accepts the common
VisionR1/Geometry3K exports (JSON, JSONL, CSV, TSV and parquet) and resolves
their image paths relative to the manifest or an explicit image root.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
import os
import random
import re
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image, ImageDraw

from vlmeval.dataset.image_base import ImageBaseDataset
from vlmeval.smp import dump, load


FINAL_ANSWER_PROMPT = (
    "Solve the problem using the image. Show your reasoning briefly, then put "
    "only the final answer inside \\boxed{...}."
)

VISIONR1_USER_SUFFIX = (
    "You FIRST think about the reasoning process as an internal monologue and then provide the final answer. "
    "The reasoning process MUST BE enclosed within <think> </think> tags. "
    "The final answer MUST BE put in \\boxed{}."
)


def _read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".jsonl", ".ndjson"}:
        return pd.read_json(path, lines=True)
    if suffix == ".json":
        obj = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(obj, dict):
            for key in ("data", "test", "validation", "examples", "records"):
                if isinstance(obj.get(key), list):
                    obj = obj[key]
                    break
        return pd.DataFrame(obj)
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if suffix in {".tsv", ".txt"}:
        return pd.read_csv(path, sep="\t")
    raise ValueError(f"Unsupported manifest type: {path}")


def _first_column(frame: pd.DataFrame, requested: str | None, candidates: tuple[str, ...], kind: str) -> str:
    if requested:
        if requested not in frame:
            raise KeyError(f"{kind} column {requested!r} is absent; columns={list(frame.columns)}")
        return requested
    for name in candidates:
        if name in frame:
            return name
    raise KeyError(f"Cannot infer {kind} column; pass --{kind}-field. columns={list(frame.columns)}")


def _mapping(value: Any) -> dict[str, Any]:
    """Return dict-like parquet cells as plain mappings when possible."""
    if isinstance(value, dict):
        return value
    if value is None:
        return {}
    for parser in (json.loads, ast.literal_eval):
        try:
            parsed = parser(str(value))
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass
    return {}


def _message_text(value: Any) -> str | None:
    """Extract the user text from VERL/OpenAI-style message arrays."""
    if isinstance(value, dict):
        value = [value]
    elif isinstance(value, (tuple, list)) or getattr(value, "ndim", 0) == 1:
        value = list(value)
    else:
        return None
    for message in reversed(value):
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        content = message.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = [str(item.get("text", "")) for item in content if isinstance(item, dict) and item.get("type") == "text"]
            return "\n".join(part for part in parts if part)
    return None


def _strip_user_suffix(text: str, suffix: str) -> str:
    """Remove one trailing dataset-provided instruction suffix."""
    normalized = text.rstrip()
    if normalized.endswith(suffix):
        return normalized[: -len(suffix)].rstrip()
    return text


def _row_answer(row: pd.Series, answer_column: str | None) -> Any:
    if answer_column is not None:
        return row[answer_column]
    reward_model = _mapping(row.get("reward_model"))
    for key in ("ground_truth", "answer", "target"):
        if reward_model.get(key) is not None:
            return reward_model[key]
    extra_info = _mapping(row.get("extra_info"))
    for key in ("answer", "ground_truth", "target"):
        if extra_info.get(key) is not None:
            return extra_info[key]
    raise KeyError("Cannot infer answer from reward_model or extra_info")


def _as_list(value: Any) -> list[str]:
    if isinstance(value, (list, tuple)):
        return [str(x) for x in value]
    if isinstance(value, dict):
        for key in ("path", "image", "filename", "file_name"):
            if key in value:
                return [str(value[key])]
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return []
    text = str(value).strip()
    if text.startswith("[") or text.startswith("{"):
        for parser in (json.loads, ast.literal_eval):
            try:
                return _as_list(parser(text))
            except Exception:
                pass
    return [text]


def _image_values(value: Any) -> list[Any]:
    """Preserve HF parquet image dictionaries while flattening containers."""
    if isinstance(value, dict):
        return [value]
    if isinstance(value, (list, tuple)) or getattr(value, "ndim", 0) == 1:
        return list(value)
    return [value]


def _json_safe_record(
    row: pd.Series | dict[str, Any], excluded_columns: str | tuple[str, ...]
) -> dict[str, Any]:
    """Copy metadata without retaining raw image or structured prompt payloads."""
    if isinstance(excluded_columns, str):
        excluded_columns = (excluded_columns,)
    return {key: value for key, value in row.items() if key not in excluded_columns}


def _materialize_images(value: Any, root: Path, cache: Path, row_index: Any) -> list[str]:
    cache.mkdir(parents=True, exist_ok=True)
    paths = []
    for image_index, item in enumerate(_image_values(value)):
        if isinstance(item, dict) and item.get("bytes") is not None:
            payload = bytes(item["bytes"])
            if payload.startswith(b"\x89PNG\r\n\x1a\n"):
                suffix = ".png"
            elif payload.startswith(b"\xff\xd8\xff"):
                suffix = ".jpg"
            elif payload.startswith((b"GIF87a", b"GIF89a")):
                suffix = ".gif"
            elif payload.startswith(b"RIFF") and payload[8:12] == b"WEBP":
                suffix = ".webp"
            else:
                suffix = ".img"
            digest = hashlib.sha1(payload).hexdigest()[:16]
            path = cache / f"{row_index}_{image_index}_{digest}{suffix}"
            if not path.is_file():
                path.write_bytes(payload)
        else:
            if isinstance(item, dict):
                item = next((item[k] for k in ("path", "image", "filename", "file_name") if item.get(k)), None)
            if item is None:
                continue
            path = Path(str(item)).expanduser()
            if not path.is_absolute():
                path = root / path
        paths.append(str(path.resolve()))
    return paths


def _random_patch_mask_images(
    paths: list[str],
    cache: Path,
    row_index: Any,
    mask_ratio: float,
    patch_size: int,
    seed: int,
) -> list[str]:
    """Create deterministic PAPO-style random-patch masked image copies."""
    if not 0.0 <= mask_ratio <= 1.0:
        raise ValueError("image_mask_ratio must be in [0, 1].")
    if patch_size <= 0:
        raise ValueError("image_mask_patch_size must be positive.")
    if seed < 0:
        raise ValueError("image_mask_seed must be non-negative.")
    if mask_ratio == 0.0:
        return paths

    output_dir = cache / f"random_patch_mask_r{mask_ratio:g}_p{patch_size}_s{seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    masked_paths = []
    for image_index, source_value in enumerate(paths):
        source = Path(source_value).resolve()
        stat = source.stat()
        cache_key = (
            f"{source}:{stat.st_size}:{stat.st_mtime_ns}:{row_index}:"
            f"{image_index}:{mask_ratio}:{patch_size}:{seed}"
        )
        digest = hashlib.sha1(cache_key.encode("utf-8")).hexdigest()[:20]
        destination = output_dir / f"{row_index}_{image_index}_{digest}.png"
        if not destination.is_file():
            with Image.open(source) as opened:
                image = opened.convert("RGB")
            draw = ImageDraw.Draw(image)
            sample_seed = int.from_bytes(
                hashlib.sha256(f"{seed}:{row_index}:{image_index}".encode("utf-8")).digest()[:8],
                byteorder="big",
            )
            rng = random.Random(sample_seed)
            for top in range(0, image.height, patch_size):
                for left in range(0, image.width, patch_size):
                    if rng.random() < mask_ratio:
                        draw.rectangle(
                            (
                                left,
                                top,
                                min(left + patch_size, image.width) - 1,
                                min(top + patch_size, image.height) - 1,
                            ),
                            fill=(0, 0, 0),
                        )
            with tempfile.NamedTemporaryFile(
                suffix=".png", dir=output_dir, delete=False
            ) as handle:
                temporary = Path(handle.name)
            try:
                image.save(temporary, format="PNG")
                os.replace(temporary, destination)
            finally:
                temporary.unlink(missing_ok=True)
        masked_paths.append(str(destination))
    return masked_paths


def _last_boxed(text: str) -> str | None:
    starts = list(re.finditer(r"\\boxed\s*\{", text))
    for match in reversed(starts):
        depth, begin = 1, match.end()
        for pos in range(begin, len(text)):
            if text[pos] == "{":
                depth += 1
            elif text[pos] == "}":
                depth -= 1
                if depth == 0:
                    return text[begin:pos]
    return None


def extract_answer(value: Any) -> str:
    text = str(value or "").strip()
    boxed = _last_boxed(text)
    if boxed is not None:
        return boxed.strip()
    matches = re.findall(r"(?i)(?:final answer|answer)\s*(?:is|:|=)\s*([^\n]+)", text)
    if matches:
        return matches[-1].strip()
    return text.splitlines()[-1].strip() if text else ""


def has_complete_boxed_answer(value: Any) -> bool:
    boxed = _last_boxed(str(value or ""))
    return boxed is not None and bool(boxed.strip())


def resolve_choice_answer(value: Any, choices: Any) -> str:
    """Map a boxed choice letter or ``A. value`` to the corresponding value."""
    extracted = extract_answer(value).strip()
    options = _as_list(choices)
    if not options:
        return extracted
    cleaned = extracted.replace("\\ ", " ").strip()
    letter_match = re.match(r"^\s*([A-Za-z])(?:\s*[.):\-]\s*|\s+)(.*)$", cleaned, re.S)
    if letter_match:
        index = ord(letter_match.group(1).upper()) - ord("A")
        if 0 <= index < len(options):
            return options[index]
    if re.fullmatch(r"\s*[A-Za-z]\s*", cleaned):
        index = ord(cleaned.strip().upper()) - ord("A")
        if 0 <= index < len(options):
            return options[index]
    return extracted


def _normalize(value: Any) -> str:
    text = extract_answer(value)
    text = re.sub(r"(?i)^\s*(?:option|choice)\s*", "", text)
    text = text.replace("$", "").replace("\\,", "").replace("\\!", "")
    text = text.replace("\\left", "").replace("\\right", "")
    text = text.replace("−", "-").replace("–", "-").replace("π", "pi")
    text = re.sub(r"\\(?:mathrm|text)\s*\{([^{}]*)\}", r"\1", text)
    text = re.sub(r"\s+", "", text).strip(".。")
    if re.fullmatch(r"\(?[A-Za-z]\)?", text):
        return text.strip("()").upper()
    return text.lower()


def _number(value: str) -> float | None:
    value = value.replace(",", "")
    frac = re.fullmatch(r"(?:\\frac\{([^{}]+)\}\{([^{}]+)\}|([-+]?\d+(?:\.\d+)?)/([-+]?\d+(?:\.\d+)?))", value)
    try:
        if frac:
            a, b = (frac.group(1), frac.group(2)) if frac.group(1) is not None else (frac.group(3), frac.group(4))
            return float(a) / float(b)
        return float(value.rstrip("%")) / (100.0 if value.endswith("%") else 1.0)
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def answers_equal(prediction: Any, reference: Any, rel_tol: float = 1e-4, abs_tol: float = 1e-6) -> bool:
    pred, ref = _normalize(prediction), _normalize(reference)
    if pred == ref:
        return True
    pred_num, ref_num = _number(pred), _number(ref)
    if pred_num is not None and ref_num is not None:
        return math.isclose(pred_num, ref_num, rel_tol=rel_tol, abs_tol=abs_tol)
    # Some datasets store several accepted answers as "a || b".
    return any(pred == _normalize(candidate) for candidate in str(reference).split("||"))


def answers_equal_with_choices(
    prediction: Any,
    reference: Any,
    choices: Any,
    rel_tol: float = 1e-4,
    abs_tol: float = 1e-6,
) -> bool:
    resolved = resolve_choice_answer(prediction, choices)
    return answers_equal(resolved, reference, rel_tol=rel_tol, abs_tol=abs_tol)


class LocalMathDataset(ImageBaseDataset):
    """VLMEvalKit dataset class for local VisionR1/Geometry3K-style data."""

    TYPE = "VQA"
    MODALITY = "IMAGE"

    def __init__(
        self,
        dataset: str,
        data_file: str,
        image_root: str | None = None,
        cache_dir: str | None = None,
        question_field: str | None = None,
        answer_field: str | None = None,
        image_field: str | None = None,
        prompt: str = FINAL_ANSWER_PROMPT,
        strip_user_suffix: bool = False,
        user_suffix: str = VISIONR1_USER_SUFFIX,
        image_mask_ratio: float = 0.0,
        image_mask_patch_size: int = 14,
        image_mask_seed: int = 0,
        require_boxed_answer: bool = False,
    ) -> None:
        manifest = Path(data_file).expanduser().resolve()
        frame = _read_table(manifest)
        qcol = _first_column(frame, question_field, ("question", "problem", "query", "prompt", "instruction"), "question")
        answer_candidates = ("answer", "final_answer", "label", "target", "solution")
        if answer_field or any(name in frame for name in answer_candidates):
            acol = _first_column(frame, answer_field, answer_candidates, "answer")
        elif "reward_model" in frame or "extra_info" in frame:
            acol = None
        else:
            raise KeyError(
                "Cannot infer answer column or nested reward_model/extra_info answer; "
                f"pass --answer-field. columns={list(frame.columns)}"
            )
        icol = _first_column(frame, image_field, ("image_path", "image", "images", "figure", "diagram"), "image")
        root = Path(image_root).expanduser().resolve() if image_root else manifest.parent
        cache = Path(cache_dir).expanduser().resolve() if cache_dir else manifest.parent / ".vlmevalkit_images"
        cache.mkdir(parents=True, exist_ok=True)

        rows = []
        for offset, row in frame.iterrows():
            paths = _materialize_images(row[icol], root, cache, offset)
            if not paths or not all(Path(path).is_file() for path in paths):
                missing = [path for path in paths if not Path(path).is_file()] or [str(row[icol])]
                raise FileNotFoundError(f"Missing image(s) at row {offset}: {missing}")
            paths = _random_patch_mask_images(
                paths,
                cache,
                row.get("index", offset),
                image_mask_ratio,
                image_mask_patch_size,
                image_mask_seed,
            )
            # The source image column can contain raw ``bytes`` and the source
            # question/prompt column can be a NumPy object array. Both break
            # VLMEvalKit's final prediction JSON after inference completes.
            # ``image_path`` and the normalized string ``question`` below are
            # their canonical JSON-safe replacements.
            record = _json_safe_record(row, (icol, qcol))
            message_question = _message_text(row[qcol])
            question_value = message_question if message_question is not None else row[qcol]
            question = re.sub(r"<image>\s*", "", str(question_value)).strip()
            if strip_user_suffix:
                question = _strip_user_suffix(question, user_suffix)
            record.update(
                index=str(row.get("index", offset)),
                question=question,
                answer=str(_row_answer(row, acol)),
                _prompt_is_complete=message_question is not None,
            )
            record["image_path"] = paths[0] if len(paths) == 1 else paths
            rows.append(record)

        self.dataset_name = dataset
        self.data_file = str(manifest)
        self.data = pd.DataFrame(rows)
        self.meta_only = True
        self.skip_noimg = False
        self.img_root = str(root)
        self.prompt_suffix = prompt
        self.require_boxed_answer = require_boxed_answer

    def build_prompt(self, line):
        if isinstance(line, int):
            line = self.data.iloc[line]
        images = _as_list(line["image_path"])
        question = str(line["question"]).strip()
        row_instruction = line.get("final_instruction")
        instruction = row_instruction.strip() if isinstance(row_instruction, str) and row_instruction.strip() else self.prompt_suffix
        add_suffix = instruction and not bool(line.get("_prompt_is_complete", False))
        text = f"{question}\n\n{instruction}" if add_suffix else question
        return [*(dict(type="image", value=p) for p in images), dict(type="text", value=text)]

    def evaluate(self, eval_file, **judge_kwargs):
        del judge_kwargs
        frame = load(eval_file)
        frame["extracted_prediction"] = frame["prediction"].map(extract_answer)
        if "choices_json" in frame:
            frame["scored_prediction"] = [
                resolve_choice_answer(prediction, choices)
                for prediction, choices in zip(frame["prediction"], frame["choices_json"])
            ]
            content_correct = [
                answers_equal_with_choices(prediction, answer, choices)
                for prediction, answer, choices in zip(
                    frame["prediction"], frame["answer"], frame["choices_json"]
                )
            ]
        else:
            frame["scored_prediction"] = frame["extracted_prediction"]
            content_correct = [
                answers_equal(prediction, answer)
                for prediction, answer in zip(frame["prediction"], frame["answer"])
            ]
        frame["answer_format_valid"] = frame["prediction"].map(has_complete_boxed_answer)
        frame["correct"] = [
            bool(content_ok) and (bool(format_ok) or not self.require_boxed_answer)
            for content_ok, format_ok in zip(content_correct, frame["answer_format_valid"])
        ]
        scored_file = str(Path(eval_file).with_name(Path(eval_file).stem + "_scored.xlsx"))
        dump(frame, scored_file)
        metrics = {"Overall": round(float(frame["correct"].mean() * 100), 4), "Count": int(len(frame))}
        if "category" in frame:
            for category, group in frame.groupby("category", dropna=False):
                metrics[f"category={category}"] = round(float(group["correct"].mean() * 100), 4)
        return metrics
