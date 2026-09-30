"""Deterministic workload ordering and a process-safe local task queue.

The dynamic vLLM inference path uses one engine per torchrun rank.  All ranks
construct the same ordered list of unfinished samples and atomically claim the
next list position from :class:`FileTaskQueue`.  Keeping the queue state to a
single integer makes recovery simple: after a failed run, completed rank-local
checkpoints are merged and a fresh queue is created for only the missing rows.
"""

from __future__ import annotations

import fcntl
import json
import os
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

PROFILE_VERSION = 1
_MEDIA_KEYS = ("image", "video", "audio")
_IGNORED_KEYS = {
    "answer",
    "prediction",
    "thinking",
    "extra_records",
    "index",
}


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        missing = value != value
    except Exception:
        return False
    return isinstance(missing, bool) and missing


def _media_count(value: Any) -> int:
    if _is_missing(value):
        return 0
    if isinstance(value, (list, tuple)):
        return max(1, len(value))
    return 1


def estimate_row_cost(row: Mapping[str, Any], history: Mapping[str, int] | None = None) -> float:
    """Estimate relative generation work without loading image payloads.

    Historical output token counts dominate when available.  The input proxy
    still accounts for text length and media count, which provides a useful
    first-run ordering before a profile has been collected.
    """

    idx = str(row.get("index"))
    text_chars = 0
    media_units = 0
    for raw_key, value in row.items():
        key = str(raw_key).lower()
        if key in _IGNORED_KEYS or _is_missing(value):
            continue
        if any(media_key in key for media_key in _MEDIA_KEYS):
            media_units += _media_count(value)
            continue
        if isinstance(value, str):
            text_chars += len(value)
        elif isinstance(value, (int, float, bool)):
            text_chars += len(str(value))
        elif isinstance(value, Sequence):
            text_chars += sum(len(str(item)) for item in value if not _is_missing(item))

    # Rough prefill proxy: four characters per text token and about 1,024
    # visual tokens per media item.  It is only used for relative ordering.
    input_tokens = max(1.0, text_chars / 4.0 + media_units * 1024.0)
    historical_tokens = int((history or {}).get(idx, 0))
    if historical_tokens <= 0:
        return input_tokens

    # Decode tokens are sequential and therefore more expensive than an
    # equivalent amount of prefill work.  The exact multiplier is not used as
    # a runtime prediction; it only keeps known long generations at the front.
    return input_tokens + historical_tokens * 4.0


def order_rows_longest_first(rows: Sequence[Mapping[str, Any]], history: Mapping[str, int] | None = None) -> list[int]:
    """Return row positions sorted by descending estimated cost."""

    scored = [
        (estimate_row_cost(row, history), str(row.get("index")), position)
        for position, row in enumerate(rows)
    ]
    scored.sort(key=lambda item: (-item[0], item[1], item[2]))
    return [position for _, _, position in scored]


def balanced_partitions(
    rows: Sequence[Mapping[str, Any]],
    world_size: int,
    history: Mapping[str, int] | None = None,
) -> tuple[list[list[int]], list[float]]:
    """Greedy LPT partitioning used by the non-dynamic fallback path."""

    if world_size <= 0:
        raise ValueError("world_size must be positive")
    partitions = [[] for _ in range(world_size)]
    loads = [0.0] * world_size
    for position in order_rows_longest_first(rows, history):
        rank = min(range(world_size), key=lambda candidate: (loads[candidate], candidate))
        cost = estimate_row_cost(rows[position], history)
        partitions[rank].append(position)
        loads[rank] += cost
    return partitions, loads


def safe_profile_name(dataset_name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", dataset_name).strip("._") or "dataset"


def profile_path(profile_dir: str | os.PathLike[str], dataset_name: str) -> Path:
    return Path(profile_dir).expanduser() / f"{safe_profile_name(dataset_name)}.json"


def load_output_token_history(profile_dir: str | os.PathLike[str] | None, dataset_name: str) -> dict[str, int]:
    if not profile_dir:
        return {}
    path = profile_path(profile_dir, dataset_name)
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("version") != PROFILE_VERSION or payload.get("dataset") != dataset_name:
            return {}
        samples = payload.get("samples", {})
        return {
            str(index): int(record["output_tokens"])
            for index, record in samples.items()
            if isinstance(record, dict) and int(record.get("output_tokens", 0)) > 0
        }
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return {}


def write_output_token_history(
    profile_dir: str | os.PathLike[str] | None,
    dataset_name: str,
    output_tokens: Mapping[Any, int],
) -> Path | None:
    if not profile_dir:
        return None
    path = profile_path(profile_dir, dataset_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": PROFILE_VERSION,
        "dataset": dataset_name,
        "samples": {
            str(index): {"output_tokens": max(1, int(tokens))}
            for index, tokens in output_tokens.items()
        },
    }
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp_path, path)
    return path


class FileTaskQueue:
    """A tiny cross-process fetch-and-increment queue backed by ``flock``."""

    def __init__(self, path: str | os.PathLike[str], total: int):
        if total < 0:
            raise ValueError("total must be non-negative")
        self.path = Path(path)
        self.total = total

    def initialize(self, start_position: int = 0) -> None:
        if not 0 <= start_position <= self.total:
            raise ValueError("start_position must be within [0, total]")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.path.with_name(f".{self.path.name}.{os.getpid()}.tmp")
        tmp_path.write_text(f"{start_position}\n", encoding="ascii")
        os.replace(tmp_path, self.path)

    def claim(self) -> int | None:
        with self.path.open("r+", encoding="ascii") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            raw = handle.read().strip()
            position = int(raw or "0")
            if position >= self.total:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
                return None
            handle.seek(0)
            handle.write(f"{position + 1}\n")
            handle.truncate()
            handle.flush()
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            return position
