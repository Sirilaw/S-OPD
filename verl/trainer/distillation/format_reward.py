# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Independent format reward utilities for on-policy distillation."""

from __future__ import annotations

import re
from typing import Any

import torch

from verl import DataProto
from verl.workers.config import DistillationLossConfig

# Permit punctuation and closing TeX math delimiters after the final box, e.g.
# ``$\boxed{A}$``, ``$$\boxed{A}$$``, or ``\[\boxed{A}\]``.
_TRAILING_FINAL_PUNCTUATION = re.compile(r"^[\s.!?。！？$\\\])]*$")


def _find_balanced_box_end(text: str, open_brace_index: int) -> int | None:
    r"""Return the index after the balanced ``\boxed{...}`` closing brace."""
    depth = 0
    for index in range(open_brace_index, len(text)):
        char = text[index]
        if char == "{" and (index == 0 or text[index - 1] != "\\"):
            depth += 1
        elif char == "}" and (index == 0 or text[index - 1] != "\\"):
            depth -= 1
            if depth == 0:
                return index + 1
            if depth < 0:
                return None
    return None


def score_boxed_format(response: str) -> float:
    r"""Score a completed response ending in one balanced ``\boxed{...}``."""
    text = response.strip()
    boxed_marker = "\\boxed{"
    if text.count(boxed_marker) != 1:
        return 0.0
    boxed_start = text.find(boxed_marker)
    open_brace_index = boxed_start + len(boxed_marker) - 1
    boxed_end = _find_balanced_box_end(text, open_brace_index)
    if boxed_end is None or not text[open_brace_index + 1 : boxed_end - 1].strip():
        return 0.0
    if not _TRAILING_FINAL_PUNCTUATION.fullmatch(text[boxed_end:]):
        return 0.0
    return 1.0


def score_think_boxed_format(response: str) -> float:
    r"""Score the stricter explicit-thought format.

    A valid response has exactly one non-empty ``<think>...</think>`` block,
    followed by exactly one balanced ``\boxed{...}`` final answer. Only
    whitespace or sentence-ending punctuation may follow the box. This rejects
    unfinished reasoning and responses that resume reasoning after the answer.
    """
    text = response.strip()
    if text.count("<think>") != 1 or text.count("</think>") != 1:
        return 0.0
    if not text.startswith("<think>"):
        return 0.0

    think_start = len("<think>")
    think_end = text.find("</think>", think_start)
    if think_end < 0 or not text[think_start:think_end].strip():
        return 0.0

    final_text = text[think_end + len("</think>") :].strip()
    return score_boxed_format(final_text)


def score_response_format(response: str, style: str = "boxed") -> float:
    """Dispatch to a configured built-in format scorer."""
    if style == "boxed":
        return score_boxed_format(response)
    if style == "think_boxed":
        return score_think_boxed_format(response)
    raise ValueError(f"Unsupported format reward style: {style!r}")


def add_format_reward_advantages(
    batch: DataProto,
    tokenizer: Any,
    loss_config: DistillationLossConfig,
) -> dict[str, float]:
    """Add token-broadcast format advantages to a rollout batch.

    This auxiliary signal is deliberately computed separately from
    ``token_level_rewards`` and ``advantages``. Consequently, enabling it does
    not enable task-reward PPO or alter the vanilla OPD advantage tensor.
    """
    if not loss_config.format_reward_enabled:
        return {}

    responses = batch.batch["responses"]
    response_mask = batch.batch["response_mask"].to(dtype=torch.float32)
    if responses.ndim != 2 or response_mask.shape != responses.shape:
        raise ValueError(
            "Format reward expects responses and response_mask with the same [batch, response_length] shape."
        )

    response_attention_mask = batch.batch["attention_mask"][:, -responses.shape[1] :]
    valid_lengths = response_attention_mask.sum(dim=-1).to(dtype=torch.long)
    max_response_length = responses.shape[1]
    scores: list[float] = []
    terminated: list[float] = []
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    eos_token_ids = {int(eos_token_id)} if isinstance(eos_token_id, int) else set()
    for row, valid_length_tensor in zip(responses, valid_lengths, strict=True):
        valid_length = int(valid_length_tensor.item())
        valid_token_ids = row[:valid_length].tolist()
        response_text = tokenizer.decode(valid_token_ids, skip_special_tokens=True)
        structure_score = score_response_format(response_text, style=loss_config.format_reward_style)
        did_terminate = valid_length < max_response_length or bool(eos_token_ids.intersection(valid_token_ids))
        if loss_config.format_reward_require_termination and not did_terminate:
            structure_score = 0.0
        scores.append(structure_score)
        terminated.append(float(did_terminate))

    score_tensor = torch.tensor(scores, dtype=torch.float32, device=response_mask.device)
    sequence_advantages = score_tensor - loss_config.format_reward_baseline
    batch.batch["format_reward_advantages"] = sequence_advantages.unsqueeze(-1) * response_mask

    return {
        "format_reward/score": float(score_tensor.mean().item()),
        "format_reward/valid_rate": float((score_tensor > 0).float().mean().item()),
        "format_reward/terminated_rate": float(sum(terminated) / len(terminated)),
        "format_reward/advantage": float(sequence_advantages.mean().item()),
    }
