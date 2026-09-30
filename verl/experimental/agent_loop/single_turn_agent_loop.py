# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
import logging
import os
from typing import Any
from uuid import uuid4

import torch
from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput, register
from verl.utils.profiler import simple_timer
from verl.workers.rollout.replica import TokenOutput

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def _build_student_teacher_image_views(
    clean_student_multi_modal_data: dict[str, Any],
    explicit_teacher_multi_modal_data: dict[str, Any] | None,
    *,
    enabled: bool,
    std: float,
    validate: bool,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Create the asymmetric student/teacher image views for one trajectory.

    The returned student view is reused by rollout and by actor postprocessing,
    so the actor forward scores the exact noisy image that produced the
    trajectory. The teacher always receives an unmodified clean view. An
    explicit teacher view (for example a privileged crop) takes precedence.
    """
    teacher_multi_modal_data = explicit_teacher_multi_modal_data
    if teacher_multi_modal_data is None and enabled and not validate:
        teacher_multi_modal_data = clean_student_multi_modal_data

    if not enabled or validate:
        return clean_student_multi_modal_data, teacher_multi_modal_data

    from verl.experimental.teacher_loop.teacher_manager import build_gaussian_near_multi_modal_data

    noisy_multi_modal_data = build_gaussian_near_multi_modal_data(
        clean_student_multi_modal_data,
        std=std,
    )
    # Text-only samples have no image to perturb. Preserve the original path.
    if noisy_multi_modal_data is None:
        noisy_multi_modal_data = clean_student_multi_modal_data
    return noisy_multi_modal_data, teacher_multi_modal_data


def _text_from_message_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for x in content:
            if isinstance(x, dict) and x.get("type") == "text":
                parts.append(str(x.get("text", "")))
        return "".join(parts)
    return ""


def _last_text_tail_from_messages(msgs: list | None, n: int = 320) -> str:
    """Take the last n characters of the last printable text content for comparing student / teacher."""
    if not msgs:
        return "<empty>"
    for m in reversed(msgs):
        if not isinstance(m, dict):
            continue
        t = _text_from_message_content(m.get("content"))
        if t.strip():
            t = t.replace("\n", " ")
            return ("..." + t[-n:]) if len(t) > n else t
    return repr(msgs)[:n]


@register("single_turn_agent")
class SingleTurnAgentLoop(AgentLoopBase):
    """Naive agent loop that only do single turn chat completion."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prompt_length = self.rollout_config.prompt_length
        self.response_length = self.rollout_config.response_length

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        messages = list(kwargs["raw_prompt"])
        teacher_messages = list(kwargs["teacher_raw_prompt"]) if kwargs.get("teacher_raw_prompt") is not None else None
        validate = bool(kwargs.get("validate", False))

        if os.environ.get("VERL_DEBUG_DUAL_PROMPT", "").lower() in ("1", "true", "yes"):
            idx = kwargs.get("index", None)
            try:
                idx_ok = int(idx) == 0 if idx is not None else False
            except (TypeError, ValueError):
                idx_ok = False
            if idx_ok:
                st = _last_text_tail_from_messages(messages)
                tt = (
                    _last_text_tail_from_messages(teacher_messages)
                    if teacher_messages is not None
                    else "<no teacher_raw_prompt>"
                )
                eq = st == tt if teacher_messages is not None else None
                print(
                    "[VERL_DEBUG_DUAL_PROMPT] index=0\n"
                    f"  student_tail: {st}\n"
                    f"  teacher_tail: {tt}\n"
                    f"  tail_equal: {eq}",
                    flush=True,
                )

        # 1. extract images and videos from messages
        clean_student_multi_modal_data = await self.process_vision_info(messages)
        teacher_multi_modal_data = None
        if teacher_messages is not None:
            teacher_multi_modal_data = await self.process_vision_info(teacher_messages)

        distillation_config = self.config.get("distillation")
        asymmetric_noise_enabled = bool(
            distillation_config
            and distillation_config.get("enabled", False)
            and distillation_config.get("student_noisy_teacher_clean_enabled", False)
        )
        asymmetric_noise_std = float(
            distillation_config.get("student_noisy_teacher_clean_std", 0.05)
            if distillation_config
            else 0.05
        )
        multi_modal_data, teacher_multi_modal_data = _build_student_teacher_image_views(
            clean_student_multi_modal_data,
            teacher_multi_modal_data,
            enabled=asymmetric_noise_enabled,
            std=asymmetric_noise_std,
            validate=validate,
        )
        images = multi_modal_data.get("images")
        videos = multi_modal_data.get("videos")

        # 2. apply chat template and tokenize
        # prompt_ids = await self.apply_chat_template(
        #     messages,
        #     images=images,
        #     videos=videos,
        # )
        prompt_ids_processor, prompt_ids_tokenizer = await self.apply_chat_template(
            messages,
            images=images,
            videos=videos,
        )
        teacher_prompt_ids_processor = None
        teacher_prompt_ids_tokenizer = None
        if teacher_messages is not None:
            # teacher uses an independent prompt (q + c), but still scores the same student trajectory y.
            teacher_prompt_ids_processor, teacher_prompt_ids_tokenizer = await self.apply_chat_template(
                teacher_messages,
                images=teacher_multi_modal_data.get("images"),
                videos=teacher_multi_modal_data.get("videos"),
            )

        # # ========== DEBUG print ==========
        # import sys
        # VIDEO_PAD = 151656    # <|video_pad|>
        # IMAGE_PAD = 151655    # <|image_pad|>
        # VISION_START = 151652 # <|vision_start|>
        # VISION_END = 151653   # <|vision_end|>

        # ids_tk = prompt_ids_tokenizer if isinstance(prompt_ids_tokenizer, list) else prompt_ids_tokenizer.tolist()
        # ids_pr = prompt_ids_processor if isinstance(prompt_ids_processor, list) else prompt_ids_processor.tolist()

        # # Tokenizer version
        # vpad_tk = ids_tk.count(VIDEO_PAD)
        # vstart_tk = ids_tk.count(VISION_START)
        # pattern_tk = sum(1 for i in range(len(ids_tk)-2)
        #                 if ids_tk[i]==VISION_START and ids_tk[i+1]==VIDEO_PAD and ids_tk[i+2]==VISION_END)

        # # Processor version
        # vpad_pr = ids_pr.count(VIDEO_PAD)
        # vstart_pr = ids_pr.count(VISION_START)
        # pattern_pr = sum(1 for i in range(len(ids_pr)-2)
        #                 if ids_pr[i]==VISION_START and ids_pr[i+1]==VIDEO_PAD and ids_pr[i+2]==VISION_END)

        # print(f"[DEBUG] TOKENIZER: len={len(ids_tk)}, <|video_pad|>={vpad_tk}, "
        #       f"<|vision_start|>={vstart_tk}, "
        #       f"start-vpad-end patterns={pattern_tk}")
        # print(f"[DEBUG] PROCESSOR: len={len(ids_pr)}, <|video_pad|>={vpad_pr}, "
        #       f"<|vision_start|>={vstart_pr}, "
        #       f"start-vpad-end patterns={pattern_pr}")
        # print(f"[DEBUG] videos={len(videos) if videos else 0}")

        # # Key: print what token follows each vision_start in tokenizer prompt
        # for i in range(len(ids_tk)-1):
        #     if ids_tk[i] == VISION_START:
        #         print(f"[DEBUG] TOKENIZER: vision_start at pos {i}, next token={ids_tk[i+1]}")
        # sys.stdout.flush()
        # # ========== DEBUG end ==========

        # 3. generate sequences
        metrics = {}
        with simple_timer("generate_sequences", metrics):
            output: TokenOutput = await self.server_manager.generate(
                request_id=uuid4().hex,
                prompt_ids=prompt_ids_tokenizer,
                sampling_params=sampling_params,
                image_data=images,
                video_data=videos,
            )
        if metrics.get("num_preempted") is None:
            metrics["num_preempted"] = output.num_preempted if output.num_preempted is not None else -1
        response_mask = [1] * len(output.token_ids)

        output: AgentLoopOutput = AgentLoopOutput(
            prompt_ids=prompt_ids_processor,
            response_ids=output.token_ids[: self.response_length],
            response_mask=response_mask[: self.response_length],
            response_logprobs=output.log_probs[: self.response_length] if output.log_probs else None,
            routed_experts=(
                output.routed_experts[: len(prompt_ids_processor) + self.response_length]
                if output.routed_experts is not None
                else None
            ),
            multi_modal_data=multi_modal_data,
            teacher_multi_modal_data=teacher_multi_modal_data,
            num_turns=2,
            metrics=metrics,
            extra_fields=output.extra_fields,
        )

        # keeping the schema consistent with tool_agent_loop
        output.extra_fields.update({"turn_scores": [], "tool_rewards": []})
        output.extra_fields["prompt_ids_tokenizer"] = prompt_ids_tokenizer
        output.extra_fields["has_teacher_prompt"] = teacher_prompt_ids_tokenizer is not None
        if teacher_prompt_ids_processor is not None and teacher_prompt_ids_tokenizer is not None:
            output.extra_fields["teacher_prompt_ids_processor"] = teacher_prompt_ids_processor
            output.extra_fields["teacher_prompt_ids_tokenizer"] = teacher_prompt_ids_tokenizer

        return output
