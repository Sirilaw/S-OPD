# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

import asyncio

import numpy as np
import pytest
import torch
from PIL import Image
from tensordict import TensorDict

from verl.experimental.agent_loop.agent_loop import AgentLoopMetrics, AgentLoopOutput, AgentLoopWorker
from verl.experimental.teacher_loop.teacher_manager import (
    AsyncTeacherLLMServerManager,
    build_pixelated_counterfactual_multi_modal_data,
)
from verl.protocol import DataProto
from verl.trainer.distillation.losses import compute_token_level_va_group_weights
from verl.trainer.ppo.core_algos import agg_loss
from verl.workers.config import DistillationLossConfig
from verl.workers.utils.padding import left_right_2_no_padding


def test_pixelated_counterfactual_preserves_layout_and_input():
    image = Image.new("RGB", (20, 10))
    image.putdata(
        [(255, 255, 255) if (x + y) % 2 else (0, 0, 0) for y in range(image.height) for x in range(image.width)]
    )
    original_pixels = image.tobytes()
    original = {"images": [image], "videos": ["unchanged"]}

    counterfactual = build_pixelated_counterfactual_multi_modal_data(original, pixelation_ratio=0.1)

    assert counterfactual is not original
    assert counterfactual["images"] is not original["images"]
    assert counterfactual["images"][0].size == image.size
    assert image.tobytes() == original_pixels
    assert counterfactual["images"][0].tobytes() != original_pixels
    # 20x10 -> 2x1 -> 20x10, so nearest-neighbor restoration has at most two colors.
    assert len(counterfactual["images"][0].getcolors(maxcolors=3)) <= 2
    assert counterfactual["videos"] is original["videos"]


def _make_loss_data(positive_response, negative_response, response_mask=None):
    response_length = len(positive_response)
    input_ids = torch.arange(2 + response_length).unsqueeze(0)
    if response_mask is None:
        response_mask = torch.ones(1, response_length, dtype=torch.long)
    attention_mask = torch.cat([torch.ones(1, 2, dtype=torch.long), response_mask], dim=1)
    # Sequence logprobs predict the next token. The first response token's
    # score therefore lives at the final prompt position; the final sequence
    # position is an unused next-token score.
    positive = torch.tensor([[[0.0], *[[value] for value in positive_response], [0.0]]])
    negative = torch.tensor([[[0.0], *[[value] for value in negative_response], [0.0]]])
    data = TensorDict(
        {
            "input_ids": input_ids,
            "prompts": input_ids[:, :2].clone(),
            "responses": input_ids[:, 2:].clone(),
            "attention_mask": attention_mask,
            "response_mask": response_mask,
            "position_ids": input_ids.clone(),
            "teacher_ids": input_ids.unsqueeze(-1).to(torch.int32),
            "teacher_logprobs": positive,
            "teacher_negative_logprobs": negative,
        },
        batch_size=[1],
    )
    data = left_right_2_no_padding(data)
    data["prompts"] = input_ids[:, :2].clone()
    data["responses"] = input_ids[:, 2:].clone()
    return data


def test_token_va_selects_exact_top_fraction_and_normalizes_each_group():
    # Rectified VA is [0.1, 2.0, 0.0, 1.0, 0.5]. With p_v=0.2, exactly
    # one token is high-VA. Its group gets 0.5 total weight; the other four
    # tokens share the remaining 0.5.
    data = _make_loss_data(
        positive_response=[0.1, 2.0, -1.0, 1.0, 0.5],
        negative_response=[0.0, 0.0, 0.0, 0.0, 0.0],
    )
    losses = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])
    config = DistillationLossConfig(loss_mode="k1", token_level_va_enabled=True)

    weights, metrics = compute_token_level_va_group_weights(losses, data, config)

    torch.testing.assert_close(weights, torch.tensor([[0.125, 0.5, 0.125, 0.125, 0.125]]))
    torch.testing.assert_close(weights.sum(dim=-1), torch.ones(1))
    assert metrics["distillation/token_va_mean"].aggregate() == pytest.approx(0.72)
    assert metrics["distillation/token_va_max"].aggregate() == pytest.approx(2.0)
    assert metrics["distillation/token_va_positive_fraction"].aggregate() == pytest.approx(0.8)


def test_token_va_grouped_aggregation_matches_paper_equation_five():
    data = _make_loss_data(
        positive_response=[0.1, 2.0, -1.0, 1.0, 0.5],
        negative_response=[0.0, 0.0, 0.0, 0.0, 0.0],
    )
    losses = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])
    config = DistillationLossConfig(loss_mode="k1", token_level_va_enabled=True)
    weights, _ = compute_token_level_va_group_weights(losses, data, config)

    grouped = agg_loss(losses * weights, data["response_mask"], "seq-mean-token-sum")
    expected = 0.5 * losses[0, 1] + 0.5 * losses[0, [0, 2, 3, 4]].mean()

    torch.testing.assert_close(grouped, expected)


def test_token_va_handles_single_valid_token_without_empty_group():
    data = _make_loss_data([1.0], [0.0])
    config = DistillationLossConfig(loss_mode="k1", token_level_va_enabled=True)

    weights, _ = compute_token_level_va_group_weights(torch.tensor([[3.0]]), data, config)

    torch.testing.assert_close(weights, torch.ones(1, 1))


def test_token_va_config_is_default_off_and_rejects_incompatible_modes():
    assert DistillationLossConfig(loss_mode="k1").token_level_va_enabled is False
    with pytest.raises(ValueError, match="independent experimental mode"):
        DistillationLossConfig(
            loss_mode="k1",
            token_level_va_enabled=True,
            visual_grounding_enabled=True,
        )
    with pytest.raises(ValueError, match="sampled-token"):
        DistillationLossConfig(loss_mode="forward_kl_topk", token_level_va_enabled=True)

    combined = DistillationLossConfig(
        loss_mode="k1",
        token_level_va_enabled=True,
        papo_enabled=True,
        gaussian_near_pull_enabled=True,
    )
    assert combined.token_level_va_enabled
    assert combined.papo_enabled
    assert combined.gaussian_near_pull_enabled


class _FakeTokenVATeacherManager:
    pad_token_id = 0

    def __init__(self):
        self.distillation_loss_config = DistillationLossConfig(
            loss_mode="k1",
            token_level_va_enabled=True,
        )
        self.calls = []

    async def compute_teacher_logprobs_single(
        self,
        sequence_ids,
        multi_modal_data=None,
        expected_len=None,
    ):
        del expected_len
        self.calls.append(multi_modal_data)
        teacher_ids = torch.tensor(sequence_ids, dtype=torch.int32).unsqueeze(-1)
        teacher_logprobs = torch.zeros_like(teacher_ids, dtype=torch.float32)
        return teacher_ids, teacher_logprobs


class _FakeTokenVAPushPullTeacherManager(_FakeTokenVATeacherManager):
    def __init__(self):
        super().__init__()
        self.distillation_loss_config = DistillationLossConfig(
            loss_mode="k1",
            token_level_va_enabled=True,
            papo_enabled=True,
            gaussian_near_pull_enabled=True,
        )


def test_batched_teacher_scores_original_and_pixelated_image_once_each():
    input_ids = torch.arange(4).unsqueeze(0)
    batch = TensorDict(
        {
            "input_ids": input_ids,
            "prompts": input_ids[:, :2],
            "responses": input_ids[:, 2:],
            "attention_mask": torch.ones(1, 4, dtype=torch.long),
        },
        batch_size=[1],
    )
    image = Image.new("RGB", (20, 10))
    image.putdata([(255, 255, 255) if (x + y) % 2 else (0, 0, 0) for y in range(10) for x in range(20)])
    multi_modal_data = {"images": [image]}
    data = DataProto(
        batch=batch,
        non_tensor_batch={"teacher_multi_modal_data": np.array([multi_modal_data], dtype=object)},
    )
    manager = _FakeTokenVATeacherManager()

    output = asyncio.run(AsyncTeacherLLMServerManager.compute_teacher_logprobs_batch(manager, data))

    assert len(manager.calls) == 2
    assert manager.calls[0]["images"][0] is image
    assert manager.calls[1]["images"][0].size == image.size
    assert manager.calls[1]["images"][0].tobytes() != image.tobytes()
    assert "teacher_negative_logprobs" in output.batch


def test_va_pushpull_batched_teacher_keeps_pixelated_and_masked_scores_separate():
    input_ids = torch.arange(4).unsqueeze(0)
    batch = TensorDict(
        {
            "input_ids": input_ids,
            "prompts": input_ids[:, :2],
            "responses": input_ids[:, 2:],
            "attention_mask": torch.ones(1, 4, dtype=torch.long),
        },
        batch_size=[1],
    )
    image = Image.new("RGB", (20, 10))
    image.putdata(
        [(255, 255, 255) if (x + y) % 2 else (0, 0, 0) for y in range(10) for x in range(20)]
    )
    masked = Image.new("RGB", image.size, color=(17, 17, 17))
    data = DataProto(
        batch=batch,
        non_tensor_batch={
            "teacher_multi_modal_data": np.array([{"images": [image]}], dtype=object),
            "teacher_counterfactual_multi_modal_data": np.array(
                [{"images": [masked]}], dtype=object
            ),
        },
    )
    manager = _FakeTokenVAPushPullTeacherManager()

    output = asyncio.run(AsyncTeacherLLMServerManager.compute_teacher_logprobs_batch(manager, data))

    assert len(manager.calls) == 3
    assert manager.calls[0]["images"][0] is image
    assert manager.calls[1]["images"][0].tobytes() != image.tobytes()
    assert manager.calls[2]["images"][0] is masked
    assert "teacher_negative_logprobs" in output.batch
    assert "teacher_counterfactual_logprobs" in output.batch


def test_va_pushpull_streaming_teacher_keeps_pixelated_and_masked_scores_separate():
    image = Image.new("RGB", (20, 10))
    image.putdata(
        [(255, 255, 255) if (x + y) % 2 else (0, 0, 0) for y in range(10) for x in range(20)]
    )
    masked = Image.new("RGB", image.size, color=(17, 17, 17))
    teacher_manager = _FakeTokenVAPushPullTeacherManager()

    class _Worker:
        stream_teacher_with_rollout = True
        distillation_loss_config = teacher_manager.distillation_loss_config
        teacher_server_manager = teacher_manager

    output = AgentLoopOutput(
        prompt_ids=[0, 1],
        response_ids=[2, 3],
        response_mask=[1, 1],
        multi_modal_data={"images": [image]},
        metrics=AgentLoopMetrics(),
        extra_fields={"counterfactual_multi_modal_data": {"images": [masked]}},
    )

    asyncio.run(
        AgentLoopWorker._compute_teacher_logprobs(
            _Worker(), output, prompt_ids=[0, 1], response_ids=[2, 3], validate=False
        )
    )

    assert len(teacher_manager.calls) == 3
    assert teacher_manager.calls[0]["images"][0] is image
    assert teacher_manager.calls[1]["images"][0].tobytes() != image.tobytes()
    assert teacher_manager.calls[2]["images"][0] is masked
    assert "teacher_negative_logprobs" in output.extra_fields
    assert "teacher_counterfactual_logprobs" in output.extra_fields
