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

import asyncio

import numpy as np
import pytest
import torch
from PIL import Image
from tensordict import TensorDict

from verl.experimental.teacher_loop.teacher_manager import (
    AsyncTeacherLLMServerManager,
    build_blank_counterfactual_multi_modal_data,
)
from verl.protocol import DataProto
from verl.trainer.distillation.losses import apply_teacher_counterfactual_visual_weighting
from verl.workers.config import DistillationLossConfig
from verl.workers.utils.padding import left_right_2_no_padding


def test_blank_counterfactual_preserves_input_and_image_size():
    image = Image.new("RGB", (5, 3), color=(10, 20, 30))
    original = {"images": [image], "videos": ["unchanged"]}

    counterfactual = build_blank_counterfactual_multi_modal_data(original, blank_value=127)

    assert counterfactual is not original
    assert counterfactual["images"] is not original["images"]
    assert counterfactual["images"][0].size == image.size
    assert counterfactual["images"][0].getpixel((0, 0)) == (127, 127, 127)
    assert original["images"][0].getpixel((0, 0)) == (10, 20, 30)
    assert counterfactual["videos"] is original["videos"]


def test_disabled_visual_grounding_is_exact_noop():
    losses = torch.tensor([[1.0, -2.0]])
    config = DistillationLossConfig(loss_mode="k1", visual_grounding_enabled=False)

    weighted, metrics = apply_teacher_counterfactual_visual_weighting(losses, TensorDict({}, []), config)

    assert weighted is losses
    assert metrics == {}


def test_teacher_visual_shift_additively_upweights_existing_loss():
    # One sample: two prompt tokens followed by three response tokens. The
    # teacher tensors contain per-sequence token logprobs and are converted to
    # the same nested layout used by the actor training path.
    input_ids = torch.arange(5).unsqueeze(0)
    positive = torch.tensor([[[0.0], [-0.1], [-0.2], [-0.3], [0.0]]])
    negative = torch.tensor([[[0.0], [-1.1], [-0.2], [-0.8], [0.0]]])
    data = TensorDict(
        {
            "input_ids": input_ids,
            "prompts": input_ids[:, :2].clone(),
            "responses": input_ids[:, 2:].clone(),
            "attention_mask": torch.ones(1, 5, dtype=torch.long),
            "response_mask": torch.ones(1, 3, dtype=torch.long),
            "position_ids": input_ids.clone(),
            "teacher_ids": input_ids.unsqueeze(-1).to(torch.int32),
            "teacher_logprobs": positive,
            "teacher_negative_logprobs": negative,
        },
        batch_size=[1],
    )
    data = left_right_2_no_padding(data)
    # Match the actor input contract: prompt/response layout tensors stay
    # padded even though sequence-level teacher tensors are jagged.
    data["prompts"] = input_ids[:, :2].clone()
    data["responses"] = input_ids[:, 2:].clone()
    losses = torch.tensor([[1.0, 2.0, -1.0]])
    config = DistillationLossConfig(
        loss_mode="k1",
        visual_grounding_enabled=True,
        visual_grounding_weight_scale=2.0,
        visual_grounding_weight_threshold=0.1,
        visual_grounding_max_weight=3.0,
    )

    weighted, metrics = apply_teacher_counterfactual_visual_weighting(losses, data, config)

    # Shifts are [1.0, 0.0, 0.5], producing additive weights
    # [2.8, 1.0, 1.8] after the 0.1 threshold.
    torch.testing.assert_close(weighted, torch.tensor([[2.8, 2.0, -1.8]]))
    assert set(metrics) == {
        "distillation/visual_grounding_shift",
        "distillation/visual_grounding_active_fraction",
        "distillation/visual_grounding_weight",
        "distillation/visual_grounding_weight_max",
    }


def test_counterfactual_visual_grounding_rejects_topk_loss():
    with pytest.raises(ValueError, match="sampled-token"):
        DistillationLossConfig(loss_mode="forward_kl_topk", visual_grounding_enabled=True)


class _FakeTeacherManager:
    pad_token_id = 0

    def __init__(self, enabled: bool):
        self.distillation_loss_config = DistillationLossConfig(
            loss_mode="k1",
            visual_grounding_enabled=enabled,
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
        pixel = multi_modal_data["images"][0].getpixel((0, 0))[0]
        teacher_logprobs = torch.full_like(teacher_ids, fill_value=-pixel / 255, dtype=torch.float32)
        return teacher_ids, teacher_logprobs


def _make_teacher_batch() -> DataProto:
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
    multi_modal_data = {"images": [Image.new("RGB", (3, 2), color=(20, 30, 40))]}
    return DataProto(
        batch=batch,
        non_tensor_batch={"teacher_multi_modal_data": np.array([multi_modal_data], dtype=object)},
    )


def test_batched_teacher_disabled_keeps_single_score_and_original_schema():
    manager = _FakeTeacherManager(enabled=False)

    output = asyncio.run(AsyncTeacherLLMServerManager.compute_teacher_logprobs_batch(manager, _make_teacher_batch()))

    assert len(manager.calls) == 1
    assert set(output.batch.keys()) == {"teacher_ids", "teacher_logprobs"}


def test_batched_teacher_enabled_scores_real_and_blank_once_each():
    manager = _FakeTeacherManager(enabled=True)

    output = asyncio.run(AsyncTeacherLLMServerManager.compute_teacher_logprobs_batch(manager, _make_teacher_batch()))

    assert len(manager.calls) == 2
    assert manager.calls[0]["images"][0].getpixel((0, 0)) == (20, 30, 40)
    assert manager.calls[1]["images"][0].getpixel((0, 0)) == (127, 127, 127)
    assert "teacher_negative_logprobs" in output.batch
