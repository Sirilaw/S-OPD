import asyncio

import numpy as np
import pytest
import torch
from PIL import Image
from tensordict import TensorDict

from verl.experimental.teacher_loop.teacher_manager import (
    AsyncTeacherLLMServerManager,
    build_random_mask_counterfactual_multi_modal_data,
)
from verl.protocol import DataProto
from verl.trainer.distillation.losses import compute_counterfactual_residual_matching_loss
from verl.workers.config import DistillationLossConfig
from verl.workers.utils.padding import left_right_2_no_padding


def test_random_mask_blackens_patches_without_mutating_input():
    original = {"images": [Image.new("RGB", (17, 15), color="red")], "videos": ["keep"]}

    result = build_random_mask_counterfactual_multi_modal_data(
        original,
        mask_ratio=1.0,
        patch_size=14,
    )

    assert result is not original
    assert result["images"][0].size == (17, 15)
    assert result["images"][0].getbbox() is None
    assert original["images"][0].getpixel((0, 0)) == (255, 0, 0)
    assert result["videos"] is original["videos"]


def test_random_mask_samples_each_patch_independently():
    original = {"images": [Image.new("RGB", (28, 28), color="red")]}
    generator = torch.Generator().manual_seed(0)

    result = build_random_mask_counterfactual_multi_modal_data(
        original,
        mask_ratio=0.6,
        patch_size=14,
        generator=generator,
    )

    # torch.rand(seed=0) samples [0.496, 0.768; 0.088, 0.132], so only the
    # top-right patch remains visible at a masking probability of 0.6.
    image = result["images"][0]
    assert image.getpixel((7, 7)) == (0, 0, 0)
    assert image.getpixel((21, 7)) == (255, 0, 0)
    assert image.getpixel((7, 21)) == (0, 0, 0)
    assert image.getpixel((21, 21)) == (0, 0, 0)


def test_residual_mode_is_mutually_exclusive_with_old_weighting():
    with pytest.raises(ValueError, match="cannot be enabled together"):
        DistillationLossConfig(
            loss_mode="k1",
            visual_grounding_enabled=True,
            counterfactual_residual_enabled=True,
        )


class _ActorConfig:
    loss_agg_mode = "token-mean"
    global_batch_info = {}


def _make_residual_loss_inputs():
    input_ids = torch.arange(5).unsqueeze(0)
    data = TensorDict(
        {
            "input_ids": input_ids,
            "prompts": input_ids[:, :2].clone(),
            "responses": input_ids[:, 2:].clone(),
            "attention_mask": torch.ones(1, 5, dtype=torch.long),
            "response_mask": torch.ones(1, 3, dtype=torch.long),
            "position_ids": input_ids.clone(),
            "teacher_ids": input_ids.unsqueeze(-1).to(torch.int32),
            # After response slicing these yield teacher residual [1, 0, -0.5].
            "teacher_logprobs": torch.tensor([[[0.0], [0.0], [-0.2], [-0.8], [0.0]]]),
            "teacher_counterfactual_logprobs": torch.tensor(
                [[[0.0], [-1.0], [-0.2], [-0.3], [0.0]]]
            ),
        },
        batch_size=[1],
    )
    data = left_right_2_no_padding(data)
    data["prompts"] = input_ids[:, :2].clone()
    data["responses"] = input_ids[:, 2:].clone()
    offsets = data["input_ids"].offsets()
    student_positive = torch.nested.nested_tensor_from_jagged(
        torch.tensor([0.0, 0.0, -0.2, -0.8, 0.0]), offsets
    )
    student_counterfactual = torch.nested.nested_tensor_from_jagged(
        torch.tensor([0.0, -1.0, -0.2, -0.3, 0.0]), offsets
    )
    return data, {
        "log_probs": student_positive,
        "counterfactual_log_probs": student_counterfactual,
    }


def test_matching_residuals_have_zero_loss_with_uniform_token_weights():
    data, model_output = _make_residual_loss_inputs()
    config = DistillationLossConfig(loss_mode="k1", counterfactual_residual_enabled=True)

    loss, metrics = compute_counterfactual_residual_matching_loss(
        model_output, data, _ActorConfig(), config
    )

    torch.testing.assert_close(loss, torch.tensor(0.0))
    assert metrics["distillation/counterfactual_residual_gap_abs"].aggregate() == pytest.approx(0.0)


def test_zero_teacher_residual_token_still_receives_uniform_loss():
    data, model_output = _make_residual_loss_inputs()
    # The middle sampled token has zero teacher residual. Give the student a
    # residual of 0.5 there; with beta=0.1 its Huber loss is 0.45, averaged over
    # all three response tokens uniformly.
    model_output["counterfactual_log_probs"].values()[2] = -0.7
    config = DistillationLossConfig(loss_mode="k1", counterfactual_residual_enabled=True)

    loss, _ = compute_counterfactual_residual_matching_loss(model_output, data, _ActorConfig(), config)

    torch.testing.assert_close(loss, torch.tensor(0.15))


def test_position_split_residual_uses_trailing_tokens_and_preserves_scale():
    data, model_output = _make_residual_loss_inputs()
    config = DistillationLossConfig(
        loss_mode="k1",
        papo_enabled=True,
        counterfactual_residual_enabled=True,
        papo_residual_position_split_enabled=True,
        papo_residual_papo_fraction=0.5,
    )

    # The first response token belongs to the leading PAPO region, so a
    # residual mismatch there must not contribute to residual matching.
    model_output["counterfactual_log_probs"].values()[1] = -0.5
    loss, _ = compute_counterfactual_residual_matching_loss(
        model_output, data, _ActorConfig(), config
    )
    torch.testing.assert_close(loss, torch.tensor(0.0))

    # The third response token is the sole trailing residual token. Its Huber
    # mismatch is 0.45; T/back_count renormalization preserves that full mean.
    model_output["counterfactual_log_probs"].values()[3] = -0.8
    loss, metrics = compute_counterfactual_residual_matching_loss(
        model_output, data, _ActorConfig(), config
    )
    torch.testing.assert_close(loss, torch.tensor(0.45))
    assert metrics[
        "distillation/counterfactual_residual_position_keep_fraction"
    ].aggregate() == pytest.approx(1 / 3)
    assert metrics[
        "distillation/counterfactual_residual_direction_keep_fraction"
    ].aggregate() == pytest.approx(1.0)


def test_direction_aware_residual_ignores_non_positive_teacher_directions():
    data, model_output = _make_residual_loss_inputs()
    # Perturb both the zero-residual token and the negative-residual token.
    # Only the unchanged positive teacher direction should remain selected.
    model_output["counterfactual_log_probs"].values()[2] = -0.7
    model_output["counterfactual_log_probs"].values()[3] = -0.8
    config = DistillationLossConfig(
        loss_mode="k1",
        counterfactual_residual_enabled=True,
        counterfactual_residual_direction_aware=True,
    )

    loss, metrics = compute_counterfactual_residual_matching_loss(
        model_output, data, _ActorConfig(), config
    )

    torch.testing.assert_close(loss, torch.tensor(0.0))
    assert metrics[
        "distillation/counterfactual_residual_direction_keep_fraction"
    ].aggregate() == pytest.approx(1 / 3)


def test_direction_aware_residual_keeps_positive_teacher_direction():
    data, model_output = _make_residual_loss_inputs()
    # The first sampled token has teacher residual 1.0. Change the student's
    # residual to 0.5: smooth-L1(beta=0.1) is 0.45, normalized over the three
    # original response tokens for a direction-aware loss of 0.15.
    model_output["counterfactual_log_probs"].values()[1] = -0.5
    config = DistillationLossConfig(
        loss_mode="k1",
        counterfactual_residual_enabled=True,
        counterfactual_residual_direction_aware=True,
    )

    loss, _ = compute_counterfactual_residual_matching_loss(model_output, data, _ActorConfig(), config)

    torch.testing.assert_close(loss, torch.tensor(0.15))


class _FakeResidualTeacherManager:
    pad_token_id = 0

    def __init__(self):
        self.distillation_loss_config = DistillationLossConfig(
            loss_mode="k1", counterfactual_residual_enabled=True
        )
        self.calls = []

    async def compute_teacher_logprobs_single(self, sequence_ids, multi_modal_data=None, expected_len=None):
        del expected_len
        self.calls.append(multi_modal_data)
        teacher_ids = torch.tensor(sequence_ids, dtype=torch.int32).unsqueeze(-1)
        pixel = multi_modal_data["images"][0].getpixel((0, 0))[0]
        return teacher_ids, torch.full_like(teacher_ids, -pixel / 255, dtype=torch.float32)


def test_batched_teacher_scores_provided_randomly_masked_image():
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
    positive = {"images": [Image.new("RGB", (3, 2), color=(20, 0, 0))]}
    masked = {"images": [Image.new("RGB", (3, 2), color=(0, 0, 0))]}
    data = DataProto(
        batch=batch,
        non_tensor_batch={
            "teacher_multi_modal_data": np.array([positive], dtype=object),
            "teacher_counterfactual_multi_modal_data": np.array([masked], dtype=object),
        },
    )
    manager = _FakeResidualTeacherManager()

    output = asyncio.run(AsyncTeacherLLMServerManager.compute_teacher_logprobs_batch(manager, data))

    assert len(manager.calls) == 2
    assert manager.calls[1]["images"][0].getpixel((0, 0)) == (0, 0, 0)
    assert "teacher_counterfactual_logprobs" in output.batch
    assert "teacher_negative_logprobs" not in output.batch
