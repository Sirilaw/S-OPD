import asyncio
import math

import numpy as np
import pytest
import torch
from PIL import Image
from tensordict import TensorDict

from verl import DataProto
from verl.experimental.teacher_loop.teacher_manager import (
    AsyncTeacherLLMServerManager,
    build_patch_shuffle_counterfactual_multi_modal_data,
)
from verl.trainer.distillation.losses import compute_papo_perception_kl
from verl.utils import tensordict_utils as tu
from verl.workers.config import DistillationLossConfig
from verl.workers.utils.padding import left_right_2_no_padding


class _ActorConfig:
    loss_agg_mode = "token-mean"
    global_batch_info = {}


def _make_papo_inputs():
    input_ids = torch.arange(5).unsqueeze(0)
    teacher_original = torch.tensor([[0.0, -0.1, -0.5, -0.2, 0.0]])
    teacher_masked = torch.tensor([[0.0, -0.6, -0.2, -0.7, 0.0]])
    data = TensorDict(
        {
            "input_ids": input_ids,
            "prompts": input_ids[:, :2].clone(),
            "responses": input_ids[:, 2:].clone(),
            "attention_mask": torch.ones(1, 5, dtype=torch.long),
            "response_mask": torch.ones(1, 3, dtype=torch.long),
            "position_ids": input_ids.clone(),
            "teacher_ids": input_ids.clone(),
            "teacher_logprobs": teacher_original,
            "teacher_counterfactual_logprobs": teacher_masked,
        },
        batch_size=[1],
    )
    data = left_right_2_no_padding(data)
    data["prompts"] = input_ids[:, :2].clone()
    data["responses"] = input_ids[:, 2:].clone()

    # no_padding_2_padding selects positions [1:4] for the three response
    # tokens, producing masked-original log-ratios [-1.0, -0.4, -0.5].
    # Teacher original-masked gaps are [0.5, -0.3, 0.5], so teacher gating
    # drops the middle token and scales each retained loss by 0.5.
    original = torch.tensor([0.0, -0.2, -0.4, -0.6, 0.0], requires_grad=True)
    masked = torch.tensor([0.0, -1.2, -0.8, -1.1, 0.0], requires_grad=True)
    return data, original, masked


def test_papo_uses_token_level_low_variance_kl_estimator():
    data, original, masked = _make_papo_inputs()
    config = DistillationLossConfig(loss_mode="k1", papo_enabled=True)

    perception_kl, metrics = compute_papo_perception_kl(
        {"log_probs": original, "counterfactual_log_probs": masked},
        data,
        _ActorConfig(),
        config,
    )

    expected = 0.5 * (math.exp(-1.0) + (math.exp(-0.5) - 0.5)) / 2.0
    torch.testing.assert_close(perception_kl, torch.tensor(expected))
    assert metrics["distillation/papo_perception_kl"].aggregate() == pytest.approx(expected)
    assert metrics["distillation/papo_teacher_gate_keep_fraction"].aggregate() == pytest.approx(2 / 3)
    assert metrics["distillation/papo_teacher_visual_scale"].aggregate() == pytest.approx(1 / 3)


def test_jsd_topk_papo_uses_separate_sampled_teacher_gate():
    data, original, masked = _make_papo_inputs()
    data["teacher_sampled_logprobs"] = data["teacher_logprobs"].clone()
    # The main jsd_topk teacher tensor is not the PAPO gate tensor. Making it
    # deliberately unusable verifies that PAPO selects the dedicated field.
    data["teacher_logprobs"] = data["teacher_logprobs"] + 100.0
    config = DistillationLossConfig(
        loss_mode="jsd_topk",
        topk=100,
        use_policy_gradient=False,
        papo_enabled=True,
    )

    perception_kl, metrics = compute_papo_perception_kl(
        {"log_probs": original, "counterfactual_log_probs": masked},
        data,
        _ActorConfig(),
        config,
    )

    expected = 0.5 * (math.exp(-1.0) + (math.exp(-0.5) - 0.5)) / 2.0
    torch.testing.assert_close(perception_kl, torch.tensor(expected))
    assert metrics["distillation/papo_teacher_gate_keep_fraction"].aggregate() == pytest.approx(2 / 3)


def test_papo_without_teacher_gating_uses_all_valid_response_tokens():
    data, original, masked = _make_papo_inputs()
    del data["teacher_logprobs"]
    del data["teacher_counterfactual_logprobs"]
    config = DistillationLossConfig(
        loss_mode="k1",
        papo_enabled=True,
        papo_teacher_gating_enabled=False,
    )

    perception_kl, metrics = compute_papo_perception_kl(
        {"log_probs": original, "counterfactual_log_probs": masked},
        data,
        _ActorConfig(),
        config,
    )

    expected = (
        math.exp(-1.0)
        + (math.exp(-0.4) + 0.4 - 1.0)
        + (math.exp(-0.5) - 0.5)
    ) / 3.0
    torch.testing.assert_close(perception_kl, torch.tensor(expected))
    assert metrics["distillation/papo_teacher_gate_keep_fraction"].aggregate() == 1.0
    assert metrics["distillation/papo_teacher_visual_scale"].aggregate() == 1.0


def test_papo_student_positive_gate_uses_only_binary_student_delta():
    data, original, masked = _make_papo_inputs()
    del data["teacher_logprobs"]
    del data["teacher_counterfactual_logprobs"]
    # Student original-minus-masked deltas at response positions are
    # [1.0, -0.2, 0.0], so only the first token must remain active. In
    # particular, delta == 0 is not selected.
    masked = torch.tensor([0.0, -1.2, -0.2, -0.6, 0.0], requires_grad=True)
    config = DistillationLossConfig(
        loss_mode="k1",
        papo_enabled=True,
        papo_teacher_gate_mode="student_positive",
    )

    perception_kl, metrics = compute_papo_perception_kl(
        {"log_probs": original, "counterfactual_log_probs": masked},
        data,
        _ActorConfig(),
        config,
    )
    (-perception_kl).backward()

    # The student ablation is a unit-weight binary mask over the original
    # candidate-token denominator; it does not scale by either contrast.
    torch.testing.assert_close(perception_kl, torch.tensor(math.exp(-1.0) / 3.0))
    assert original.grad[1].abs() > 0
    assert original.grad[2] == 0
    assert original.grad[3] == 0
    assert masked.grad is None
    assert metrics["distillation/papo_gate_keep_fraction"].aggregate() == pytest.approx(1 / 3)
    assert metrics["distillation/papo_gate_scale"].aggregate() == pytest.approx(1 / 3)
    assert metrics["distillation/papo_teacher_gate_keep_fraction"].aggregate() == pytest.approx(1 / 3)
    assert metrics["distillation/papo_teacher_visual_scale"].aggregate() == pytest.approx(1 / 3)
    assert metrics["distillation/papo_student_gate_enabled"].aggregate() == 1.0


def test_papo_rate_matched_random_gate_preserves_count_and_weights():
    data, original, masked = _make_papo_inputs()
    config = DistillationLossConfig(
        loss_mode="k1",
        papo_enabled=True,
        papo_teacher_gate_mode="rate_matched_random",
    )

    # For this seed, two random destinations are response positions 1 and 2,
    # rather than the teacher-positive positions 0 and 2.
    torch.manual_seed(1)
    perception_kl, metrics = compute_papo_perception_kl(
        {"log_probs": original, "counterfactual_log_probs": masked},
        data,
        _ActorConfig(),
        config,
    )
    (-perception_kl).backward()

    response_grad = original.grad[1:4]
    assert response_grad[0] == 0
    assert response_grad[1].abs() > 0
    assert response_grad[2].abs() > 0
    assert metrics["distillation/papo_teacher_gate_keep_fraction"].aggregate() == pytest.approx(2 / 3)
    assert metrics["distillation/papo_teacher_visual_scale"].aggregate() == pytest.approx(1 / 3)
    assert metrics["distillation/papo_random_gate_enabled"].aggregate() == 1.0


def test_papo_detaches_masked_policy_but_updates_original_policy():
    data, original, masked = _make_papo_inputs()
    config = DistillationLossConfig(loss_mode="k1", papo_enabled=True)

    perception_kl, _ = compute_papo_perception_kl(
        {"log_probs": original, "counterfactual_log_probs": masked},
        data,
        _ActorConfig(),
        config,
    )
    (-perception_kl).backward()

    assert original.grad is not None
    assert original.grad.abs().sum() > 0
    assert masked.grad is None


def test_papo_all_inactive_teacher_gate_returns_zero():
    data, original, masked = _make_papo_inputs()
    data["teacher_counterfactual_logprobs"] = data["teacher_logprobs"].clone()
    config = DistillationLossConfig(loss_mode="k1", papo_enabled=True)

    perception_kl, metrics = compute_papo_perception_kl(
        {"log_probs": original, "counterfactual_log_probs": masked},
        data,
        _ActorConfig(),
        config,
    )

    torch.testing.assert_close(perception_kl, torch.tensor(0.0))
    assert metrics["distillation/papo_teacher_gate_keep_fraction"].aggregate() == 0.0


def test_papo_unwraps_global_active_count_metadata():
    data, original, masked = _make_papo_inputs()
    # FSDP computes this count before splitting the batch into microbatches and
    # stores it as NonTensorData. The loss must unwrap that metadata instead of
    # trying to cast the wrapper itself to int.
    tu.assign_non_tensor(data, papo_active_token_count=4, dp_size=2)

    class _TwoRankActorConfig:
        loss_agg_mode = "token-mean"
        # Exercise the real path where FSDP attaches dp_size to the batch as
        # NonTensorData, without relying on mutable config state.
        global_batch_info = {}

    perception_kl, _ = compute_papo_perception_kl(
        {"log_probs": original, "counterfactual_log_probs": masked},
        data,
        _TwoRankActorConfig(),
        DistillationLossConfig(loss_mode="k1", papo_enabled=True),
    )

    local_weighted_sum = 0.5 * (math.exp(-1.0) + (math.exp(-0.5) - 0.5))
    expected = local_weighted_sum / 4.0 * 2
    torch.testing.assert_close(perception_kl, torch.tensor(expected))


def test_papo_unwraps_zero_global_active_count_metadata():
    data, original, masked = _make_papo_inputs()
    tu.assign_non_tensor(data, papo_active_token_count=0)

    perception_kl, _ = compute_papo_perception_kl(
        {"log_probs": original, "counterfactual_log_probs": masked},
        data,
        _ActorConfig(),
        DistillationLossConfig(loss_mode="k1", papo_enabled=True),
    )

    torch.testing.assert_close(perception_kl, torch.tensor(0.0))


def test_papo_position_split_uses_leading_tokens_and_preserves_scale():
    data, original, masked = _make_papo_inputs()
    config = DistillationLossConfig(
        loss_mode="k1",
        papo_enabled=True,
        counterfactual_residual_enabled=True,
        papo_residual_position_split_enabled=True,
        papo_residual_papo_fraction=0.5,
    )

    perception_kl, metrics = compute_papo_perception_kl(
        {"log_probs": original, "counterfactual_log_probs": masked},
        data,
        _ActorConfig(),
        config,
    )

    # The leading region contains two tokens, but only its first token passes
    # the teacher gate, so that one token is the PAPO denominator.
    expected = 0.5 * (math.exp(-1.0) - 1.0 + 1.0)
    torch.testing.assert_close(perception_kl, torch.tensor(expected))
    assert metrics["distillation/papo_position_keep_fraction"].aggregate() == pytest.approx(2 / 3)


def test_papo_front_only_uses_leading_tokens_without_residual():
    data, original, masked = _make_papo_inputs()
    config = DistillationLossConfig(
        loss_mode="k1",
        papo_enabled=True,
        papo_front_only_enabled=True,
        papo_front_fraction=0.5,
    )

    perception_kl, metrics = compute_papo_perception_kl(
        {"log_probs": original, "counterfactual_log_probs": masked},
        data,
        _ActorConfig(),
        config,
    )

    expected = 0.5 * (math.exp(-1.0) - 1.0 + 1.0)
    torch.testing.assert_close(perception_kl, torch.tensor(expected))
    assert metrics["distillation/papo_position_keep_fraction"].aggregate() == pytest.approx(2 / 3)


def test_papo_is_disabled_by_default_and_allows_residual_combination():
    default_config = DistillationLossConfig(loss_mode="k1")
    assert default_config.papo_enabled is False
    assert default_config.papo_teacher_gating_enabled is True
    assert default_config.papo_teacher_gate_mode == "teacher_positive"
    config = DistillationLossConfig(
        loss_mode="k1",
        papo_enabled=True,
        counterfactual_residual_enabled=True,
    )
    assert config.papo_enabled is True
    assert config.counterfactual_residual_enabled is True


def test_papo_with_vgd_uses_shared_teacher_negative_scores():
    data, original, masked = _make_papo_inputs()
    data["teacher_negative_logprobs"] = data.pop("teacher_counterfactual_logprobs")
    config = DistillationLossConfig(
        loss_mode="k1",
        vgd_enabled=True,
        papo_enabled=True,
    )

    perception_kl, _ = compute_papo_perception_kl(
        {"log_probs": original, "counterfactual_log_probs": masked},
        data,
        _ActorConfig(),
        config,
    )

    assert torch.isfinite(perception_kl)


def test_position_split_requires_both_objectives_and_valid_fraction():
    with pytest.raises(ValueError, match="requires both"):
        DistillationLossConfig(
            loss_mode="k1",
            papo_enabled=True,
            papo_residual_position_split_enabled=True,
        )
    with pytest.raises(ValueError, match="papo_residual_papo_fraction"):
        DistillationLossConfig(
            loss_mode="k1",
            papo_enabled=True,
            counterfactual_residual_enabled=True,
            papo_residual_position_split_enabled=True,
            papo_residual_papo_fraction=1.0,
        )


def test_papo_front_only_requires_papo_and_valid_fraction():
    with pytest.raises(ValueError, match="requires papo_enabled"):
        DistillationLossConfig(loss_mode="k1", papo_front_only_enabled=True)
    with pytest.raises(ValueError, match="papo_front_fraction"):
        DistillationLossConfig(
            loss_mode="k1",
            papo_enabled=True,
            papo_front_only_enabled=True,
            papo_front_fraction=1.0,
        )
    with pytest.raises(ValueError, match="cannot be combined"):
        DistillationLossConfig(
            loss_mode="k1",
            papo_enabled=True,
            counterfactual_residual_enabled=True,
            papo_residual_position_split_enabled=True,
            papo_front_only_enabled=True,
        )


@pytest.mark.parametrize("incompatible_mode", ["visual_grounding_enabled"])
def test_papo_still_rejects_incompatible_visual_modes(incompatible_mode):
    with pytest.raises(ValueError, match="cannot be combined"):
        DistillationLossConfig(loss_mode="k1", papo_enabled=True, **{incompatible_mode: True})


def test_papo_can_combine_with_token_va_and_uses_its_dedicated_teacher_view():
    data, original, masked = _make_papo_inputs()
    # VA-OPD's pixelated scores deliberately imply an all-inactive gate. PAPO
    # must ignore them and retain its separately masked teacher scores.
    data["teacher_negative_logprobs"] = data["teacher_logprobs"].clone()
    config = DistillationLossConfig(
        loss_mode="k1",
        papo_enabled=True,
        token_level_va_enabled=True,
    )

    perception_kl, metrics = compute_papo_perception_kl(
        {"log_probs": original, "counterfactual_log_probs": masked},
        data,
        _ActorConfig(),
        config,
    )

    assert perception_kl > 0
    assert metrics["distillation/papo_teacher_gate_keep_fraction"].aggregate() == pytest.approx(2 / 3)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("counterfactual_residual_mask_ratio", 0.5, "mask_ratio must equal"),
        ("counterfactual_residual_patch_size", 16, "patch_size must equal"),
    ],
)
def test_combined_papo_and_residual_require_the_same_mask(field, value, message):
    with pytest.raises(ValueError, match=message):
        DistillationLossConfig(
            loss_mode="k1",
            papo_enabled=True,
            counterfactual_residual_enabled=True,
            **{field: value},
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("papo_mask_ratio", 1.1, "papo_mask_ratio"),
        ("papo_patch_size", 0, "papo_patch_size"),
        ("papo_coef", -0.1, "papo_coef"),
        ("papo_log_ratio_clip", 0.0, "papo_log_ratio_clip"),
        ("papo_kl_max", 0.0, "papo_kl_max"),
    ],
)
def test_papo_config_rejects_invalid_values(field, value, message):
    with pytest.raises(ValueError, match=message):
        DistillationLossConfig(loss_mode="k1", papo_enabled=True, **{field: value})


def test_papo_config_rejects_unknown_teacher_gate_mode():
    with pytest.raises(ValueError, match="papo_teacher_gate_mode"):
        DistillationLossConfig(
            loss_mode="k1",
            papo_enabled=True,
            papo_teacher_gate_mode="unknown",
        )


def test_papo_config_accepts_student_positive_gate_mode():
    config = DistillationLossConfig(
        loss_mode="k1",
        papo_enabled=True,
        papo_teacher_gate_mode="student_positive",
    )
    assert config.papo_teacher_gate_mode == "student_positive"


def test_papo_patch_shuffle_keeps_partial_boundaries_and_permutes_full_blocks():
    image = Image.new("RGB", (5, 5), color=(99, 99, 99))
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
    for index, color in enumerate(colors):
        x = (index % 2) * 2
        y = (index // 2) * 2
        for px in range(x, x + 2):
            for py in range(y, y + 2):
                image.putpixel((px, py), color)

    output = build_patch_shuffle_counterfactual_multi_modal_data(
        {"images": [image]},
        block_size=2,
        generator=torch.Generator().manual_seed(0),
    )
    shuffled = output["images"][0]

    shuffled_colors = [shuffled.getpixel((x, y)) for y in (0, 2) for x in (0, 2)]
    assert sorted(shuffled_colors) == sorted(colors)
    assert shuffled_colors != colors
    assert [shuffled.getpixel((4, y)) for y in range(5)] == [(99, 99, 99)] * 5
    assert [shuffled.getpixel((x, 4)) for x in range(5)] == [(99, 99, 99)] * 5


def test_papo_patch_shuffle_falls_back_to_noise_for_tiny_images():
    image = Image.new("RGB", (3, 3), color=(0, 0, 0))
    output = build_patch_shuffle_counterfactual_multi_modal_data(
        {"images": [image]},
        block_size=4,
        generator=torch.Generator().manual_seed(0),
    )

    assert output["images"][0].size == image.size
    assert output["images"][0].tobytes() != image.tobytes()


def test_papo_patch_shuffle_config_validation():
    config = DistillationLossConfig(
        loss_mode="k1",
        papo_enabled=True,
        papo_negative_strategy="patch_shuffle",
    )
    assert config.papo_patch_shuffle_block_size is None

    with pytest.raises(ValueError, match="random_mask.*patch_shuffle"):
        DistillationLossConfig(
            loss_mode="k1",
            papo_enabled=True,
            papo_negative_strategy="unknown",
        )
    with pytest.raises(ValueError, match="papo_patch_shuffle_block_size"):
        DistillationLossConfig(
            loss_mode="k1",
            papo_enabled=True,
            papo_negative_strategy="patch_shuffle",
            papo_patch_shuffle_block_size=0,
        )
    with pytest.raises(ValueError, match="Combined PAPO"):
        DistillationLossConfig(
            loss_mode="k1",
            papo_enabled=True,
            papo_negative_strategy="patch_shuffle",
            counterfactual_residual_enabled=True,
        )


class _FakePapoTeacherManager:
    def __init__(self):
        self.distillation_loss_config = DistillationLossConfig(loss_mode="k1", papo_enabled=True)
        self.calls = []
        self.pad_token_id = 0

    async def compute_teacher_logprobs_single(self, sequence_ids, multi_modal_data=None, expected_len=None):
        del expected_len
        self.calls.append(multi_modal_data)
        teacher_ids = torch.tensor(sequence_ids, dtype=torch.int32).unsqueeze(-1)
        pixel = multi_modal_data["images"][0].getpixel((0, 0))[0]
        return teacher_ids, torch.full_like(teacher_ids, -pixel / 255, dtype=torch.float32)


def test_papo_batched_teacher_scores_the_same_masked_image():
    input_ids = torch.arange(4).unsqueeze(0)
    data = DataProto(
        batch=TensorDict(
            {
                "input_ids": input_ids,
                "prompts": input_ids[:, :2],
                "responses": input_ids[:, 2:],
                "attention_mask": torch.ones(1, 4, dtype=torch.long),
            },
            batch_size=[1],
        ),
        non_tensor_batch={
            "teacher_multi_modal_data": np.array(
                [{"images": [Image.new("RGB", (3, 2), color=(20, 0, 0))]}], dtype=object
            ),
            "teacher_counterfactual_multi_modal_data": np.array(
                [{"images": [Image.new("RGB", (3, 2), color=(0, 0, 0))]}], dtype=object
            ),
        },
    )
    manager = _FakePapoTeacherManager()

    output = asyncio.run(AsyncTeacherLLMServerManager.compute_teacher_logprobs_batch(manager, data))

    assert len(manager.calls) == 2
    assert manager.calls[1]["images"][0].getpixel((0, 0)) == (0, 0, 0)
    assert "teacher_counterfactual_logprobs" in output.batch


def test_papo_student_positive_gate_skips_teacher_counterfactual_scoring():
    input_ids = torch.arange(4).unsqueeze(0)
    data = DataProto(
        batch=TensorDict(
            {
                "input_ids": input_ids,
                "prompts": input_ids[:, :2],
                "responses": input_ids[:, 2:],
                "attention_mask": torch.ones(1, 4, dtype=torch.long),
            },
            batch_size=[1],
        ),
        non_tensor_batch={
            "teacher_multi_modal_data": np.array(
                [{"images": [Image.new("RGB", (3, 2), color=(20, 0, 0))]}],
                dtype=object,
            ),
            "teacher_counterfactual_multi_modal_data": np.array(
                [{"images": [Image.new("RGB", (3, 2), color=(0, 0, 0))]}],
                dtype=object,
            ),
        },
    )
    manager = _FakePapoTeacherManager()
    manager.distillation_loss_config = DistillationLossConfig(
        loss_mode="k1",
        papo_enabled=True,
        papo_teacher_gate_mode="student_positive",
    )

    output = asyncio.run(
        AsyncTeacherLLMServerManager.compute_teacher_logprobs_batch(manager, data)
    )

    assert len(manager.calls) == 1
    assert "teacher_counterfactual_logprobs" not in output.batch


class _FakeJsdPapoTeacherManager:
    def __init__(self):
        self.distillation_loss_config = DistillationLossConfig(
            loss_mode="jsd_topk",
            topk=2,
            use_policy_gradient=False,
            papo_enabled=True,
            papo_teacher_gating_enabled=True,
        )
        self.calls = []
        self.pad_token_id = 0

    async def compute_teacher_logprobs_single(
        self,
        sequence_ids,
        multi_modal_data=None,
        expected_len=None,
        num_logprobs_override=None,
    ):
        del expected_len
        self.calls.append((multi_modal_data, num_logprobs_override))
        seq = torch.tensor(sequence_ids, dtype=torch.int32)
        pixel = multi_modal_data["images"][0].getpixel((0, 0))[0]
        if num_logprobs_override == 0:
            ids = seq.unsqueeze(-1)
            return ids, torch.full_like(ids, -pixel / 255, dtype=torch.float32)
        ids = torch.stack((seq, (seq + 1) % 16), dim=-1)
        logprobs = torch.tensor([-0.2, -1.8], dtype=torch.float32).expand_as(ids)
        return ids, logprobs


def test_jsd_topk_papo_keeps_topk_and_sampled_teacher_outputs_separate():
    input_ids = torch.arange(4).unsqueeze(0)
    data = DataProto(
        batch=TensorDict(
            {
                "input_ids": input_ids,
                "prompts": input_ids[:, :2],
                "responses": input_ids[:, 2:],
                "attention_mask": torch.ones(1, 4, dtype=torch.long),
            },
            batch_size=[1],
        ),
        non_tensor_batch={
            "teacher_multi_modal_data": np.array(
                [{"images": [Image.new("RGB", (3, 2), color=(20, 0, 0))]}], dtype=object
            ),
            "teacher_counterfactual_multi_modal_data": np.array(
                [{"images": [Image.new("RGB", (3, 2), color=(0, 0, 0))]}], dtype=object
            ),
        },
    )
    manager = _FakeJsdPapoTeacherManager()

    output = asyncio.run(AsyncTeacherLLMServerManager.compute_teacher_logprobs_batch(manager, data))

    assert len(manager.calls) == 3
    assert output.batch["teacher_logprobs"].shape == (1, 4, 2)
    assert output.batch["teacher_ids"].shape == (1, 4, 2)
    assert output.batch["teacher_sampled_logprobs"].shape == (1, 4, 1)
    assert output.batch["teacher_counterfactual_logprobs"].shape == (1, 4, 1)
