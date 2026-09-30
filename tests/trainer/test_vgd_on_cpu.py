import asyncio
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image
from tensordict import TensorDict

from verl import DataProto
from verl.experimental.teacher_loop.teacher_manager import AsyncTeacherLLMServerManager
from verl.trainer.distillation.losses import compute_distillation_loss_reverse_kl_estimator
from verl.workers.config import DistillationLossConfig
from verl.workers.utils.padding import left_right_2_no_padding


def _make_vgd_inputs():
    input_ids = torch.arange(5).unsqueeze(0)
    data = TensorDict(
        {
            "input_ids": input_ids,
            "prompts": input_ids[:, :2].clone(),
            "responses": input_ids[:, 2:].clone(),
            "attention_mask": torch.ones(1, 5, dtype=torch.long),
            "response_mask": torch.ones(1, 3, dtype=torch.long),
            "position_ids": input_ids.clone(),
            "teacher_ids": input_ids.unsqueeze(-1).clone(),
            "teacher_logprobs": torch.tensor([[[0.0], [-0.2], [-0.5], [-0.4], [0.0]]]),
            "teacher_negative_logprobs": torch.tensor(
                [[[0.0], [-1.2], [-0.1], [-2.4], [0.0]]]
            ),
        },
        batch_size=[1],
    )
    data = left_right_2_no_padding(data)
    data["prompts"] = input_ids[:, :2].clone()
    data["responses"] = input_ids[:, 2:].clone()
    student = torch.tensor([0.0, -0.3, -0.6, -0.8, 0.0])
    return data, student


def test_vgd_shifts_k1_teacher_target_by_clipped_visual_gap():
    data, student = _make_vgd_inputs()
    loss_config = DistillationLossConfig(
        loss_mode="k1",
        use_policy_gradient=True,
        vgd_enabled=True,
        vgd_alpha=2.0,
        vgd_gap_clip=0.5,
    )

    losses, metrics = compute_distillation_loss_reverse_kl_estimator(
        config=SimpleNamespace(),
        distillation_config=SimpleNamespace(distillation_loss=loss_config),
        model_output={"log_probs": student},
        data=data,
    )

    # Teacher gaps [1.0, -0.4, 2.0] are clipped to [0.5, -0.4, 0.5].
    # The shifted targets are therefore [0.8, -1.3, 0.6], and k1 is
    # student_logprob - shifted_teacher_target.
    torch.testing.assert_close(losses, torch.tensor([[-1.1, 0.7, -1.4]]))
    assert metrics["distillation/vgd_visual_gap"].aggregate() == pytest.approx(2.6 / 3)
    assert metrics["distillation/vgd_positive_gap_fraction"].aggregate() == pytest.approx(2 / 3)
    assert metrics["distillation/vgd_clipped_fraction"].aggregate() == pytest.approx(2 / 3)
    assert metrics["distillation/vgd_target_shift"].aggregate() == pytest.approx(0.4)


def test_positive_only_vgd_does_not_lower_teacher_target_for_negative_gaps():
    data, student = _make_vgd_inputs()
    loss_config = DistillationLossConfig(
        loss_mode="k1",
        use_policy_gradient=True,
        vgd_enabled=True,
        vgd_alpha=2.0,
        vgd_gap_clip=0.5,
        vgd_positive_only=True,
    )

    losses, metrics = compute_distillation_loss_reverse_kl_estimator(
        config=SimpleNamespace(),
        distillation_config=SimpleNamespace(distillation_loss=loss_config),
        model_output={"log_probs": student},
        data=data,
    )

    # Teacher gaps [1.0, -0.4, 2.0] become [0.5, 0.0, 0.5]. The negative
    # gap leaves the clean teacher target unchanged instead of lowering it.
    torch.testing.assert_close(losses, torch.tensor([[-1.1, -0.1, -1.4]]))
    assert metrics["distillation/vgd_clipped_fraction"].aggregate() == pytest.approx(2 / 3)
    assert metrics["distillation/vgd_target_shift"].aggregate() == pytest.approx(2 / 3)


def test_vgd_requires_negative_teacher_scores():
    data, student = _make_vgd_inputs()
    del data["teacher_negative_logprobs"]
    loss_config = DistillationLossConfig(loss_mode="k1", vgd_enabled=True)

    with pytest.raises(KeyError, match="teacher_negative_logprobs"):
        compute_distillation_loss_reverse_kl_estimator(
            config=SimpleNamespace(),
            distillation_config=SimpleNamespace(distillation_loss=loss_config),
            model_output={"log_probs": student},
            data=data,
        )


def test_vgd_allows_papo_combination():
    config = DistillationLossConfig(
        loss_mode="k1",
        use_policy_gradient=True,
        vgd_enabled=True,
        papo_enabled=True,
    )

    assert config.vgd_enabled is True
    assert config.papo_enabled is True


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"loss_mode": "k3"}, "loss_mode='k1'"),
        ({"use_policy_gradient": False}, "Directly backpropagating k1"),
        ({"vgd_alpha": -1.0}, "vgd_alpha"),
        ({"vgd_gap_clip": 0.0}, "vgd_gap_clip"),
        ({"vgd_negative_strategy": "unknown"}, "vgd_negative_strategy"),
        ({"vgd_patch_shuffle_block_size": 0}, "vgd_patch_shuffle_block_size"),
        ({"vgd_random_mask_ratio": 1.1}, "vgd_random_mask_ratio"),
        ({"vgd_random_mask_patch_size": 0}, "vgd_random_mask_patch_size"),
    ],
)
def test_vgd_config_validation(kwargs, message):
    config_kwargs = {"loss_mode": "k1", "use_policy_gradient": True, "vgd_enabled": True}
    config_kwargs.update(kwargs)
    with pytest.raises(ValueError, match=message):
        DistillationLossConfig(**config_kwargs)


class _FakeVGDTeacherManager:
    pad_token_id = 0

    def __init__(self):
        self.distillation_loss_config = DistillationLossConfig(loss_mode="k1", vgd_enabled=True)
        self.calls = []

    async def compute_teacher_logprobs_single(
        self, sequence_ids, multi_modal_data=None, expected_len=None
    ):
        del expected_len
        self.calls.append(multi_modal_data)
        teacher_ids = torch.tensor(sequence_ids, dtype=torch.int32).unsqueeze(-1)
        pixel = multi_modal_data["images"][0].getpixel((0, 0))[0]
        return teacher_ids, torch.full_like(teacher_ids, -pixel / 255, dtype=torch.float32)


def test_vgd_batched_teacher_scores_the_supplied_negative_image():
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
                [{"images": [Image.new("RGB", (3, 2), color=(80, 0, 0))]}], dtype=object
            ),
        },
    )
    manager = _FakeVGDTeacherManager()

    output = asyncio.run(AsyncTeacherLLMServerManager.compute_teacher_logprobs_batch(manager, data))

    assert len(manager.calls) == 2
    assert manager.calls[1]["images"][0].getpixel((0, 0)) == (80, 0, 0)
    assert "teacher_negative_logprobs" in output.batch
    assert "teacher_counterfactual_logprobs" not in output.batch


def test_vgd_and_papo_share_one_batched_negative_teacher_score():
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
                [{"images": [Image.new("RGB", (3, 2), color=(80, 0, 0))]}], dtype=object
            ),
        },
    )
    manager = _FakeVGDTeacherManager()
    manager.distillation_loss_config = DistillationLossConfig(
        loss_mode="k1",
        vgd_enabled=True,
        papo_enabled=True,
    )

    output = asyncio.run(AsyncTeacherLLMServerManager.compute_teacher_logprobs_batch(manager, data))

    # One positive call plus one shared negative call. The shared tensor uses the
    # canonical VGD field; PAPO consumes this field when both objectives are on.
    assert len(manager.calls) == 2
    assert "teacher_negative_logprobs" in output.batch
    assert "teacher_counterfactual_logprobs" not in output.batch
