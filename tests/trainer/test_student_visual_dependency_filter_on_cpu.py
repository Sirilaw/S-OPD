import math

import pytest
import torch
from tensordict import TensorDict

from verl.trainer.distillation.losses import compute_student_visual_dependency_mask
from verl.workers.config import DistillationLossConfig
from verl.workers.utils.padding import left_right_2_no_padding


def _make_inputs():
    input_ids = torch.arange(7).unsqueeze(0)
    data = TensorDict(
        {
            "input_ids": input_ids,
            "prompts": input_ids[:, :2].clone(),
            "responses": input_ids[:, 2:].clone(),
            "attention_mask": torch.ones(1, 7, dtype=torch.long),
            "response_mask": torch.ones(1, 5, dtype=torch.long),
            "position_ids": input_ids.clone(),
        },
        batch_size=[1],
    )
    data = left_right_2_no_padding(data)
    data["prompts"] = input_ids[:, :2].clone()
    data["responses"] = input_ids[:, 2:].clone()

    clean = torch.zeros(7, requires_grad=True)
    masked = torch.tensor(
        [0.0, 0.1, 2.0, -1.0, 0.5, -3.0, 0.0], requires_grad=True
    )
    return data, clean, masked


def test_filter_keeps_exact_top_fraction_per_response():
    data, clean, masked = _make_inputs()
    config = DistillationLossConfig(
        loss_mode="k1",
        student_visual_dependency_filter_enabled=True,
        student_visual_dependency_top_fraction=0.4,
    )

    mask, metrics = compute_student_visual_dependency_mask(
        {"log_probs": clean, "counterfactual_log_probs": masked}, data, config
    )

    # ceil(5 * 0.4) = 2. The largest low-variance KL scores correspond to
    # d=2.0 and d=-3.0, at response positions 1 and 4.
    torch.testing.assert_close(mask, torch.tensor([[False, True, False, False, True]]))
    assert metrics["distillation/student_visual_dependency_keep_fraction"].aggregate() == pytest.approx(0.4)
    expected_selected_mean = ((math.exp(2.0) - 3.0) + (math.exp(-3.0) + 2.0)) / 2.0
    assert metrics["distillation/student_visual_dependency_selected_mean"].aggregate() == pytest.approx(
        expected_selected_mean
    )
    assert mask.requires_grad is False


def test_filter_is_disabled_by_default():
    data, clean, masked = _make_inputs()
    mask, metrics = compute_student_visual_dependency_mask(
        {"log_probs": clean, "counterfactual_log_probs": masked},
        data,
        DistillationLossConfig(loss_mode="k1"),
    )
    assert mask is None
    assert metrics == {}


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"student_visual_dependency_top_fraction": 0.0}, "top_fraction"),
        ({"student_visual_dependency_top_fraction": 1.1}, "top_fraction"),
        ({"student_visual_dependency_mask_ratio": -0.1}, "mask_ratio"),
        ({"student_visual_dependency_patch_size": 0}, "patch_size"),
        ({"vgd_enabled": True}, "standalone ablation"),
        ({"counterfactual_residual_enabled": True}, "standalone ablation"),
        ({"papo_enabled": True}, "standalone ablation"),
    ],
)
def test_filter_config_validation(kwargs, message):
    config_kwargs = {
        "loss_mode": "k1",
        "use_policy_gradient": True,
        "student_visual_dependency_filter_enabled": True,
    }
    config_kwargs.update(kwargs)
    with pytest.raises(ValueError, match=message):
        DistillationLossConfig(**config_kwargs)
