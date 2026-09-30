import math

import numpy as np
import pytest
import torch
from PIL import Image
from tensordict import TensorDict

from verl.experimental.teacher_loop.teacher_manager import build_gaussian_near_multi_modal_data
from verl.trainer.distillation.losses import compute_gaussian_near_pull_kl
from verl.workers.config import DistillationLossConfig
from verl.workers.utils.padding import left_right_2_no_padding


class _ActorConfig:
    loss_agg_mode = "token-mean"
    global_batch_info = {}


def _make_inputs():
    input_ids = torch.arange(5).unsqueeze(0)
    data = TensorDict(
        {
            "input_ids": input_ids,
            "prompts": input_ids[:, :2].clone(),
            "responses": input_ids[:, 2:].clone(),
            "attention_mask": torch.ones(1, 5, dtype=torch.long),
            "response_mask": torch.ones(1, 3, dtype=torch.long),
            "position_ids": input_ids.clone(),
        },
        batch_size=[1],
    )
    data = left_right_2_no_padding(data)
    data["prompts"] = input_ids[:, :2].clone()
    data["responses"] = input_ids[:, 2:].clone()

    # no_padding_2_padding selects positions [1:4], giving near-original
    # log-ratios [-0.1, +0.2, -0.4].
    original = torch.tensor([0.0, -0.2, -0.4, -0.6, 0.0], requires_grad=True)
    near = torch.tensor([0.0, -0.3, -0.2, -1.0, 0.0], requires_grad=True)
    return data, original, near


def test_gaussian_near_builder_is_seeded_and_non_mutating():
    pixels = np.full((6, 7, 3), 128, dtype=np.uint8)
    image = Image.fromarray(pixels, mode="RGB")
    source = {"images": [image], "tag": "kept"}

    first = build_gaussian_near_multi_modal_data(
        source, std=0.2, generator=torch.Generator().manual_seed(7)
    )
    second = build_gaussian_near_multi_modal_data(
        source, std=0.2, generator=torch.Generator().manual_seed(7)
    )

    assert first is not source
    assert first["tag"] == "kept"
    np.testing.assert_array_equal(np.asarray(first["images"][0]), np.asarray(second["images"][0]))
    np.testing.assert_array_equal(np.asarray(source["images"][0]), pixels)
    assert not np.array_equal(np.asarray(first["images"][0]), pixels)


def test_gaussian_near_builder_std_zero_is_identity():
    pixels = np.arange(5 * 4 * 3, dtype=np.uint8).reshape(5, 4, 3)
    source = {"images": [Image.fromarray(pixels, mode="RGB")]}

    near = build_gaussian_near_multi_modal_data(source, std=0.0)

    np.testing.assert_array_equal(np.asarray(near["images"][0]), pixels)


def test_gaussian_near_pull_uses_low_variance_kl_and_detaches_near_arm():
    data, original, near = _make_inputs()
    config = DistillationLossConfig(loss_mode="k1", gaussian_near_pull_enabled=True)

    near_kl, metrics = compute_gaussian_near_pull_kl(
        {"log_probs": original, "gaussian_near_log_probs": near},
        data,
        _ActorConfig(),
        config,
    )

    ratios = (-0.1, 0.2, -0.4)
    expected = sum(math.exp(value) - value - 1.0 for value in ratios) / len(ratios)
    torch.testing.assert_close(near_kl, torch.tensor(expected))
    assert metrics["distillation/gaussian_near_pull_kl"].aggregate() == pytest.approx(
        expected, abs=1e-7
    )

    near_kl.backward()
    assert original.grad is not None
    assert original.grad.abs().sum() > 0
    assert near.grad is None


def test_gaussian_near_pull_rejects_distributional_topk_mode():
    with pytest.raises(ValueError, match="sampled-token distillation losses"):
        DistillationLossConfig(
            loss_mode="jsd_topk",
            use_policy_gradient=False,
            gaussian_near_pull_enabled=True,
        )
