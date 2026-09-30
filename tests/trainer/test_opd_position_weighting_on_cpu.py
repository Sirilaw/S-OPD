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

import pytest
import torch

from verl.trainer.distillation.losses import apply_opd_position_weighting
from verl.workers.config import DistillationLossConfig


def test_disabled_position_weighting_is_exact_noop():
    losses = torch.tensor([[1.0, 2.0]])
    config = DistillationLossConfig(loss_mode="k1")

    weighted, metrics = apply_opd_position_weighting(losses, torch.ones_like(losses), config)

    assert weighted is losses
    assert metrics == {}


def test_position_weights_decrease_and_preserve_each_response_scale():
    losses = torch.ones(3, 5)
    response_mask = torch.tensor(
        [
            [1, 1, 1, 1, 1],
            [1, 1, 1, 0, 0],
            [1, 0, 0, 0, 0],
        ],
        dtype=torch.bool,
    )
    config = DistillationLossConfig(
        loss_mode="k1",
        opd_position_weighting_enabled=True,
        opd_position_weight_alpha=1.0,
        opd_position_weight_decay=0.25,
        opd_position_weight_min=0.5,
        opd_position_weight_max=1.5,
    )

    weighted, metrics = apply_opd_position_weighting(losses, response_mask, config)

    for row, count in enumerate([5, 3, 1]):
        valid = weighted[row, :count]
        torch.testing.assert_close(valid.mean(), torch.tensor(1.0))
        assert torch.all(valid[:-1] >= valid[1:])
        assert valid.min() >= 0.5
        assert valid.max() <= 1.5
        assert torch.count_nonzero(weighted[row, count:]) == 0
    assert weighted[0, 0] > weighted[0, -1]
    assert weighted[2, 0] == pytest.approx(1.0)
    assert metrics["distillation/opd_position_weight_mean"].aggregate() == pytest.approx(1.0)
    assert (
        metrics["distillation/opd_position_weight_front_quarter"].aggregate()
        > metrics["distillation/opd_position_weight_back_quarter"].aggregate()
    )


def test_position_weighting_multiplies_signed_opd_losses():
    losses = torch.tensor([[2.0, -3.0, 4.0]])
    response_mask = torch.ones_like(losses, dtype=torch.bool)
    config = DistillationLossConfig(loss_mode="k1", opd_position_weighting_enabled=True)

    weighted, _ = apply_opd_position_weighting(losses, response_mask, config)
    recovered_weights = weighted / losses

    assert recovered_weights[0, 0] > recovered_weights[0, 1] > recovered_weights[0, 2]
    torch.testing.assert_close(recovered_weights.mean(), torch.tensor(1.0))


def test_position_weighting_is_finite_with_left_padding_and_tiny_decay():
    losses = torch.ones(1, 5)
    response_mask = torch.tensor([[0, 0, 1, 1, 1]], dtype=torch.bool)
    config = DistillationLossConfig(
        loss_mode="k1",
        opd_position_weighting_enabled=True,
        opd_position_weight_decay=1e-6,
    )

    weighted, _ = apply_opd_position_weighting(losses, response_mask, config)

    assert torch.isfinite(weighted).all()
    assert torch.count_nonzero(weighted[:, :2]) == 0
    assert weighted[0, 2] > weighted[0, 3]
    assert weighted[0, 3] == pytest.approx(weighted[0, 4])
    assert weighted[0, 2:].mean() == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("opd_position_weight_alpha", -0.1, "alpha"),
        ("opd_position_weight_decay", 0.0, "decay"),
        ("opd_position_weight_min", 0.0, "min"),
        ("opd_position_weight_min", 1.1, "at most 1"),
        ("opd_position_weight_max", 0.9, "at least 1"),
    ],
)
def test_position_weighting_rejects_invalid_config(field, value, message):
    with pytest.raises(ValueError, match=message):
        DistillationLossConfig(loss_mode="k1", **{field: value})
