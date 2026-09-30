from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict

from verl.trainer.distillation.losses import mask_opd_termination_token


def _config(eos_token_id=99):
    return SimpleNamespace(model_config=SimpleNamespace(eos_token_id=eos_token_id))


def _loss_config(enabled: bool):
    return SimpleNamespace(mask_termination_token=enabled)


def _data():
    return TensorDict(
        {
            "responses": torch.tensor([[10, 99, 0], [99, 20, 99]]),
            "response_mask": torch.tensor([[1, 1, 0], [1, 1, 1]], dtype=torch.bool),
        },
        batch_size=[2],
    )


def test_disabled_termination_mask_is_exact_noop():
    losses = torch.arange(6, dtype=torch.float32).reshape(2, 3)

    output, metrics = mask_opd_termination_token(losses, _data(), _config(), _loss_config(False))

    assert output is losses
    assert metrics == {}


def test_masks_only_valid_eos_positions_from_opd_loss():
    losses = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    output, metrics = mask_opd_termination_token(losses, _data(), _config(), _loss_config(True))

    assert torch.equal(output, torch.tensor([[1.0, 0.0, 3.0], [0.0, 5.0, 0.0]]))
    assert torch.equal(_data()["response_mask"], torch.tensor([[1, 1, 0], [1, 1, 1]], dtype=torch.bool))
    assert metrics["distillation/termination_tokens_masked"].aggregate() == pytest.approx(3.0)
    assert metrics["distillation/termination_token_fraction"].aggregate() == pytest.approx(3 / 5)


def test_supports_multiple_model_eos_token_ids():
    losses = torch.ones(2, 3)

    output, _ = mask_opd_termination_token(losses, _data(), _config([20, 99]), _loss_config(True))

    assert torch.equal(output, torch.tensor([[1.0, 0.0, 1.0], [0.0, 0.0, 0.0]]))


def test_enabled_mask_requires_model_eos_token_id():
    losses = torch.ones(2, 3)

    with pytest.raises(ValueError, match="eos_token_id"):
        mask_opd_termination_token(losses, _data(), _config(None), _loss_config(True))
