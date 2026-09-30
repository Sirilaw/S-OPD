from types import SimpleNamespace

import pytest
import torch

import verl.workers.utils.losses as worker_losses
from verl.workers.config.actor import ActorConfig


def _actor_config(**overrides):
    values = {
        "global_batch_info": {},
        "policy_loss": {"loss_mode": "vanilla"},
        "loss_agg_mode": "token-mean",
        "loss_scale_factor": None,
        "entropy_coeff": 0.0,
        "use_kl_loss": False,
        "kl_loss_coef": 0.01,
        "kl_loss_type": "low_var_kl",
        "papo_enabled": True,
        "papo_coef": 0.25,
        "papo_use_aug_entropy_loss": False,
        "papo_aug_entropy_loss_coef": 0.03,
        "papo_use_ori_entropy_loss": False,
        "papo_ori_entropy_loss_coef": 0.03,
        "papo_recompute_aug_log_probs": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_papo_grpo_maximizes_low_variance_perception_kl(monkeypatch):
    original_log_probs = torch.tensor([[-0.2, -1.0]], requires_grad=True)
    masked_log_probs = torch.tensor([[-0.5, -0.7]], requires_grad=True)
    response_mask = torch.ones_like(original_log_probs, dtype=torch.bool)
    data = {
        "dp_size": 1,
        "batch_num_tokens": 2,
        "global_batch_size": 1,
        "response_mask": response_mask,
        "old_log_probs": original_log_probs.detach(),
        "advantages": torch.ones_like(original_log_probs),
    }

    monkeypatch.setattr(worker_losses, "no_padding_2_padding", lambda tensor, _: tensor)
    monkeypatch.setattr(
        worker_losses,
        "get_policy_loss_fn",
        lambda _: lambda **kwargs: (kwargs["log_prob"].new_tensor(2.0), {}),
    )

    loss, metrics = worker_losses.ppo_loss(
        _actor_config(),
        {
            "log_probs": original_log_probs,
            "counterfactual_log_probs": masked_log_probs,
        },
        data,
    )

    log_ratio = masked_log_probs.detach() - original_log_probs
    expected_kl = (log_ratio.exp() - log_ratio - 1.0).mean()
    assert loss.detach().item() == pytest.approx((2.0 - 0.25 * expected_kl).item())
    assert metrics["papo_kl"].aggregate() == pytest.approx(expected_kl.detach().item())
    assert metrics["papo_coef"] == 0.25
    assert not any("entropy" in key for key in metrics)

    loss.backward()
    assert original_log_probs.grad is not None
    assert torch.count_nonzero(original_log_probs.grad) == original_log_probs.numel()
    assert masked_log_probs.grad is None


def test_papo_grpo_is_optional_and_validates_mask_settings():
    default_config = ActorConfig(strategy="fsdp2", rollout_n=1, use_dynamic_bsz=True)
    assert default_config.papo_enabled is False
    assert default_config.papo_use_aug_entropy_loss is False
    assert default_config.papo_use_ori_entropy_loss is False
    assert default_config.papo_recompute_aug_log_probs is False

    with pytest.raises(ValueError, match="papo_mask_ratio"):
        ActorConfig(
            strategy="fsdp2",
            rollout_n=1,
            use_dynamic_bsz=True,
            papo_enabled=True,
            papo_mask_ratio=1.1,
        )
    with pytest.raises(ValueError, match="papo_patch_size"):
        ActorConfig(
            strategy="fsdp2",
            rollout_n=1,
            use_dynamic_bsz=True,
            papo_enabled=True,
            papo_patch_size=0,
        )


def test_papo_double_entropy_matches_official_sampled_nll_and_cached_aug_path(monkeypatch):
    original_log_probs = torch.tensor([[-0.2, -1.0]], requires_grad=True)
    masked_log_probs = torch.tensor([[-0.5, -0.7]], requires_grad=True)
    response_mask = torch.ones_like(original_log_probs, dtype=torch.bool)
    data = {
        "dp_size": 1,
        "batch_num_tokens": 2,
        "global_batch_size": 1,
        "response_mask": response_mask,
        "old_log_probs": original_log_probs.detach(),
        "advantages": torch.ones_like(original_log_probs),
    }

    monkeypatch.setattr(worker_losses, "no_padding_2_padding", lambda tensor, _: tensor)
    monkeypatch.setattr(
        worker_losses,
        "get_policy_loss_fn",
        lambda _: lambda **kwargs: (kwargs["log_prob"].new_tensor(2.0), {}),
    )
    config = _actor_config(
        papo_use_aug_entropy_loss=True,
        papo_aug_entropy_loss_coef=0.03,
        papo_use_ori_entropy_loss=True,
        papo_ori_entropy_loss_coef=0.03,
    )

    loss, metrics = worker_losses.ppo_loss(
        config,
        {"log_probs": original_log_probs, "counterfactual_log_probs": masked_log_probs},
        data,
    )

    log_ratio = masked_log_probs.detach() - original_log_probs
    expected_kl = (log_ratio.exp() - log_ratio - 1.0).mean()
    expected_ori_entropy = -original_log_probs.mean()
    expected_aug_entropy = -masked_log_probs.detach().mean()
    expected_loss = 2.0 - 0.25 * expected_kl + 0.03 * (expected_ori_entropy + expected_aug_entropy)
    torch.testing.assert_close(loss, expected_loss)
    assert metrics["papo_ori_entropy_loss"].aggregate() == pytest.approx(expected_ori_entropy.item())
    assert metrics["papo_aug_entropy_loss"].aggregate() == pytest.approx(expected_aug_entropy.item())

    loss.backward()
    assert original_log_probs.grad is not None
    assert masked_log_probs.grad is None


def test_papo_recomputed_aug_entropy_has_masked_branch_gradients(monkeypatch):
    original_log_probs = torch.tensor([[-0.2, -1.0]], requires_grad=True)
    masked_log_probs = torch.tensor([[-0.5, -0.7]], requires_grad=True)
    data = {
        "dp_size": 1,
        "batch_num_tokens": 2,
        "global_batch_size": 1,
        "response_mask": torch.ones_like(original_log_probs, dtype=torch.bool),
        "old_log_probs": original_log_probs.detach(),
        "advantages": torch.ones_like(original_log_probs),
    }
    monkeypatch.setattr(worker_losses, "no_padding_2_padding", lambda tensor, _: tensor)
    monkeypatch.setattr(
        worker_losses,
        "get_policy_loss_fn",
        lambda _: lambda **kwargs: (kwargs["log_prob"].new_tensor(2.0), {}),
    )

    loss, _ = worker_losses.ppo_loss(
        _actor_config(
            papo_use_aug_entropy_loss=True,
            papo_recompute_aug_log_probs=True,
        ),
        {"log_probs": original_log_probs, "counterfactual_log_probs": masked_log_probs},
        data,
    )
    loss.backward()

    assert masked_log_probs.grad is not None
    assert torch.count_nonzero(masked_log_probs.grad) == masked_log_probs.numel()


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"papo_aug_entropy_loss_coef": -0.1}, "papo_aug_entropy_loss_coef"),
        ({"papo_ori_entropy_loss_coef": -0.1}, "papo_ori_entropy_loss_coef"),
        ({"papo_use_aug_entropy_loss": True}, "require papo_enabled"),
        ({"papo_use_ori_entropy_loss": True}, "require papo_enabled"),
        ({"papo_recompute_aug_log_probs": True}, "require papo_enabled"),
    ],
)
def test_papo_entropy_config_validation(overrides, message):
    with pytest.raises(ValueError, match=message):
        ActorConfig(strategy="fsdp2", rollout_n=1, use_dynamic_bsz=True, **overrides)
