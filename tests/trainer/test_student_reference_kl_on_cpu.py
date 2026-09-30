from types import SimpleNamespace

import pytest
import torch

import verl.trainer.distillation.losses as distillation_losses
import verl.workers.utils.losses as worker_losses


def _actor_config(*, use_kl_loss: bool = True):
    return SimpleNamespace(
        global_batch_info={},
        policy_loss={"loss_mode": "vanilla"},
        loss_agg_mode="token-mean",
        loss_scale_factor=None,
        entropy_coeff=0.0,
        use_kl_loss=use_kl_loss,
        kl_loss_coef=0.5,
        kl_loss_type="low_var_kl",
    )


def test_reference_kl_survives_when_task_policy_loss_is_disabled(monkeypatch):
    current_log_probs = torch.tensor([[-0.2, -1.0]], requires_grad=True)
    ref_log_probs = torch.tensor([[-0.5, -0.7]])
    response_mask = torch.ones_like(current_log_probs, dtype=torch.bool)
    data = {
        "dp_size": 1,
        "batch_num_tokens": 2,
        "global_batch_size": 1,
        "response_mask": response_mask,
        "old_log_probs": current_log_probs.detach(),
        "advantages": torch.ones_like(current_log_probs),
        "ref_log_prob": ref_log_probs,
    }

    monkeypatch.setattr(worker_losses, "no_padding_2_padding", lambda tensor, _: tensor)
    monkeypatch.setattr(
        worker_losses,
        "get_policy_loss_fn",
        lambda _: lambda **kwargs: (kwargs["log_prob"].new_tensor(7.0), {}),
    )

    loss, metrics = worker_losses.ppo_loss(
        _actor_config(),
        {"log_probs": current_log_probs},
        data,
        include_task_loss=False,
    )

    log_ratio = ref_log_probs - current_log_probs
    expected_kl = (log_ratio.exp() - log_ratio - 1.0).mean()
    assert loss.detach().item() == pytest.approx((0.5 * expected_kl).item())
    assert metrics["kl_loss"].aggregate() == pytest.approx(expected_kl.item())
    loss.backward()
    assert current_log_probs.grad is not None
    assert torch.count_nonzero(current_log_probs.grad) == current_log_probs.numel()


def test_pure_opd_requests_kl_only_ppo_path(monkeypatch):
    observed = {}

    def fake_distillation_loss(config, distillation_config, model_output, data):
        return torch.tensor(2.0), {}

    def fake_ppo_loss(config, model_output, data, dp_group, *, include_task_loss):
        observed["include_task_loss"] = include_task_loss
        return torch.tensor(1.5), {"kl_loss": "kept"}

    monkeypatch.setattr(distillation_losses, "distillation_loss", fake_distillation_loss)
    monkeypatch.setattr(distillation_losses, "ppo_loss", fake_ppo_loss)

    loss_config = SimpleNamespace(
        use_task_rewards=False,
        distillation_loss_coef=1.0,
        format_reward_enabled=False,
        counterfactual_residual_enabled=False,
        papo_enabled=False,
    )
    loss, metrics = distillation_losses.distillation_ppo_loss(
        config=SimpleNamespace(),
        distillation_config=SimpleNamespace(distillation_loss=loss_config),
        model_output={},
        data={},
    )

    assert observed["include_task_loss"] is False
    assert loss.item() == pytest.approx(3.5)
    assert metrics["kl_loss"] == "kept"
