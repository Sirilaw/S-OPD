import pytest
import torch
from tensordict import TensorDict

from verl import DataProto
from verl.trainer.distillation.format_reward import (
    add_format_reward_advantages,
    score_boxed_format,
    score_think_boxed_format,
)
from verl.trainer.distillation.losses import compute_format_reward_loss
from verl.workers.config import DistillationLossConfig
from verl.workers.utils.padding import left_right_2_no_padding


@pytest.mark.parametrize(
    "response",
    [
        "<think>Use the image.</think> Therefore, \\boxed{2}.",
        "<think>Nested math.</think> \\boxed{\\frac{1}{2}}",
    ],
)
def test_think_boxed_format_accepts_complete_response(response):
    assert score_think_boxed_format(response) == 1.0


@pytest.mark.parametrize(
    "response",
    [
        "<think>I keep thinking and never close the tag",
        "<think>Done.</think> The answer is 2.",
        "<think>Done.</think> \\boxed{2} then I resume reasoning",
        "<think></think> \\boxed{2}",
        "prefix <think>Done.</think> \\boxed{2}",
    ],
)
def test_think_boxed_format_rejects_incomplete_or_resumed_reasoning(response):
    assert score_think_boxed_format(response) == 0.0


def test_boxed_style_matches_current_opd_outputs_without_explicit_think_tags():
    assert score_boxed_format("Reason through the image, then conclude. \\boxed{A}") == 1.0
    assert score_boxed_format("Reason through the image. $$\\boxed{\\text{A}}$$") == 1.0
    assert score_boxed_format("Reason forever without a final answer") == 0.0
    assert score_boxed_format("Answer: \\boxed{A} then resume reasoning") == 0.0


class _FakeTokenizer:
    eos_token_id = 99

    def decode(self, token_ids, skip_special_tokens=True):
        del skip_special_tokens
        return {
            1: "<think>Done.</think> \\boxed{2}",
            2: "<think>Also done.</think> \\boxed{3}",
        }[token_ids[0]]


def _make_batch():
    responses = torch.tensor([[1, 11, 12, 0], [2, 21, 22, 23]])
    response_mask = torch.tensor([[1, 1, 1, 0], [1, 1, 1, 1]])
    attention_mask = torch.tensor([[1, 1, 1, 1, 1, 0], [1, 1, 1, 1, 1, 1]])
    return DataProto(
        batch=TensorDict(
            {
                "responses": responses,
                "response_mask": response_mask,
                "attention_mask": attention_mask,
            },
            batch_size=[2],
        )
    )


def test_format_advantage_penalizes_max_length_truncation_independently():
    batch = _make_batch()
    config = DistillationLossConfig(
        loss_mode="k1",
        format_reward_enabled=True,
        format_reward_baseline=0.5,
    )

    metrics = add_format_reward_advantages(batch, _FakeTokenizer(), config)

    torch.testing.assert_close(
        batch.batch["format_reward_advantages"],
        torch.tensor([[0.5, 0.5, 0.5, 0.0], [-0.5, -0.5, -0.5, -0.5]]),
    )
    assert metrics["format_reward/valid_rate"] == pytest.approx(0.5)
    assert metrics["format_reward/terminated_rate"] == pytest.approx(0.5)


def test_disabled_format_reward_is_exact_noop():
    batch = _make_batch()
    config = DistillationLossConfig(loss_mode="k1", format_reward_enabled=False)

    assert add_format_reward_advantages(batch, _FakeTokenizer(), config) == {}
    assert "format_reward_advantages" not in batch.batch


def test_format_reward_config_validation():
    with pytest.raises(ValueError, match="between 0 and 1"):
        DistillationLossConfig(loss_mode="k1", format_reward_baseline=1.5)


class _ActorConfig:
    loss_agg_mode = "token-mean"
    global_batch_info = {}


def test_format_policy_loss_uses_separate_advantages():
    input_ids = torch.arange(5).unsqueeze(0)
    data = TensorDict(
        {
            "input_ids": input_ids,
            "prompts": input_ids[:, :2].clone(),
            "responses": input_ids[:, 2:].clone(),
            "attention_mask": torch.ones(1, 5, dtype=torch.long),
            "response_mask": torch.ones(1, 3, dtype=torch.long),
            "old_log_probs": torch.zeros(1, 3),
            "format_reward_advantages": torch.full((1, 3), 0.5),
            # Deliberately different: the format objective must not consume the
            # ordinary task-reward advantage tensor.
            "advantages": torch.full((1, 3), -10.0),
            "position_ids": input_ids.clone(),
        },
        batch_size=[1],
    )
    data = left_right_2_no_padding(data)
    data["prompts"] = input_ids[:, :2].clone()
    data["responses"] = input_ids[:, 2:].clone()
    offsets = data["input_ids"].offsets()
    model_output = {
        "log_probs": torch.nested.nested_tensor_from_jagged(torch.zeros(5), offsets),
    }
    config = DistillationLossConfig(loss_mode="k1", format_reward_enabled=True)

    loss, metrics = compute_format_reward_loss(model_output, data, _ActorConfig(), config)

    torch.testing.assert_close(loss, torch.tensor(-0.5))
    assert "format_reward/pg_clipfrac" in metrics
