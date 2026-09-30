# Copyright 2025 Bytedance Ltd. and/or its affiliates
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

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import torch
import torch.nn.functional as F
from tensordict import TensorDict

from verl.base_config import BaseConfig
from verl.trainer.ppo.core_algos import agg_loss, get_policy_loss_fn, kl_penalty
from verl.utils import tensordict_utils as tu
from verl.utils.metric import AggregationType, Metric
from verl.workers.config import ActorConfig, DistillationConfig, DistillationLossConfig
from verl.workers.utils.losses import ppo_loss
from verl.workers.utils.padding import no_padding_2_padding

DistillationLossFn = Callable[
    [
        ActorConfig,  # actor_config
        DistillationConfig,  # distillation_config
        dict,  # model_output
        TensorDict,  # micro batch input
    ],
    tuple[torch.Tensor, dict[str, Any]],
]


def is_distillation_enabled(config: Optional[DistillationConfig]) -> bool:
    """Check if distillation is enabled based on the provided configuration."""
    if config is None:
        return False
    return config.enabled


@dataclass
class DistillationLossSettings(BaseConfig):
    """
    Settings for a distillation loss function to be registered.

    Args:
        names (str | list[str]): Name(s) to register the distillation loss function under.
        use_topk (bool): Whether the loss function uses top-k log probabilities.
        use_estimator (bool): Whether the loss function uses single-sample KL estimators.
    """

    names: str | list[str] = field(default_factory=list)
    use_topk: bool = False
    use_estimator: bool = False

    _mutable_fields = {"names"}

    def __post_init__(self):
        self.names = [self.names] if isinstance(self.names, str) else self.names
        if sum([self.use_topk, self.use_estimator]) != 1:
            raise ValueError(
                f"Expected only one of use_estimator, use_topk, but got {self.use_estimator=}, {self.use_topk=}."
            )


DISTILLATION_LOSS_REGISTRY: dict[str, DistillationLossFn] = {}
DISTILLATION_SETTINGS_REGISTRY: dict[str, DistillationLossSettings] = {}


def register_distillation_loss(
    loss_settings: DistillationLossSettings,
) -> Callable[[DistillationLossFn], DistillationLossFn]:
    """Register a distillation loss function with the given name."""

    def decorator(func: DistillationLossFn) -> DistillationLossFn:
        for name in loss_settings.names:
            if name in DISTILLATION_LOSS_REGISTRY:
                raise ValueError(f"Distillation loss function with name '{name}' is already registered.")
            DISTILLATION_LOSS_REGISTRY[name] = func
            DISTILLATION_SETTINGS_REGISTRY[name] = loss_settings
        return func

    return decorator


def get_distillation_loss_fn(loss_name: str) -> DistillationLossFn:
    """Get the distillation loss function with a given name."""
    if loss_name not in DISTILLATION_LOSS_REGISTRY:
        raise ValueError(
            f"Unsupported loss mode: {loss_name}. Supported modes are: {list(DISTILLATION_LOSS_REGISTRY.keys())}"
        )
    return DISTILLATION_LOSS_REGISTRY[loss_name]


def get_distillation_loss_settings(loss_name: str) -> DistillationLossSettings:
    """Get the distillation loss settings with a given name."""
    if loss_name not in DISTILLATION_SETTINGS_REGISTRY:
        raise ValueError(
            f"Unsupported loss mode: {loss_name}. Supported modes are: {list(DISTILLATION_SETTINGS_REGISTRY.keys())}"
        )
    return DISTILLATION_SETTINGS_REGISTRY[loss_name]


def compute_distillation_loss_range(
    distillation_losses: torch.Tensor, response_mask: torch.Tensor
) -> dict[str, Metric]:
    """Compute min and max distillation loss over valid response tokens."""
    distillation_losses_response = distillation_losses[response_mask.bool()]
    return {
        "distillation/loss_min": Metric(AggregationType.MIN, distillation_losses_response.min()),
        "distillation/loss_max": Metric(AggregationType.MAX, distillation_losses_response.max()),
    }


def mask_opd_termination_token(
    distillation_losses: torch.Tensor,
    data: TensorDict,
    config: ActorConfig,
    loss_config: DistillationLossConfig,
) -> tuple[torch.Tensor, dict[str, Metric]]:
    """Zero the OPD contribution of model termination tokens.

    SimpleOPD replaces the teacher log-probability at configured termination
    tokens with the student's log-probability, which makes the sampled-token
    OPD delta exactly zero. Applying the equivalent zero mask to the already
    computed distillation loss keeps this concern local to OPD: task-reward
    PPO, format rewards, and student-reference KL retain the original response
    mask and the OPD aggregation denominator is unchanged.
    """
    if not loss_config.mask_termination_token:
        return distillation_losses, {}

    if "responses" not in data:
        raise KeyError("mask_termination_token=True requires response token ids in data['responses'].")
    responses = data["responses"]
    if responses.shape != distillation_losses.shape:
        raise ValueError(
            "Response token ids must match the OPD loss layout when masking termination tokens: "
            f"responses={tuple(responses.shape)}, loss={tuple(distillation_losses.shape)}."
        )

    model_config = getattr(config, "model_config", None)
    eos_token_id = getattr(model_config, "eos_token_id", None)
    if eos_token_id is None:
        raise ValueError(
            "mask_termination_token=True requires config.model_config.eos_token_id; "
            "for Qwen this id represents <|im_end|>."
        )
    eos_token_ids = [eos_token_id] if isinstance(eos_token_id, int) else list(eos_token_id)
    if not eos_token_ids:
        raise ValueError("config.model_config.eos_token_id must contain at least one termination token id.")

    termination_mask = torch.zeros_like(responses, dtype=torch.bool)
    for token_id in eos_token_ids:
        termination_mask |= responses == int(token_id)
    termination_mask &= data["response_mask"].bool()

    masked_losses = distillation_losses.masked_fill(termination_mask, 0.0)
    masked_count = termination_mask.sum()
    valid_count = data["response_mask"].bool().sum().clamp_min(1)
    metrics = {
        "distillation/termination_tokens_masked": Metric(AggregationType.SUM, masked_count),
        "distillation/termination_token_fraction": Metric(
            AggregationType.MEAN,
            masked_count.to(distillation_losses.dtype) / valid_count,
        ),
    }
    return masked_losses, metrics


def apply_opd_position_weighting(
    distillation_losses: torch.Tensor,
    response_mask: torch.Tensor,
    loss_config: DistillationLossConfig,
) -> tuple[torch.Tensor, dict[str, Metric]]:
    """Shift main OPD credit toward earlier response positions.

    For every response, valid tokens receive raw weights
    ``1 + alpha * exp(-relative_position / decay)``. Relative position runs
    from zero at the first valid token to one at the last valid token. The raw
    weights are normalized per response and projected onto the configured
    bounds while retaining an exact mean of one. Consequently, this ablation
    changes the temporal allocation of OPD credit without changing the total
    weight assigned to either short or long responses.

    Only the main distillation loss tensor passes through this helper. Other
    objectives combined by ``distillation_ppo_loss`` remain unchanged.
    """
    if not loss_config.opd_position_weighting_enabled:
        return distillation_losses, {}

    response_mask_bool = response_mask.bool()
    if response_mask_bool.shape != distillation_losses.shape:
        raise ValueError(
            "response_mask must match the OPD loss layout for position weighting: "
            f"mask={tuple(response_mask_bool.shape)}, loss={tuple(distillation_losses.shape)}."
        )

    weight_dtype = distillation_losses.dtype
    valid_counts = response_mask_bool.sum(dim=-1, keepdim=True)
    valid_positions = response_mask_bool.to(torch.long).cumsum(dim=-1) - 1
    relative_positions = valid_positions.to(weight_dtype) / (valid_counts - 1).clamp_min(1).to(weight_dtype)
    # Keep masked positions numerically benign before the exponential. This is
    # relevant for left padding (where cumsum - 1 is negative) and very small
    # decay values, for which computing exp on a masked element could overflow.
    relative_positions = relative_positions.masked_fill(~response_mask_bool, 0)

    raw_weights = 1.0 + loss_config.opd_position_weight_alpha * torch.exp(
        -relative_positions / loss_config.opd_position_weight_decay
    )
    raw_weights = raw_weights * response_mask_bool.to(weight_dtype)
    raw_means = raw_weights.sum(dim=-1, keepdim=True) / valid_counts.clamp_min(1).to(weight_dtype)
    position_weights = raw_weights / raw_means.clamp_min(torch.finfo(weight_dtype).tiny)
    position_weights = position_weights.clamp(
        min=loss_config.opd_position_weight_min,
        max=loss_config.opd_position_weight_max,
    )
    position_weights = position_weights * response_mask_bool.to(weight_dtype)

    # Clipping can move a row's mean away from one. Project the clipped values
    # back to the feasible mean-one hyperplane using available upper/lower box
    # capacity. The affine adjustment preserves the decreasing position order.
    target_sums = valid_counts.to(weight_dtype)
    current_sums = position_weights.sum(dim=-1, keepdim=True)
    sum_gap = target_sums - current_sums
    upper_capacity = (
        loss_config.opd_position_weight_max - position_weights
    ) * response_mask_bool.to(weight_dtype)
    lower_capacity = (
        position_weights - loss_config.opd_position_weight_min
    ) * response_mask_bool.to(weight_dtype)
    position_weights = position_weights + sum_gap.clamp_min(0) * upper_capacity / upper_capacity.sum(
        dim=-1, keepdim=True
    ).clamp_min(torch.finfo(weight_dtype).tiny)
    position_weights = position_weights - (-sum_gap).clamp_min(0) * lower_capacity / lower_capacity.sum(
        dim=-1, keepdim=True
    ).clamp_min(torch.finfo(weight_dtype).tiny)
    position_weights = position_weights * response_mask_bool.to(weight_dtype)

    valid_weights = position_weights[response_mask_bool]
    zero = distillation_losses.new_zeros(())
    if valid_weights.numel():
        front_counts = torch.ceil(valid_counts.to(torch.float32) * 0.25).to(torch.long).clamp_min(1)
        back_starts = torch.floor(valid_counts.to(torch.float32) * 0.75).to(torch.long)
        front_mask = response_mask_bool & (valid_positions < front_counts)
        back_mask = response_mask_bool & (valid_positions >= back_starts)
        front_mean = position_weights[front_mask].mean()
        back_mean = position_weights[back_mask].mean()
        weight_mean = valid_weights.mean()
        weight_min = valid_weights.min()
        weight_max = valid_weights.max()
    else:
        front_mean = back_mean = weight_mean = weight_min = weight_max = zero

    metrics = {
        "distillation/opd_position_weight_mean": Metric(AggregationType.MEAN, weight_mean.detach()),
        "distillation/opd_position_weight_min": Metric(AggregationType.MIN, weight_min.detach()),
        "distillation/opd_position_weight_max": Metric(AggregationType.MAX, weight_max.detach()),
        "distillation/opd_position_weight_front_quarter": Metric(
            AggregationType.MEAN, front_mean.detach()
        ),
        "distillation/opd_position_weight_back_quarter": Metric(
            AggregationType.MEAN, back_mean.detach()
        ),
    }
    return distillation_losses * position_weights.detach(), metrics


def compute_student_visual_dependency_mask(
    model_output: dict,
    data: TensorDict,
    loss_config: DistillationLossConfig,
) -> tuple[Optional[torch.Tensor], dict[str, Metric]]:
    """Select each response's most visually dependent student tokens.

    This follows VPPO-RL's Token Gradient Filtering score. For sampled tokens
    from the clean-image student, let ``d = log p_masked - log p_clean``. The
    visual-dependency score is Schulman's non-negative low-variance estimator
    ``exp(d) - d - 1``. Exactly ``ceil(valid_count * fraction)`` tokens are
    selected independently in every response.

    The returned mask is detached. Callers multiply only the main OPD token
    loss by it while retaining the original aggregation denominator, matching
    VPPO-RL's gradient-mask semantics.
    """
    if not loss_config.student_visual_dependency_filter_enabled:
        return None, {}

    required = {"log_probs", "counterfactual_log_probs"}
    missing = required - model_output.keys()
    if missing:
        raise KeyError(
            "student_visual_dependency_filter_enabled=True is missing student outputs: "
            f"{sorted(missing)}"
        )

    clean_log_probs = no_padding_2_padding(model_output["log_probs"], data).detach()
    masked_log_probs = no_padding_2_padding(model_output["counterfactual_log_probs"], data).detach()
    response_mask = data["response_mask"].bool()
    if not (clean_log_probs.shape == masked_log_probs.shape == response_mask.shape):
        raise ValueError(
            "Student visual-dependency tensors must match the response layout: "
            f"clean={tuple(clean_log_probs.shape)}, masked={tuple(masked_log_probs.shape)}, "
            f"mask={tuple(response_mask.shape)}."
        )

    log_ratio = (masked_log_probs - clean_log_probs).clamp(-20.0, 20.0)
    visual_dependency = (log_ratio.exp() - log_ratio - 1.0).clamp(min=0.0, max=10.0)
    scores_for_sort = visual_dependency.masked_fill(~response_mask, -torch.inf)

    valid_counts = response_mask.sum(dim=-1)
    keep_counts = torch.ceil(
        valid_counts.to(torch.float32) * loss_config.student_visual_dependency_top_fraction
    ).to(torch.long)
    sorted_indices = torch.argsort(scores_for_sort, dim=-1, descending=True)
    ranks = torch.arange(response_mask.shape[-1], device=response_mask.device).expand_as(response_mask)
    ranked_keep = ranks < keep_counts.unsqueeze(-1)
    selected_mask = torch.zeros_like(response_mask)
    selected_mask.scatter_(dim=-1, index=sorted_indices, src=ranked_keep)
    selected_mask &= response_mask

    zero = visual_dependency.new_zeros(())
    valid_scores = visual_dependency[response_mask]
    selected_scores = visual_dependency[selected_mask]
    rejected_scores = visual_dependency[response_mask & ~selected_mask]
    selected_count = selected_mask.sum()
    valid_count = response_mask.sum().clamp_min(1)

    safe_keep_counts = keep_counts.clamp_min(1)
    sorted_scores = torch.gather(scores_for_sort, dim=-1, index=sorted_indices)
    thresholds = torch.gather(
        sorted_scores,
        dim=-1,
        index=(safe_keep_counts - 1).unsqueeze(-1),
    ).squeeze(-1)
    thresholds = thresholds[keep_counts > 0]

    metrics = {
        "distillation/student_visual_dependency_mean": Metric(
            AggregationType.MEAN, valid_scores.mean() if valid_scores.numel() else zero
        ),
        "distillation/student_visual_dependency_selected_mean": Metric(
            AggregationType.MEAN, selected_scores.mean() if selected_scores.numel() else zero
        ),
        "distillation/student_visual_dependency_rejected_mean": Metric(
            AggregationType.MEAN, rejected_scores.mean() if rejected_scores.numel() else zero
        ),
        "distillation/student_visual_dependency_threshold": Metric(
            AggregationType.MEAN, thresholds.mean() if thresholds.numel() else zero
        ),
        "distillation/student_visual_dependency_keep_fraction": Metric(
            AggregationType.MEAN,
            selected_count.to(visual_dependency.dtype) / valid_count,
        ),
    }
    return selected_mask.detach(), metrics


def apply_teacher_counterfactual_visual_weighting(
    distillation_losses: torch.Tensor,
    data: TensorDict,
    loss_config: DistillationLossConfig,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Upweight existing token losses using teacher-only image sensitivity.

    The additive weight keeps every original OPD token at weight >= 1. Only
    tokens whose sampled-token probability is higher with the real image than
    with the blank counterfactual receive extra weight. When disabled, this is
    an exact no-op and does not require the counterfactual tensor.
    """
    if not loss_config.visual_grounding_enabled:
        return distillation_losses, {}
    if "teacher_negative_logprobs" not in data:
        raise KeyError(
            "visual_grounding_enabled=True requires teacher_negative_logprobs; "
            "ensure counterfactual teacher scoring is enabled in the rollout path."
        )

    teacher_positive = no_padding_2_padding(data["teacher_logprobs"], data).squeeze(-1)
    teacher_negative = no_padding_2_padding(data["teacher_negative_logprobs"], data).squeeze(-1)
    if teacher_positive.shape != distillation_losses.shape or teacher_negative.shape != distillation_losses.shape:
        raise ValueError(
            "Counterfactual teacher logprob shapes must match token losses: "
            f"positive={tuple(teacher_positive.shape)}, negative={tuple(teacher_negative.shape)}, "
            f"loss={tuple(distillation_losses.shape)}."
        )

    visual_shift = teacher_positive - teacher_negative
    visual_evidence = torch.relu(visual_shift - loss_config.visual_grounding_weight_threshold)
    visual_weight = 1.0 + loss_config.visual_grounding_weight_scale * visual_evidence
    visual_weight = visual_weight.clamp(max=loss_config.visual_grounding_max_weight).detach()

    response_mask = data["response_mask"].bool()
    valid_shift = visual_shift[response_mask]
    valid_weight = visual_weight[response_mask]
    metrics = {
        "distillation/visual_grounding_shift": Metric(AggregationType.MEAN, valid_shift.mean()),
        "distillation/visual_grounding_active_fraction": Metric(
            AggregationType.MEAN,
            (valid_shift > loss_config.visual_grounding_weight_threshold).float().mean(),
        ),
        "distillation/visual_grounding_weight": Metric(AggregationType.MEAN, valid_weight.mean()),
        "distillation/visual_grounding_weight_max": Metric(AggregationType.MAX, valid_weight.max()),
    }
    return distillation_losses * visual_weight, metrics


def compute_token_level_va_group_weights(
    distillation_losses: torch.Tensor,
    data: TensorDict,
    loss_config: DistillationLossConfig,
) -> tuple[Optional[torch.Tensor], dict[str, Any]]:
    """Build the per-rollout grouped-KL weights from token visual advantage.

    VA-OPD defines ``VA_t = relu(log p_T(y_t | image) -
    log p_T(y_t | pixelated_image))``. Within each rollout, exactly the top
    ``p_v`` fraction of valid response tokens forms the high-VA group. Each
    group is normalized independently, so its total weight is ``lambda`` or
    ``1 - lambda`` regardless of sequence length.

    The returned weights sum to one on every non-empty response. Callers use
    token-sum followed by sequence-mean aggregation to reproduce Equation (5)
    of the paper. When disabled, ``None`` is returned as an exact no-op signal.
    """
    if not loss_config.token_level_va_enabled:
        return None, {}
    if "teacher_negative_logprobs" not in data:
        raise KeyError(
            "token_level_va_enabled=True requires teacher_negative_logprobs; "
            "ensure pixelated-image teacher scoring is enabled in the rollout path."
        )

    teacher_positive = no_padding_2_padding(data["teacher_logprobs"], data).squeeze(-1)
    teacher_negative = no_padding_2_padding(data["teacher_negative_logprobs"], data).squeeze(-1)
    response_mask = data["response_mask"].bool()
    if not (
        teacher_positive.shape
        == teacher_negative.shape
        == distillation_losses.shape
        == response_mask.shape
    ):
        raise ValueError(
            "Token-level VA tensors must match the response loss layout: "
            f"positive={tuple(teacher_positive.shape)}, negative={tuple(teacher_negative.shape)}, "
            f"loss={tuple(distillation_losses.shape)}, mask={tuple(response_mask.shape)}."
        )

    visual_advantage = torch.relu(teacher_positive - teacher_negative).detach()
    group_weights = torch.zeros_like(distillation_losses)
    high_mask = torch.zeros_like(response_mask)
    high_fraction = loss_config.token_level_va_high_fraction
    high_weight = loss_config.token_level_va_high_weight

    for row in range(response_mask.shape[0]):
        valid_indices = torch.nonzero(response_mask[row], as_tuple=False).squeeze(-1)
        token_count = int(valid_indices.numel())
        if token_count == 0:
            continue
        if token_count == 1:
            # Equation (5) assumes both groups are non-empty. For the only
            # degenerate case, preserve a well-defined unit-weight token.
            group_weights[row, valid_indices] = 1.0
            high_mask[row, valid_indices] = True
            continue

        # The paper does not specify fractional-count rounding. Ceil preserves
        # the requested top fraction while ensuring both groups are non-empty.
        high_count = min(token_count - 1, max(1, math.ceil(token_count * high_fraction)))
        ranked_positions = torch.topk(
            visual_advantage[row, valid_indices], k=high_count, largest=True, sorted=False
        ).indices
        high_indices = valid_indices[ranked_positions]
        high_mask[row, high_indices] = True
        low_mask_row = response_mask[row] & ~high_mask[row]
        low_count = token_count - high_count
        group_weights[row, high_indices] = high_weight / high_count
        group_weights[row, low_mask_row] = (1.0 - high_weight) / low_count

    valid_va = visual_advantage[response_mask]
    valid_high_mask = high_mask & response_mask
    valid_low_mask = response_mask & ~high_mask
    zero = visual_advantage.new_zeros(())
    high_va_mean = visual_advantage[valid_high_mask].mean() if valid_high_mask.any() else zero
    low_va_mean = visual_advantage[valid_low_mask].mean() if valid_low_mask.any() else zero
    metrics = {
        "distillation/token_va_mean": Metric(
            AggregationType.MEAN, valid_va.mean() if valid_va.numel() else zero
        ),
        "distillation/token_va_max": Metric(
            AggregationType.MAX, valid_va.max() if valid_va.numel() else zero
        ),
        "distillation/token_va_positive_fraction": Metric(
            AggregationType.MEAN, (valid_va > 0).float().mean() if valid_va.numel() else zero
        ),
        "distillation/token_va_high_mean": Metric(AggregationType.MEAN, high_va_mean),
        "distillation/token_va_low_mean": Metric(AggregationType.MEAN, low_va_mean),
        "distillation/token_va_actual_high_fraction": Metric(
            AggregationType.MEAN,
            valid_high_mask.float().sum() / response_mask.float().sum().clamp_min(1.0),
        ),
    }
    return group_weights.detach(), metrics


def _response_position_split(
    response_mask: torch.Tensor,
    papo_fraction: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split each valid response into leading PAPO and trailing residual regions.

    The leading region receives ``ceil(valid_length * papo_fraction)`` tokens.
    For responses of length at least two, both regions contain at least one
    token. Float weights renormalize each region by ``valid_length / region_length``
    so the auxiliary coefficients retain their full-token scale while the
    existing global response-token denominator remains valid.
    """
    response_mask = response_mask.bool()
    valid_counts = response_mask.sum(dim=-1)
    papo_counts = torch.ceil(valid_counts.to(torch.float32) * papo_fraction).to(torch.long)
    papo_counts = torch.where(
        valid_counts >= 2,
        papo_counts.clamp_min(1).minimum(valid_counts - 1),
        valid_counts,
    )
    valid_positions = response_mask.to(torch.long).cumsum(dim=-1) - 1
    papo_mask = response_mask & (valid_positions < papo_counts.unsqueeze(-1))
    residual_mask = response_mask & ~papo_mask

    residual_counts = valid_counts - papo_counts
    valid_counts_float = valid_counts.to(torch.float32)
    papo_weights = papo_mask.to(torch.float32) * (
        valid_counts_float / papo_counts.clamp_min(1).to(torch.float32)
    ).unsqueeze(-1)
    residual_weights = residual_mask.to(torch.float32) * (
        valid_counts_float / residual_counts.clamp_min(1).to(torch.float32)
    ).unsqueeze(-1)
    return papo_mask, residual_mask, papo_weights, residual_weights


def compute_counterfactual_residual_matching_loss(
    model_output: dict,
    data: TensorDict,
    config: ActorConfig,
    loss_config: DistillationLossConfig,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Uniformly match teacher and student sensitivity to random image masking.

    The original OPD loss is not modified. This auxiliary objective matches
    ``log p(y_t | original) - log p(y_t | masked)`` at each sampled response
    token. When direction-aware matching is enabled, tokens whose teacher
    residual is non-positive contribute zero loss. The original response-token
    denominator is retained so batches with few selected tokens do not receive
    disproportionately large gradients.
    """
    if not loss_config.counterfactual_residual_enabled:
        log_probs = model_output["log_probs"]
        log_prob_values = log_probs.values() if log_probs.is_nested else log_probs
        return log_prob_values.sum() * 0.0, {}

    required_model_keys = {"log_probs", "counterfactual_log_probs"}
    required_data_keys = {"teacher_logprobs", "teacher_counterfactual_logprobs"}
    missing = (required_model_keys - model_output.keys()) | (required_data_keys - set(data.keys()))
    if missing:
        raise KeyError(f"Counterfactual residual matching is missing required fields: {sorted(missing)}")

    student_positive = no_padding_2_padding(model_output["log_probs"], data)
    student_counterfactual = no_padding_2_padding(model_output["counterfactual_log_probs"], data)
    teacher_positive = no_padding_2_padding(data["teacher_logprobs"], data).squeeze(-1)
    teacher_counterfactual = no_padding_2_padding(data["teacher_counterfactual_logprobs"], data).squeeze(-1)
    response_mask = data["response_mask"].bool()
    if not (
        student_positive.shape
        == student_counterfactual.shape
        == teacher_positive.shape
        == teacher_counterfactual.shape
        == response_mask.shape
    ):
        raise ValueError("Counterfactual residual tensors must all match the response layout.")

    teacher_residual = (teacher_positive - teacher_counterfactual).detach()
    student_residual = student_positive - student_counterfactual

    per_token_loss = F.smooth_l1_loss(
        student_residual,
        teacher_residual,
        reduction="none",
        beta=loss_config.counterfactual_residual_beta,
    )
    if loss_config.counterfactual_residual_direction_aware:
        direction_mask = teacher_residual > 0
    else:
        direction_mask = torch.ones_like(response_mask, dtype=torch.bool)
    position_mask = response_mask
    position_weights = response_mask.to(per_token_loss.dtype)
    if loss_config.papo_residual_position_split_enabled:
        _, position_mask, _, position_weights = _response_position_split(
            response_mask,
            loss_config.papo_residual_papo_fraction,
        )
        position_weights = position_weights.to(per_token_loss.dtype)
    selected_mask = position_mask & direction_mask
    per_token_loss = per_token_loss * direction_mask.to(per_token_loss.dtype) * position_weights
    residual_loss = agg_loss(
        loss_mat=per_token_loss,
        loss_mask=response_mask,
        loss_agg_mode=config.loss_agg_mode,
        **config.global_batch_info,
    )

    valid_teacher_residual = teacher_residual[selected_mask]
    valid_student_residual = student_residual[selected_mask]
    zero = per_token_loss.sum() * 0.0
    teacher_abs = valid_teacher_residual.abs().mean() if valid_teacher_residual.numel() else zero
    student_abs = valid_student_residual.abs().mean() if valid_student_residual.numel() else zero
    gap_abs = (
        (valid_student_residual - valid_teacher_residual).abs().mean()
        if valid_teacher_residual.numel()
        else zero
    )
    direction_keep_fraction = (response_mask & direction_mask).float().sum() / response_mask.float().sum().clamp_min(
        1.0
    )
    position_keep_fraction = position_mask.float().sum() / response_mask.float().sum().clamp_min(1.0)
    metrics = {
        "distillation/counterfactual_residual_teacher_abs": Metric(
            AggregationType.MEAN, teacher_abs
        ),
        "distillation/counterfactual_residual_student_abs": Metric(
            AggregationType.MEAN, student_abs
        ),
        "distillation/counterfactual_residual_gap_abs": Metric(
            AggregationType.MEAN, gap_abs
        ),
        "distillation/counterfactual_residual_direction_keep_fraction": Metric(
            AggregationType.MEAN, direction_keep_fraction
        ),
        "distillation/counterfactual_residual_position_keep_fraction": Metric(
            AggregationType.MEAN, position_keep_fraction
        ),
    }
    return residual_loss, metrics


def compute_papo_perception_kl(
    model_output: dict,
    data: TensorDict,
    config: ActorConfig,
    loss_config: DistillationLossConfig,
    dp_group=None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Compute PAPO's sampled-token low-variance perception KL estimate.

    Rollout tokens are sampled with the original image. For each response token,
    PAPO compares its current student log-probability under the original image
    with the log-probability of the same token under a randomly masked image.
    The masked branch is a detached reference, as in the efficient path of the
    official implementation. By default, a teacher gate keeps only sampled
    tokens whose log-probability is higher with the original image than with
    the same masked image, and scales their PAPO loss by that teacher gap. The
    ``student_positive`` ablation instead uses only the detached binary gate
    ``log p_student(original) - log p_student(masked) > 0``; selected tokens
    receive unit weight and no teacher contrast is consumed. When gating is
    disabled, every selected valid response token receives the unweighted PAPO
    loss. The rate-matched random ablation randomly moves
    the positive teacher-gap weights to an equal number of valid tokens within
    each response. This function returns a positive KL estimate;
    callers maximize it by subtracting
    ``papo_coef * perception_kl``.
    """
    if not loss_config.papo_enabled:
        log_probs = model_output["log_probs"]
        log_prob_values = log_probs.values() if log_probs.is_nested else log_probs
        return log_prob_values.sum() * 0.0, {}

    required_model_keys = {"log_probs", "counterfactual_log_probs"}
    # VGD and PAPO intentionally share one negative view. Token-level VA-OPD
    # does not: its pixelated scores rank VA tokens, while PAPO's separately
    # masked scores gate the far-push objective.
    teacher_counterfactual_key = (
        "teacher_negative_logprobs" if loss_config.vgd_enabled else "teacher_counterfactual_logprobs"
    )
    teacher_original_key = "teacher_sampled_logprobs" if "teacher_sampled_logprobs" in data else "teacher_logprobs"
    teacher_gate_required = (
        loss_config.papo_teacher_gating_enabled
        and loss_config.papo_teacher_gate_mode != "student_positive"
    )
    required_data_keys = (
        {teacher_original_key, teacher_counterfactual_key}
        if teacher_gate_required
        else set()
    )
    missing = (required_model_keys - model_output.keys()) | (required_data_keys - set(data.keys()))
    if missing:
        raise KeyError(f"PAPO is missing required fields: {sorted(missing)}")

    original_log_probs = no_padding_2_padding(model_output["log_probs"], data)
    masked_log_probs = no_padding_2_padding(model_output["counterfactual_log_probs"], data).detach()
    response_mask = data["response_mask"].bool()
    if not (original_log_probs.shape == masked_log_probs.shape == response_mask.shape):
        raise ValueError("PAPO student log-probability tensors must match the response layout.")

    if (
        loss_config.papo_teacher_gating_enabled
        and loss_config.papo_teacher_gate_mode == "student_positive"
    ):
        student_visual_delta = (original_log_probs.detach() - masked_log_probs).detach()
        teacher_visual_scale = (student_visual_delta > 0).to(original_log_probs.dtype)
    elif loss_config.papo_teacher_gating_enabled:
        teacher_original_log_probs = no_padding_2_padding(data[teacher_original_key], data).squeeze(-1)
        teacher_masked_log_probs = no_padding_2_padding(
            data[teacher_counterfactual_key], data
        ).squeeze(-1)
        if not (
            teacher_original_log_probs.shape
            == teacher_masked_log_probs.shape
            == response_mask.shape
        ):
            raise ValueError("PAPO teacher log-probability tensors must match the response layout.")
        teacher_visual_scale = (
            teacher_original_log_probs - teacher_masked_log_probs
        ).clamp_min(0).detach()
    else:
        teacher_visual_scale = torch.ones_like(original_log_probs)

    # Official PAPO low-variance estimator (Schulman's k3): with rollout tokens
    # sampled from pi(original), E[exp(d) - d - 1] estimates
    # KL[pi(original) || pi(masked)], where d=log pi(masked)-log pi(original).
    masked_to_original_log_ratio = (masked_log_probs - original_log_probs).clamp(
        min=-loss_config.papo_log_ratio_clip,
        max=loss_config.papo_log_ratio_clip,
    )
    per_token_kl = (
        masked_to_original_log_ratio.exp() - masked_to_original_log_ratio - 1.0
    ).clamp(max=loss_config.papo_kl_max)
    position_mask = response_mask
    if loss_config.papo_residual_position_split_enabled:
        position_mask, _, _, _ = _response_position_split(
            response_mask,
            loss_config.papo_residual_papo_fraction,
        )
    elif loss_config.papo_front_only_enabled:
        position_mask, _, _, _ = _response_position_split(
            response_mask,
            loss_config.papo_front_fraction,
        )
    if (
        loss_config.papo_teacher_gating_enabled
        and loss_config.papo_teacher_gate_mode == "rate_matched_random"
    ):
        # Preserve the positive-weight multiset and active-token count for
        # every response, but destroy its alignment with the teacher-selected
        # positions. Restrict both source weights and sampled destinations to
        # the active positional region so positional ablations remain valid.
        randomized_scale = torch.zeros_like(teacher_visual_scale)
        for row in range(position_mask.shape[0]):
            candidate_indices = position_mask[row].nonzero(as_tuple=False).flatten()
            positive_scales = teacher_visual_scale[row, candidate_indices]
            positive_scales = positive_scales[positive_scales > 0]
            positive_count = positive_scales.numel()
            if positive_count == 0:
                continue
            destination_order = torch.randperm(
                candidate_indices.numel(), device=candidate_indices.device
            )[:positive_count]
            scale_order = torch.randperm(
                positive_count, device=positive_scales.device
            )
            randomized_scale[
                row, candidate_indices[destination_order]
            ] = positive_scales[scale_order]
        teacher_visual_scale = randomized_scale
    teacher_gate_mask = (
        teacher_visual_scale > 0
        if loss_config.papo_teacher_gating_enabled
        else torch.ones_like(position_mask)
    )
    active_mask = position_mask & teacher_gate_mask

    # Compute the active-token denominator globally so DDP partitions with
    # different valid lengths or gate rates retain the same weighting.
    del dp_group
    global_active_count = tu.get_non_tensor_data(
        data=data,
        key="papo_normalization_token_count",
        default=None,
    )
    if global_active_count is None:
        # Backward compatibility for batches produced before the normalization
        # metadata received its more precise name.
        global_active_count = tu.get_non_tensor_data(
            data=data,
            key="papo_active_token_count",
            default=None,
        )
    if global_active_count is None:
        # A student-derived gate is unavailable before the model forward. Keep
        # its normalization exact across ranks and microbatches by treating it
        # as a binary gradient mask over the fixed candidate-token denominator.
        normalization_mask = (
            position_mask
            if loss_config.papo_teacher_gate_mode == "student_positive"
            else active_mask
        )
        global_active_count = normalization_mask.sum()
    if isinstance(global_active_count, torch.Tensor):
        global_active_count_value = int(global_active_count.item())
    else:
        global_active_count_value = int(global_active_count)
    dp_size = int(
        tu.get_non_tensor_data(
            data=data,
            key="dp_size",
            default=config.global_batch_info.get("dp_size", 1),
        )
    )

    zero = per_token_kl.sum() * 0.0
    if global_active_count_value == 0:
        perception_kl = zero
    else:
        perception_kl = agg_loss(
            loss_mat=per_token_kl * teacher_visual_scale,
            loss_mask=active_mask,
            loss_agg_mode="token-mean",
            dp_size=dp_size,
            batch_num_tokens=global_active_count_value,
        )

    valid_log_ratio = masked_to_original_log_ratio[position_mask]
    valid_teacher_scale = teacher_visual_scale[position_mask]
    log_ratio_mean = valid_log_ratio.mean() if valid_log_ratio.numel() else zero
    log_ratio_abs = valid_log_ratio.abs().mean() if valid_log_ratio.numel() else zero
    position_keep_fraction = position_mask.float().sum() / response_mask.float().sum().clamp_min(1.0)
    metrics = {
        "distillation/papo_perception_kl": Metric(AggregationType.MEAN, perception_kl.detach()),
        "distillation/papo_masked_to_original_log_ratio": Metric(
            AggregationType.MEAN, log_ratio_mean.detach()
        ),
        "distillation/papo_log_ratio_abs": Metric(AggregationType.MEAN, log_ratio_abs.detach()),
        "distillation/papo_position_keep_fraction": Metric(
            AggregationType.MEAN, position_keep_fraction.detach()
        ),
        "distillation/papo_gate_keep_fraction": Metric(
            AggregationType.MEAN,
            (
                (valid_teacher_scale > 0).float().mean()
                if valid_teacher_scale.numel()
                else zero
            ).detach(),
        ),
        "distillation/papo_gate_scale": Metric(
            AggregationType.MEAN,
            (valid_teacher_scale.mean() if valid_teacher_scale.numel() else zero).detach(),
        ),
        # Backward-compatible aliases for existing experiment dashboards. In
        # student_positive mode these contain the binary student gate values.
        "distillation/papo_teacher_gate_keep_fraction": Metric(
            AggregationType.MEAN,
            (
                (valid_teacher_scale > 0).float().mean()
                if valid_teacher_scale.numel()
                else zero
            ).detach(),
        ),
        "distillation/papo_teacher_visual_scale": Metric(
            AggregationType.MEAN,
            (valid_teacher_scale.mean() if valid_teacher_scale.numel() else zero).detach(),
        ),
        "distillation/papo_student_gate_enabled": Metric(
            AggregationType.MEAN,
            zero.new_tensor(
                float(
                    loss_config.papo_teacher_gating_enabled
                    and loss_config.papo_teacher_gate_mode == "student_positive"
                )
            ),
        ),
        "distillation/papo_random_gate_enabled": Metric(
            AggregationType.MEAN,
            zero.new_tensor(
                float(
                    loss_config.papo_teacher_gating_enabled
                    and loss_config.papo_teacher_gate_mode == "rate_matched_random"
                )
            ),
        ),
    }
    return perception_kl, metrics


def compute_gaussian_near_pull_kl(
    model_output: dict,
    data: TensorDict,
    config: ActorConfig,
    loss_config: DistillationLossConfig,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Compute clean-to-Gaussian sampled-token consistency KL.

    Rollout tokens are sampled under the clean image. The Gaussian-noise arm
    is detached and used as the reference in the same k3/low-variance KL
    estimator as PAPO. Unlike PAPO, callers *minimize* this positive quantity
    to pull the clean and meaning-preserving near policies together.
    """
    if not loss_config.gaussian_near_pull_enabled:
        log_probs = model_output["log_probs"]
        log_prob_values = log_probs.values() if log_probs.is_nested else log_probs
        return log_prob_values.sum() * 0.0, {}

    required_model_keys = {"log_probs", "gaussian_near_log_probs"}
    missing = required_model_keys - model_output.keys()
    if missing:
        raise KeyError(f"Gaussian near pull is missing required fields: {sorted(missing)}")

    original_log_probs = no_padding_2_padding(model_output["log_probs"], data)
    near_log_probs = no_padding_2_padding(
        model_output["gaussian_near_log_probs"], data
    ).detach()
    response_mask = data["response_mask"].bool()
    if not (original_log_probs.shape == near_log_probs.shape == response_mask.shape):
        raise ValueError("Gaussian near-pull log-probability tensors must match the response layout.")

    near_to_original_log_ratio = (near_log_probs - original_log_probs).clamp(
        min=-loss_config.gaussian_near_pull_log_ratio_clip,
        max=loss_config.gaussian_near_pull_log_ratio_clip,
    )
    per_token_kl = (
        near_to_original_log_ratio.exp() - near_to_original_log_ratio - 1.0
    ).clamp(max=loss_config.gaussian_near_pull_kl_max)
    near_pull_kl = agg_loss(
        loss_mat=per_token_kl,
        loss_mask=response_mask,
        loss_agg_mode=config.loss_agg_mode,
        **config.global_batch_info,
    )

    valid_ratio = near_to_original_log_ratio[response_mask]
    zero = per_token_kl.sum() * 0.0
    ratio_mean = valid_ratio.mean() if valid_ratio.numel() else zero
    ratio_abs = valid_ratio.abs().mean() if valid_ratio.numel() else zero
    metrics = {
        "distillation/gaussian_near_pull_kl": Metric(
            AggregationType.MEAN, near_pull_kl.detach()
        ),
        "distillation/gaussian_near_to_original_log_ratio": Metric(
            AggregationType.MEAN, ratio_mean.detach()
        ),
        "distillation/gaussian_near_log_ratio_abs": Metric(
            AggregationType.MEAN, ratio_abs.detach()
        ),
    }
    return near_pull_kl, metrics


def compute_topk_loss(
    config: ActorConfig,
    distillation_config: DistillationConfig,
    data: TensorDict,
    student_logits: torch.Tensor,
    data_format: str,
) -> torch.Tensor:
    """Compute the topk loss in logit processor.

    Returns:
    - distillation_losses: (bsz, seqlen/cp_size)
    - student_mass: (bsz, seqlen/cp_size)
    - teacher_mass: (bsz, seqlen/cp_size)
    """
    loss_mode = distillation_config.distillation_loss.loss_mode
    match config.strategy:
        case "fsdp":
            import verl.trainer.distillation.fsdp.losses as fsdp_losses

            if loss_mode == "jsd_topk":
                distillation_loss_fn = fsdp_losses.compute_jsd_topk
            else:
                distillation_loss_fn = fsdp_losses.compute_forward_kl_topk
        case "megatron":
            if loss_mode == "jsd_topk":
                raise NotImplementedError(
                    "jsd_topk currently supports FSDP; Megatron vocab-parallel JSD is not implemented."
                )
            import verl.trainer.distillation.megatron.losses as megatron_losses

            distillation_loss_fn = megatron_losses.compute_forward_kl_topk
        case _:
            raise NotImplementedError(f"Unsupported strategy: {config.strategy=}")

    outputs = distillation_loss_fn(
        student_logits=student_logits,
        teacher_topk_log_probs=data["teacher_logprobs"],
        teacher_topk_ids=data["teacher_ids"],
        config=distillation_config,
        data_format=data_format,
    )

    expected_shape = student_logits.shape[:2]
    for k, v in outputs.items():
        assert v.shape == expected_shape, f"Expected shape {expected_shape}, but got {v.shape} for {k=}."

    return outputs


def distillation_ppo_loss(
    config: ActorConfig,
    distillation_config: Optional[DistillationConfig],
    model_output: dict = None,
    data: TensorDict = None,
    dp_group=None,
    student_logits: torch.Tensor = None,
    data_format: str = "thd",
):
    """Loss function used both for logit processor and final policy loss.
    - student_logits is not None, compute the topk loss in logit processor.
    - student_logits is None, compute final policy loss.

    [split sequence across sp/cp groups]
                   |
    [model forward and output logits: (bsz, seqlen/cp_size, vocab_size/tp_size)]
                   |
    [logits processor compute topk loss: (bsz, seqlen/cp_size)]
                   |
    [all gather topk loss across sp/cp groups: (bsz, seqlen)]
                   |
    [combine topk loss with policy loss]

    Args:
        config: Actor configuration.
        distillation_config: Distillation configuration.
        model_output: Model output, including log_probs, entropy.
        data: Micro input batch, contains
          - teacher_logprobs: (bsz, seqlen, topk)
          - teacher_ids: (bsz, seqlen, topk)
        student_logits: (bsz, seqlen/cp_size, vocab_size/tp_size).
        data_format: "thd" or "bshd", models not support THD format, e.g GPT-OSS, Qwen3.5

    Returns:
    - student_logits is not None, return the topk loss tensor (bsz, seqlen/cp_size).
    - student_logits is None, return the final policy loss scalar and metrics.
    """

    # Called as logits processor
    if student_logits is not None:
        return compute_topk_loss(config, distillation_config, data, student_logits, data_format)

    # Called as final policy loss
    distillation_loss_config = distillation_config.distillation_loss
    distill_loss, distill_metrics = distillation_loss(config, distillation_config, model_output, data)
    # Pure OPD replaces task-reward PPO with the teacher distillation
    # objective, but an enabled frozen student-reference KL remains additive.
    # This matches SimpleOPD's L = L_OPD + beta * KL(student || initial_student).
    policy_loss, policy_metrics = ppo_loss(
        config,
        model_output,
        data,
        dp_group,
        include_task_loss=distillation_loss_config.use_task_rewards,
    )

    # Combine distillation with policy loss
    policy_metrics.update(distill_metrics)
    distillation_loss_coef = (
        distillation_loss_config.distillation_loss_coef if distillation_loss_config.use_task_rewards else 1.0
    )
    policy_loss += distill_loss * distillation_loss_coef
    policy_metrics["distillation/loss"] = Metric(value=distill_loss, aggregation=AggregationType.SUM)

    if distillation_loss_config.format_reward_enabled:
        format_reward_loss, format_reward_metrics = compute_format_reward_loss(
            model_output=model_output,
            data=data,
            config=config,
            loss_config=distillation_loss_config,
        )
        policy_loss += format_reward_loss * distillation_loss_config.format_reward_coef
        policy_metrics.update(format_reward_metrics)
        policy_metrics["format_reward/loss"] = Metric(
            value=format_reward_loss, aggregation=AggregationType.SUM
        )

    if distillation_loss_config.counterfactual_residual_enabled:
        residual_loss, residual_metrics = compute_counterfactual_residual_matching_loss(
            model_output=model_output,
            data=data,
            config=config,
            loss_config=distillation_loss_config,
        )
        policy_loss += residual_loss * distillation_loss_config.counterfactual_residual_coef
        policy_metrics.update(residual_metrics)
        policy_metrics["distillation/counterfactual_residual_loss"] = Metric(
            value=residual_loss, aggregation=AggregationType.SUM
        )

    if distillation_loss_config.papo_enabled:
        perception_kl, papo_metrics = compute_papo_perception_kl(
            model_output=model_output,
            data=data,
            config=config,
            loss_config=distillation_loss_config,
            dp_group=dp_group,
        )
        # The surrounding training code minimizes policy_loss; PAPO maximizes
        # the divergence between the original and masked student policies.
        policy_loss -= perception_kl * distillation_loss_config.papo_coef
        policy_metrics.update(papo_metrics)

    if distillation_loss_config.gaussian_near_pull_enabled:
        near_pull_kl, near_pull_metrics = compute_gaussian_near_pull_kl(
            model_output=model_output,
            data=data,
            config=config,
            loss_config=distillation_loss_config,
        )
        # Near pull has the opposite sign from PAPO: minimizing the policy loss
        # minimizes clean-to-Gaussian divergence.
        policy_loss += near_pull_kl * distillation_loss_config.gaussian_near_pull_coef
        policy_metrics.update(near_pull_metrics)

    return policy_loss, policy_metrics


def compute_format_reward_loss(
    model_output: dict,
    data: TensorDict,
    config: ActorConfig,
    loss_config: DistillationLossConfig,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Compute the independent clipped policy loss for format advantages."""
    if "format_reward_advantages" not in data:
        raise KeyError(
            "format_reward_enabled=True requires format_reward_advantages; "
            "ensure the trainer-side format scorer ran before the actor update."
        )

    log_prob = no_padding_2_padding(model_output["log_probs"], data)
    response_mask = data["response_mask"]
    format_advantages = data["format_reward_advantages"]
    if format_advantages.shape != response_mask.shape:
        raise ValueError(
            "format_reward_advantages must match response_mask, got "
            f"{tuple(format_advantages.shape)} and {tuple(response_mask.shape)}."
        )

    for key, value in config.global_batch_info.items():
        loss_config.global_batch_info[key] = value
    policy_loss_fn = get_policy_loss_fn(loss_config.policy_loss_mode)
    format_reward_loss, pg_metrics = policy_loss_fn(
        old_log_prob=data["old_log_probs"],
        log_prob=log_prob,
        advantages=format_advantages,
        response_mask=response_mask,
        loss_agg_mode=config.loss_agg_mode,
        config=loss_config,
        rollout_is_weights=data.get("rollout_is_weights", None),
    )
    metrics = {
        f"format_reward/{name[len('actor/') :]}": Metric(value=value, aggregation=AggregationType.MEAN)
        for name, value in pg_metrics.items()
    }
    return format_reward_loss, metrics


def distillation_loss(
    config: ActorConfig,
    distillation_config: DistillationConfig,
    model_output: dict,
    data: TensorDict,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    Compute the distillation loss and related metrics.

    Returns:
    - distillation_loss: Aggregated distillation loss scalar.
    - distillation_metrics: Dictionary of metrics.
    """
    assert distillation_config is not None
    loss_config: DistillationLossConfig = distillation_config.distillation_loss
    distillation_loss_fn = get_distillation_loss_fn(loss_config.loss_mode)
    distillation_losses, distillation_metrics = distillation_loss_fn(
        config=config,
        distillation_config=distillation_config,
        model_output=model_output,
        data=data,
    )
    response_mask = data["response_mask"]
    loss_agg_mode = config.loss_agg_mode

    distillation_losses, termination_metrics = mask_opd_termination_token(
        distillation_losses=distillation_losses,
        data=data,
        config=config,
        loss_config=loss_config,
    )
    distillation_metrics.update(termination_metrics)

    distillation_losses, visual_grounding_metrics = apply_teacher_counterfactual_visual_weighting(
        distillation_losses=distillation_losses,
        data=data,
        loss_config=loss_config,
    )
    distillation_metrics.update(visual_grounding_metrics)

    distillation_metrics.update(
        compute_distillation_loss_range(distillation_losses=distillation_losses, response_mask=response_mask)
    )
    if loss_config.loss_max_clamp is not None:
        # clamping min is for k1 loss which can be negative
        distillation_losses = distillation_losses.clamp(min=-loss_config.loss_max_clamp, max=loss_config.loss_max_clamp)

    distillation_losses, position_weight_metrics = apply_opd_position_weighting(
        distillation_losses=distillation_losses,
        response_mask=response_mask,
        loss_config=loss_config,
    )
    distillation_metrics.update(position_weight_metrics)

    token_va_weights, token_va_metrics = compute_token_level_va_group_weights(
        distillation_losses=distillation_losses,
        data=data,
        loss_config=loss_config,
    )
    distillation_metrics.update(token_va_metrics)

    student_visual_dependency_mask, student_visual_dependency_metrics = (
        compute_student_visual_dependency_mask(
            model_output=model_output,
            data=data,
            loss_config=loss_config,
        )
    )
    distillation_metrics.update(student_visual_dependency_metrics)
    if student_visual_dependency_mask is not None:
        # Preserve the original response-token denominator, as in VPPO-RL:
        # rejected tokens contribute exactly zero OPD gradient rather than
        # renormalizing the selected subset back to full strength.
        distillation_losses = distillation_losses * student_visual_dependency_mask.to(
            distillation_losses.dtype
        )

    # Top-k JSD may apply token-level truncated importance weights between the
    # vLLM rollout policy and actor policy without changing sampled-token OPD.
    if loss_config.loss_mode == "jsd_topk":
        rollout_is_weights = data.get("rollout_is_weights", None)
        if rollout_is_weights is not None:
            if rollout_is_weights.shape != distillation_losses.shape:
                raise ValueError(
                    "JSD top-k rollout IS weights must match token losses, got "
                    f"{tuple(rollout_is_weights.shape)} and {tuple(distillation_losses.shape)}."
                )
            rollout_is_weights = rollout_is_weights.detach().to(distillation_losses.dtype)
            distillation_losses = distillation_losses * rollout_is_weights
            valid_weights = rollout_is_weights[response_mask.bool()]
            distillation_metrics["distillation/jsd_topk_rollout_is_mean"] = Metric(
                AggregationType.MEAN,
                valid_weights.mean(),
            )

    if loss_config.use_policy_gradient:
        # Use negative distillation loss as reward, as done by https://thinkingmachines.ai/blog/on-policy-distillation/.
        policy_loss_fn = get_policy_loss_fn(loss_config.policy_loss_mode)
        for k, v in config.global_batch_info.items():
            loss_config.global_batch_info[k] = v
        log_prob = no_padding_2_padding(model_output["log_probs"], data)
        old_log_prob = data["old_log_probs"]
        rollout_is_weights = data.get("rollout_is_weights", None)
        advantages = -distillation_losses.detach()
        effective_loss_agg_mode = loss_agg_mode
        if token_va_weights is not None:
            advantages = advantages * token_va_weights
            effective_loss_agg_mode = "seq-mean-token-sum"
        distillation_loss, pg_metrics = policy_loss_fn(
            old_log_prob=old_log_prob,
            log_prob=log_prob,
            advantages=advantages,
            response_mask=response_mask,
            loss_agg_mode=effective_loss_agg_mode,
            config=loss_config,
            rollout_is_weights=rollout_is_weights,
        )
        pg_metrics = {f"distillation/{k[len('actor/') :]}": v for k, v in pg_metrics.items()}
        distillation_metrics.update(pg_metrics)
    else:
        # Directly backpropagate distillation loss as a supervised loss, as in https://arxiv.org/abs/2306.13649.
        effective_loss_agg_mode = loss_agg_mode
        if token_va_weights is not None:
            distillation_losses = distillation_losses * token_va_weights
            effective_loss_agg_mode = "seq-mean-token-sum"
        distillation_loss = agg_loss(
            loss_mat=distillation_losses,
            loss_mask=response_mask,
            loss_agg_mode=effective_loss_agg_mode,
            **config.global_batch_info,
        )

    return distillation_loss, distillation_metrics


@register_distillation_loss(DistillationLossSettings(names=["forward_kl_topk"], use_topk=True))  # type: ignore[arg-type]
def compute_forward_kl_topk(
    config: ActorConfig,
    distillation_config: DistillationConfig,
    model_output: dict,
    data: TensorDict,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Compute forward KL distillation loss and related metrics using top-k log probabilities.

    Returns:
    - distillation_losses: (bsz, resp_len)
    - distillation_metrics: Dictionary of metrics.
    """
    # topk loss has been computed in logits processor
    distillation_losses = no_padding_2_padding(model_output["distillation_losses"], data)
    student_mass = no_padding_2_padding(model_output["student_mass"], data)
    teacher_mass = no_padding_2_padding(model_output["teacher_mass"], data)
    response_mask_bool = data["response_mask"].bool()
    assert distillation_losses.shape == student_mass.shape == teacher_mass.shape == response_mask_bool.shape

    # Log amount of mass in the top-k log probabilities for both student and teacher.
    student_mass = student_mass[response_mask_bool]
    teacher_mass = teacher_mass[response_mask_bool]
    distillation_metrics = {
        "distillation/student_mass": student_mass.mean().item(),
        "distillation/student_mass_min": Metric(AggregationType.MIN, student_mass.min()),
        "distillation/student_mass_max": Metric(AggregationType.MAX, student_mass.max()),
        "distillation/teacher_mass": teacher_mass.mean().item(),
        "distillation/teacher_mass_min": Metric(AggregationType.MIN, teacher_mass.min()),
        "distillation/teacher_mass_max": Metric(AggregationType.MAX, teacher_mass.max()),
    }

    # Due to use of top-k, student and teacher distributions don't sum to 1 -> divergences can be negative.
    distillation_losses = distillation_losses.clamp_min(0.0)

    return distillation_losses, distillation_metrics


@register_distillation_loss(DistillationLossSettings(names=["jsd_topk"], use_topk=True))  # type: ignore[arg-type]
def compute_jsd_topk(
    config: ActorConfig,
    distillation_config: DistillationConfig,
    model_output: dict,
    data: TensorDict,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Consume the directly backpropagated per-token generalized JSD."""
    del distillation_config
    distillation_losses = no_padding_2_padding(model_output["distillation_losses"], data)
    student_mass = no_padding_2_padding(model_output["student_mass"], data)
    teacher_mass = no_padding_2_padding(model_output["teacher_mass"], data)
    teacher_kl = no_padding_2_padding(model_output["jsd_teacher_kl"], data)
    student_kl = no_padding_2_padding(model_output["jsd_student_kl"], data)
    response_mask = data["response_mask"].bool()
    expected_shape = response_mask.shape
    for name, value in {
        "loss": distillation_losses,
        "student_mass": student_mass,
        "teacher_mass": teacher_mass,
        "teacher_kl": teacher_kl,
        "student_kl": student_kl,
    }.items():
        if value.shape != expected_shape:
            raise ValueError(
                f"JSD top-k {name} shape {tuple(value.shape)} does not match "
                f"response mask {tuple(expected_shape)}."
            )

    metrics = {
        "distillation/jsd_topk": Metric(
            AggregationType.MEAN,
            distillation_losses[response_mask].mean(),
        ),
        "distillation/jsd_teacher_kl": Metric(
            AggregationType.MEAN,
            teacher_kl[response_mask].mean(),
        ),
        "distillation/jsd_student_kl": Metric(
            AggregationType.MEAN,
            student_kl[response_mask].mean(),
        ),
        "distillation/student_mass": student_mass[response_mask].mean().item(),
        "distillation/teacher_mass": teacher_mass[response_mask].mean().item(),
    }
    return distillation_losses.clamp_min(0.0), metrics


@register_distillation_loss(
    DistillationLossSettings(names=["kl", "k1", "abs", "mse", "k2", "low_var_kl", "k3"], use_estimator=True)
)  # type: ignore[arg-type]
def compute_distillation_loss_reverse_kl_estimator(
    config: ActorConfig,
    distillation_config: DistillationConfig,
    model_output,
    data: TensorDict,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    Compute the distillation loss and related metrics using single-sample KL estimators.

    Uses the kl_penalty function from core_algos which supports various KL divergence
    estimators: "kl", "k1", "abs", "mse", "k2", "low_var_kl", "k3".

    Returns:
    - distillation_losses: (bsz, resp_len)
    - distillation_metrics: Dictionary of metrics.
    """
    student_log_probs = no_padding_2_padding(model_output["log_probs"], data)
    teacher_log_probs = no_padding_2_padding(data["teacher_logprobs"], data).squeeze(-1)
    response_mask_bool = data["response_mask"].bool()
    assert teacher_log_probs.shape == student_log_probs.shape == response_mask_bool.shape

    loss_config: DistillationLossConfig = distillation_config.distillation_loss
    vgd_metrics = {}
    if loss_config.vgd_enabled:
        if "teacher_negative_logprobs" not in data:
            raise KeyError(
                "vgd_enabled=True requires teacher_negative_logprobs; ensure VGD negative-image "
                "teacher scoring is enabled in the rollout path."
            )
        teacher_negative_log_probs = no_padding_2_padding(
            data["teacher_negative_logprobs"], data
        ).squeeze(-1)
        if teacher_negative_log_probs.shape != teacher_log_probs.shape:
            raise ValueError("VGD teacher log-probability tensors must match the response layout.")
        visual_gap = (teacher_log_probs - teacher_negative_log_probs).detach()
        if loss_config.vgd_positive_only:
            clipped_visual_gap = visual_gap.clamp(min=0.0, max=loss_config.vgd_gap_clip)
        else:
            clipped_visual_gap = visual_gap.clamp(
                min=-loss_config.vgd_gap_clip,
                max=loss_config.vgd_gap_clip,
            )
        teacher_log_probs = (
            teacher_log_probs + loss_config.vgd_alpha * clipped_visual_gap
        ).detach()
        valid_gap = visual_gap[response_mask_bool]
        valid_clipped_gap = clipped_visual_gap[response_mask_bool]
        zero = student_log_probs.sum() * 0.0
        vgd_metrics = {
            "distillation/vgd_visual_gap": Metric(
                AggregationType.MEAN, valid_gap.mean() if valid_gap.numel() else zero
            ),
            "distillation/vgd_visual_gap_abs": Metric(
                AggregationType.MEAN, valid_gap.abs().mean() if valid_gap.numel() else zero
            ),
            "distillation/vgd_positive_gap_fraction": Metric(
                AggregationType.MEAN,
                (valid_gap > 0).float().mean() if valid_gap.numel() else zero,
            ),
            "distillation/vgd_clipped_fraction": Metric(
                AggregationType.MEAN,
                (
                    (valid_gap > loss_config.vgd_gap_clip)
                    if loss_config.vgd_positive_only
                    else (valid_gap.abs() > loss_config.vgd_gap_clip)
                ).float().mean()
                if valid_gap.numel()
                else zero,
            ),
            "distillation/vgd_target_shift": Metric(
                AggregationType.MEAN,
                (loss_config.vgd_alpha * valid_clipped_gap).mean()
                if valid_clipped_gap.numel()
                else zero,
            ),
        }
    distillation_losses = kl_penalty(
        logprob=student_log_probs, ref_logprob=teacher_log_probs, kl_penalty=loss_config.loss_mode
    )
    # Since k1 can be negative, log the mean absolute loss.
    metrics = {
        "distillation/abs_loss": Metric(AggregationType.MEAN, distillation_losses[response_mask_bool].abs().mean()),
        **vgd_metrics,
    }
    return distillation_losses, metrics
