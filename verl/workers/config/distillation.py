# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

import logging
import os
from dataclasses import dataclass, field
from typing import Optional

from verl.base_config import BaseConfig

from .rollout import RolloutConfig

__all__ = ["DistillationLossConfig", "DistillationTeacherModelConfig", "DistillationConfig"]

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


@dataclass
class DistillationLossConfig(BaseConfig):
    """Configuration for distillation loss settings.

    loss_mode (str):
        Distillation loss function to use.
    topk (int, optional):
        Number of top tokens to consider for top-k distillation losses.
    use_task_rewards (bool):
        Whether to include task rewards alongside distillation loss.
    distillation_loss_coef (float):
        Coefficient for distillation loss when combined with task rewards.
    loss_max_clamp (float, optional):
        Maximum value to clamp distillation loss. If None, no clamping is applied.
    log_prob_min_clamp (float, optional):
        Minimum value to clamp log probabilities for stability, e.g., log q - log p where p or q are
        very close to zero. If None, no clamping is applied.
    use_policy_gradient (bool):
        Whether to incorporate distillation loss as a reward, as done
        by https://thinkingmachines.ai/blog/on-policy-distillation/. Recommended to use loss_mode=k1.
        Otherwise, distillation loss is directly backpropagated as a supervised loss,
        as in https://arxiv.org/abs/2306.13649. Recommended to use loss_mode=k3 or forward_kl_topk.
    policy_loss_mode (str):
        Name of the policy loss to use when use_policy_gradient is true.
    clip_ratio (float):
        PPO clipping ratio for policy loss.
    clip_ratio_low (float):
        Lower bound for PPO clipping ratio.
    clip_ratio_high (float):
        Upper bound for PPO clipping ratio.
    mask_termination_token (bool):
        Whether to zero the OPD contribution of the model EOS token. For Qwen
        chat tokenizers this is ``<|im_end|>``. Other policy objectives retain
        their original response masks.
    loss_settings (DistillationLossSettings, optional):
        Runtime-populated settings based on loss_mode. Not set by user.
    """

    loss_mode: str = "k3"
    topk: Optional[int] = 128
    use_task_rewards: bool = True
    distillation_loss_coef: float = 1.0
    loss_max_clamp: Optional[float] = 10.0
    log_prob_min_clamp: Optional[float] = -10.0

    use_policy_gradient: bool = True
    policy_loss_mode: str = "vanilla"
    clip_ratio: float = 0.2
    clip_ratio_low: float = 0.2
    clip_ratio_high: float = 0.2

    # Optional SimpleOPD-style termination-token masking. For Qwen chat
    # tokenizers, <|im_end|> is the model EOS token. When enabled, its OPD
    # delta is zeroed while the task loss and student-reference KL keep their
    # original response masks.
    mask_termination_token: bool = False

    # Optional trajectory-position weighting for the main OPD objective. The
    # unnormalized weight at relative response position r in [0, 1] is
    #   1 + alpha * exp(-r / decay).
    # Weights are normalized independently within every response to have mean
    # one, then projected onto [min_weight, max_weight] while preserving that
    # mean. This changes where OPD credit is allocated without changing its
    # per-response coefficient scale. Additive objectives such as task PPO,
    # student-reference KL, format reward, residual matching, and PAPO are not
    # position weighted.
    opd_position_weighting_enabled: bool = False
    opd_position_weight_alpha: float = 0.5
    opd_position_weight_decay: float = 0.25
    opd_position_weight_min: float = 0.5
    opd_position_weight_max: float = 1.5

    # VPPO-style Token Gradient Filtering (TGF) for the main OPD objective.
    # The student scores the sampled response with the original image and a
    # randomly masked image. Per-token visual dependency is the low-variance
    # KL estimate between those two policies; only the top fraction within
    # each response contributes OPD gradients. Other additive objectives keep
    # their original masks.
    student_visual_dependency_filter_enabled: bool = False
    student_visual_dependency_top_fraction: float = 0.4
    student_visual_dependency_mask_ratio: float = 0.5
    student_visual_dependency_patch_size: int = 14

    # Optional teacher-only counterfactual visual grounding. When enabled, the
    # teacher also scores the sampled response with a blank image. The existing
    # distillation loss is then upweighted where the real image increases the
    # teacher probability of the sampled token. Defaults preserve the original
    # training path exactly.
    visual_grounding_enabled: bool = False
    visual_grounding_negative_strategy: str = "blank"
    visual_grounding_blank_value: int = 127
    visual_grounding_weight_scale: float = 1.0
    visual_grounding_weight_threshold: float = 0.0
    visual_grounding_max_weight: float = 5.0

    # Token-level component of VA-OPD (arXiv:2605.21924). The teacher scores
    # the sampled response once more with a pixelated image. Tokens are ranked
    # by relu(log p_teacher(real) - log p_teacher(pixelated)) within each
    # rollout, then the high- and low-VA groups are normalized independently.
    # This is disabled by default and deliberately excludes the paper's
    # rollout-level reweighting component. It may be combined with PAPO: in
    # that case VA-OPD uses a pixelated teacher view while PAPO uses its own
    # random-mask/patch-shuffle teacher view.
    token_level_va_enabled: bool = False
    token_level_va_negative_strategy: str = "pixelate"
    token_level_va_pixelation_ratio: float = 0.1
    token_level_va_high_fraction: float = 0.2
    token_level_va_high_weight: float = 0.5

    # Visual-Grounding Distillation (VGD): shift the sampled-token teacher
    # target by the teacher's original-versus-destroyed-image log-probability
    # gap, then use the standard k1 policy-gradient OPD path.
    vgd_enabled: bool = False
    vgd_alpha: float = 1.0
    vgd_gap_clip: float = 1.0
    # Keep only original-image-positive teacher gaps when shaping the target.
    # This prevents counterfactual noise from lowering the clean-image target.
    vgd_positive_only: bool = False
    vgd_negative_strategy: str = "patch_shuffle"
    # None derives patch_size * merge_size from the active image processor.
    vgd_patch_shuffle_block_size: Optional[int] = None
    vgd_random_mask_ratio: float = 0.6
    vgd_random_mask_patch_size: int = 14

    # Generic top-k generalized JSD. The reduced support is the teacher's top-k
    # vocabulary plus an optional exact tail bucket.
    jsd_beta: float = 0.5
    jsd_add_tail: bool = True
    # FP32 vocab chunks avoid a full [tokens, vocab] FP32 log-softmax tensor.
    jsd_vocab_chunk_size: int = 4096

    # Optional sampled-token counterfactual residual matching. Teacher and
    # student score the same rollout under the original image and a PAPO-style
    # randomly masked image. By default, all sampled response tokens are
    # weighted equally. Direction-aware mode keeps only tokens for which the
    # original image raises the teacher log-probability relative to masking.
    counterfactual_residual_enabled: bool = False
    counterfactual_residual_negative_strategy: str = "random_mask"
    counterfactual_residual_mask_ratio: float = 0.6
    counterfactual_residual_patch_size: int = 14
    counterfactual_residual_coef: float = 0.1
    counterfactual_residual_beta: float = 0.1
    counterfactual_residual_direction_aware: bool = False

    # Optional positional decomposition for the combined PAPO + residual
    # experiment. PAPO is applied only to the leading fraction of each valid
    # response, while residual matching is applied only to the remainder.
    # Each region is renormalized to preserve the original coefficient scale.
    papo_residual_position_split_enabled: bool = False
    papo_residual_papo_fraction: float = 0.5

    # Optional PAPO-only positional ablation. PAPO is applied only to the
    # leading fraction of each valid response, without enabling residual
    # matching. The selected region is renormalized to preserve coefficient
    # scale relative to full-token PAPO.
    papo_front_only_enabled: bool = False
    papo_front_fraction: float = 0.5

    # Optional PAPO objective (arXiv:2507.06448) on top of vanilla OPD. The
    # student scores its sampled response under the original and a randomly
    # masked image. We maximize the sampled-token low-variance estimate of
    # KL[pi_student(original) || pi_student(masked)]. The masked log-probability
    # is detached, matching the efficient path in the official implementation.
    # By default, each token is gated and scaled by
    # relu(log pi_teacher(original) - log pi_teacher(masked)). Disable teacher
    # gating to apply the unweighted PAPO loss to every selected response token.
    # ``student_positive`` uses only the detached binary student contrast
    # log p_student(original) - log p_student(masked) > 0 and gives each
    # selected token unit weight. ``rate_matched_random`` preserves, within
    # each response, both the number and values of positive teacher-gap weights but assigns them to uniformly
    # sampled valid positions. This isolates teacher-token alignment from the
    # sparsity and scale of the gate.
    # The PAPO denominator is the number of tokens active under the chosen mode.
    # PAPO may be combined with VGD. In that mode the student PAPO branch and
    # teacher VGD branch share one negative image selected by the PAPO negative
    # strategy settings, and PAPO reuses VGD's teacher-negative log-probability.
    # PAPO may also be combined with counterfactual residual matching; those
    # objectives share the same randomly masked image.
    papo_enabled: bool = False
    papo_teacher_gating_enabled: bool = True
    papo_teacher_gate_mode: str = "teacher_positive"
    papo_negative_strategy: str = "random_mask"
    # None derives patch_size * merge_size from the active image processor.
    papo_patch_shuffle_block_size: Optional[int] = None
    papo_mask_ratio: float = 0.6
    papo_patch_size: int = 14
    papo_coef: float = 0.02
    papo_log_ratio_clip: float = 20.0
    papo_kl_max: float = 10.0

    # Meaning-preserving near-neighbour consistency. The student scores its
    # sampled response under the clean image and an additive-Gaussian-noise
    # image. The Gaussian branch is detached and the sampled-token k3 estimate
    # of KL[pi(clean) || pi(gaussian)] is minimized. This is independent of
    # PAPO, so near-only and joint push-pull ablations share the same code path.
    gaussian_near_pull_enabled: bool = False
    gaussian_near_pull_std: float = 0.2
    gaussian_near_pull_coef: float = 0.02
    gaussian_near_pull_log_ratio_clip: float = 20.0
    gaussian_near_pull_kl_max: float = 10.0

    # Optional format-only policy-gradient objective. It is additive and does
    # not enable or reuse task-reward PPO, so vanilla OPD remains unchanged
    # when disabled. The baseline gives malformed/truncated responses a
    # negative advantage instead of merely withholding a positive reward.
    format_reward_enabled: bool = False
    format_reward_coef: float = 0.1
    format_reward_baseline: float = 0.5
    format_reward_style: str = "boxed"
    format_reward_require_termination: bool = True

    # Store global batch info for loss aggregation:
    # dp_size: data parallel size
    # batch_num_tokens: number of valid tokens in global batch
    # global_batch_size: global batch size
    global_batch_info: dict = field(default_factory=dict)

    # Store distillation loss settings for computing the specified loss_mode
    # Not set by user, populated at runtime
    loss_settings: Optional[dict] = None

    def __post_init__(self):
        self._mutable_fields.add("loss_settings")
        from verl.trainer.distillation.losses import DistillationLossSettings, get_distillation_loss_settings

        self.loss_settings: DistillationLossSettings = get_distillation_loss_settings(self.loss_mode)

        if self.policy_loss_mode != "vanilla":
            raise NotImplementedError(
                f"Only vanilla policy loss is currently supported when use_policy_gradient is True, "
                f"but got {self.policy_loss_mode}."
            )

        if self.use_policy_gradient and self.loss_mode == "forward_kl_topk":
            print(
                "WARNING: forward_kl_topk is most effective as a supervised distillation loss "
                "(use_policy_gradient=False). With policy gradient, the update uses only the sampled"
                " token's logprob ∇logπ(a), so the top-k distributional signal (how non-sampled logits "
                "should move) is largely unused."
            )

        if not self.use_policy_gradient and self.loss_mode == "k1":
            raise ValueError(
                "Directly backpropagating k1 loss is incorrect since gradient of k1 loss"
                " wrt model weights does not depend on teacher log probabilities."
            )

        if self.opd_position_weight_alpha < 0:
            raise ValueError("opd_position_weight_alpha must be non-negative.")
        if self.opd_position_weight_decay <= 0:
            raise ValueError("opd_position_weight_decay must be positive.")
        if self.opd_position_weight_min <= 0:
            raise ValueError("opd_position_weight_min must be positive.")
        if self.opd_position_weight_min > 1:
            raise ValueError("opd_position_weight_min must be at most 1 to preserve mean-one weights.")
        if self.opd_position_weight_max < 1:
            raise ValueError("opd_position_weight_max must be at least 1 to preserve mean-one weights.")
        if self.opd_position_weight_min > self.opd_position_weight_max:
            raise ValueError("opd_position_weight_min cannot exceed opd_position_weight_max.")

        if self.student_visual_dependency_filter_enabled:
            if not 0.0 < self.student_visual_dependency_top_fraction <= 1.0:
                raise ValueError("student_visual_dependency_top_fraction must be in (0, 1].")
            if not 0.0 <= self.student_visual_dependency_mask_ratio <= 1.0:
                raise ValueError("student_visual_dependency_mask_ratio must be in [0, 1].")
            if self.student_visual_dependency_patch_size <= 0:
                raise ValueError("student_visual_dependency_patch_size must be positive.")
            incompatible_modes = {
                "vgd_enabled": self.vgd_enabled,
                "counterfactual_residual_enabled": self.counterfactual_residual_enabled,
                "papo_enabled": self.papo_enabled,
            }
            enabled_incompatible = [name for name, enabled in incompatible_modes.items() if enabled]
            if enabled_incompatible:
                raise ValueError(
                    "student_visual_dependency_filter_enabled is a standalone ablation and cannot be combined "
                    "with " + ", ".join(enabled_incompatible) + "."
                )

        if self.visual_grounding_enabled:
            if not self.loss_settings.use_estimator:
                raise ValueError(
                    "Teacher-only counterfactual visual grounding currently supports sampled-token "
                    "distillation losses only; top-k teacher token sets are not aligned across images."
                )
            if self.visual_grounding_negative_strategy != "blank":
                raise ValueError(
                    "Only visual_grounding_negative_strategy='blank' is currently supported, "
                    f"got {self.visual_grounding_negative_strategy!r}."
                )
            if not 0 <= self.visual_grounding_blank_value <= 255:
                raise ValueError("visual_grounding_blank_value must be between 0 and 255.")
            if self.visual_grounding_weight_scale < 0:
                raise ValueError("visual_grounding_weight_scale must be non-negative.")
            if self.visual_grounding_weight_threshold < 0:
                raise ValueError("visual_grounding_weight_threshold must be non-negative.")
            if self.visual_grounding_max_weight < 1:
                raise ValueError("visual_grounding_max_weight must be at least 1.")

        if self.token_level_va_enabled:
            if self.visual_grounding_enabled or self.counterfactual_residual_enabled:
                raise ValueError(
                    "token_level_va_enabled is an independent experimental mode and cannot be combined "
                    "with visual_grounding_enabled or counterfactual_residual_enabled."
                )
            if not self.loss_settings.use_estimator:
                raise ValueError(
                    "Token-level VA-OPD supports sampled-token distillation losses only; "
                    "top-k teacher token sets are not aligned across the two images."
                )
            if self.token_level_va_negative_strategy != "pixelate":
                raise ValueError(
                    "Only token_level_va_negative_strategy='pixelate' is supported, "
                    f"got {self.token_level_va_negative_strategy!r}."
                )
            if not 0 < self.token_level_va_pixelation_ratio <= 1:
                raise ValueError("token_level_va_pixelation_ratio must be in (0, 1].")
            if not 0 < self.token_level_va_high_fraction < 1:
                raise ValueError("token_level_va_high_fraction must be in (0, 1).")
            if not 0 < self.token_level_va_high_weight < 1:
                raise ValueError("token_level_va_high_weight must be in (0, 1).")

        if self.vgd_enabled:
            incompatible_modes = {
                "visual_grounding_enabled": self.visual_grounding_enabled,
                "token_level_va_enabled": self.token_level_va_enabled,
                "counterfactual_residual_enabled": self.counterfactual_residual_enabled,
            }
            enabled_incompatible = [name for name, enabled in incompatible_modes.items() if enabled]
            if enabled_incompatible:
                raise ValueError(
                    "vgd_enabled cannot be combined with "
                    + ", ".join(enabled_incompatible)
                    + "."
                )
            if self.loss_mode != "k1" or not self.use_policy_gradient:
                raise ValueError("VGD requires loss_mode='k1' and use_policy_gradient=True.")
            if self.vgd_alpha < 0:
                raise ValueError("vgd_alpha must be non-negative.")
            if self.vgd_gap_clip <= 0:
                raise ValueError("vgd_gap_clip must be positive.")
            if self.vgd_negative_strategy not in {"random_mask", "patch_shuffle"}:
                raise ValueError(
                    "vgd_negative_strategy must be 'random_mask' or 'patch_shuffle'."
                )
            if (
                self.vgd_patch_shuffle_block_size is not None
                and self.vgd_patch_shuffle_block_size <= 0
            ):
                raise ValueError("vgd_patch_shuffle_block_size must be positive when set.")
            if not 0.0 <= self.vgd_random_mask_ratio <= 1.0:
                raise ValueError("vgd_random_mask_ratio must be in [0, 1].")
            if self.vgd_random_mask_patch_size <= 0:
                raise ValueError("vgd_random_mask_patch_size must be positive.")

        if self.loss_mode == "jsd_topk":
            if self.use_policy_gradient:
                raise ValueError(
                    "Top-k JSD is a directly backpropagated distribution loss; "
                    "set use_policy_gradient=False."
                )
            if self.topk is None or self.topk <= 0:
                raise ValueError("jsd_topk requires a positive distillation topk value.")
            if not 0.0 < self.jsd_beta < 1.0:
                raise ValueError("jsd_beta must be in (0, 1).")
            if self.jsd_vocab_chunk_size <= 0:
                raise ValueError("jsd_vocab_chunk_size must be positive.")
            if (
                self.visual_grounding_enabled
                or self.token_level_va_enabled
                or self.vgd_enabled
                or self.counterfactual_residual_enabled
            ):
                raise ValueError(
                    "jsd_topk cannot be combined with visual-grounding, token-level VA-OPD, "
                    "VGD, or counterfactual residual modes."
                )

        if self.counterfactual_residual_enabled:
            if self.visual_grounding_enabled:
                raise ValueError(
                    "counterfactual_residual_enabled and visual_grounding_enabled are separate experimental "
                    "modes and cannot be enabled together."
                )
            if not self.loss_settings.use_estimator:
                raise ValueError(
                    "Counterfactual residual matching currently supports sampled-token distillation losses only."
                )
            if self.counterfactual_residual_negative_strategy != "random_mask":
                raise ValueError(
                    "Only counterfactual_residual_negative_strategy='random_mask' is currently supported."
                )
            if not 0.0 <= self.counterfactual_residual_mask_ratio <= 1.0:
                raise ValueError("counterfactual_residual_mask_ratio must be in [0, 1].")
            if self.counterfactual_residual_patch_size <= 0:
                raise ValueError("counterfactual_residual_patch_size must be positive.")
            if self.counterfactual_residual_coef < 0:
                raise ValueError("counterfactual_residual_coef must be non-negative.")
            if self.counterfactual_residual_beta <= 0:
                raise ValueError("counterfactual_residual_beta must be positive.")

        if self.papo_enabled:
            if self.visual_grounding_enabled:
                raise ValueError(
                    "papo_enabled is an independent counterfactual objective and cannot be combined "
                    "with visual_grounding_enabled."
                )
            if not (self.loss_settings.use_estimator or self.loss_mode == "jsd_topk"):
                raise ValueError("PAPO supports sampled-token distillation losses or jsd_topk.")
            if self.papo_negative_strategy not in {"random_mask", "patch_shuffle"}:
                raise ValueError(
                    "papo_negative_strategy must be 'random_mask' or 'patch_shuffle'."
                )
            if self.papo_teacher_gate_mode not in {
                "teacher_positive",
                "student_positive",
                "rate_matched_random",
            }:
                raise ValueError(
                    "papo_teacher_gate_mode must be 'teacher_positive', "
                    "'student_positive', or 'rate_matched_random'."
                )
            if (
                self.papo_patch_shuffle_block_size is not None
                and self.papo_patch_shuffle_block_size <= 0
            ):
                raise ValueError("papo_patch_shuffle_block_size must be positive when set.")
            if not 0.0 <= self.papo_mask_ratio <= 1.0:
                raise ValueError("papo_mask_ratio must be in [0, 1].")
            if self.papo_patch_size <= 0:
                raise ValueError("papo_patch_size must be positive.")
            if self.papo_coef < 0:
                raise ValueError("papo_coef must be non-negative.")
            if self.papo_log_ratio_clip <= 0:
                raise ValueError("papo_log_ratio_clip must be positive.")
            if self.papo_kl_max <= 0:
                raise ValueError("papo_kl_max must be positive.")
            if self.counterfactual_residual_enabled:
                if self.papo_negative_strategy != "random_mask":
                    raise ValueError(
                        "Combined PAPO and counterfactual residual matching currently require "
                        "papo_negative_strategy='random_mask'."
                    )
                if self.papo_mask_ratio != self.counterfactual_residual_mask_ratio:
                    raise ValueError(
                        "Combined PAPO and counterfactual residual matching share one masked image, so "
                        "papo_mask_ratio must equal counterfactual_residual_mask_ratio."
                    )
                if self.papo_patch_size != self.counterfactual_residual_patch_size:
                    raise ValueError(
                        "Combined PAPO and counterfactual residual matching share one masked image, so "
                        "papo_patch_size must equal counterfactual_residual_patch_size."
                    )

        if self.gaussian_near_pull_enabled:
            if not self.loss_settings.use_estimator:
                raise ValueError(
                    "Gaussian near pull currently supports sampled-token distillation losses only."
                )
            if self.gaussian_near_pull_std < 0:
                raise ValueError("gaussian_near_pull_std must be non-negative.")
            if self.gaussian_near_pull_coef < 0:
                raise ValueError("gaussian_near_pull_coef must be non-negative.")
            if self.gaussian_near_pull_log_ratio_clip <= 0:
                raise ValueError("gaussian_near_pull_log_ratio_clip must be positive.")
            if self.gaussian_near_pull_kl_max <= 0:
                raise ValueError("gaussian_near_pull_kl_max must be positive.")

        if self.papo_residual_position_split_enabled:
            if not (self.papo_enabled and self.counterfactual_residual_enabled):
                raise ValueError(
                    "papo_residual_position_split_enabled requires both papo_enabled and "
                    "counterfactual_residual_enabled."
                )
            if not 0.0 < self.papo_residual_papo_fraction < 1.0:
                raise ValueError("papo_residual_papo_fraction must be in (0, 1).")

        if self.papo_front_only_enabled:
            if not self.papo_enabled:
                raise ValueError("papo_front_only_enabled requires papo_enabled=True.")
            if self.papo_residual_position_split_enabled:
                raise ValueError(
                    "papo_front_only_enabled cannot be combined with "
                    "papo_residual_position_split_enabled."
                )
            if not 0.0 < self.papo_front_fraction < 1.0:
                raise ValueError("papo_front_fraction must be in (0, 1).")

        if self.format_reward_coef < 0:
            raise ValueError("format_reward_coef must be non-negative.")
        if not 0 <= self.format_reward_baseline <= 1:
            raise ValueError("format_reward_baseline must be between 0 and 1.")
        if self.format_reward_style not in {"boxed", "think_boxed"}:
            raise ValueError(
                "format_reward_style must be 'boxed' or 'think_boxed', "
                f"got {self.format_reward_style!r}."
            )


@dataclass
class DistillationTeacherModelConfig(BaseConfig):
    """Configuration for on-policy distillation teacher.

    enable_resource_pool (bool):
        Whether to enable separate resource pool for teacher model(s).
    n_gpus_per_node (int):
        Number of GPUs per node to use for distillation teacher model(s).
    nnodes (int):
        Number of nodes to use for distillation teacher model(s).
    model_path (str, optional):
        Model path for the teacher model. Can be a local path or a Hugging Face model
    inference (RolloutConfig):
        Rollout configuration for the teacher model inference during distillation.
    """

    _mutable_fields = BaseConfig._mutable_fields

    enable_resource_pool: bool = False
    n_gpus_per_node: int = 0
    nnodes: int = 0
    model_path: Optional[str] = None
    inference: RolloutConfig = field(default_factory=RolloutConfig)


@dataclass
class DistillationConfig(BaseConfig):
    """Configuration for on-policy distillation.

    enabled (bool):
        Whether on-policy distillation is enabled.
    num_workers (int):
        Number of teacher model replicas.
    teacher_prompt_key (str, optional):
        Column name in dataset for teacher-only prompt (e.g. q+c). If null, teacher falls back to student prompt.
    teacher_image_key (str, optional):
        Column name containing privileged teacher images. When set, the teacher
        uses these images while the student continues to use data.image_key.
    student_noisy_teacher_clean_enabled (bool):
        Whether training-time student rollout and forward use a Gaussian-noisy
        image while the teacher keeps the original clean image.
    student_noisy_teacher_clean_std (float):
        Gaussian standard deviation in RGB pixel space normalized to [0, 1].
    teacher_model (TeacherModelConfig):
        Configuration for the teacher model used for distillation.
    distillation_loss (DistillationLossConfig):
        Configuration for distillation loss settings.
    """

    _mutable_fields = BaseConfig._mutable_fields

    enabled: bool = False
    num_workers: int = 8
    teacher_prompt_key: Optional[str] = None
    teacher_image_key: Optional[str] = None
    student_noisy_teacher_clean_enabled: bool = False
    student_noisy_teacher_clean_std: float = 0.05
    teacher_model: DistillationTeacherModelConfig = field(default_factory=DistillationTeacherModelConfig)
    distillation_loss: DistillationLossConfig = field(default_factory=DistillationLossConfig)

    def __post_init__(self):
        if self.student_noisy_teacher_clean_enabled:
            if not self.enabled:
                raise ValueError("student_noisy_teacher_clean_enabled requires distillation.enabled=True.")
            if self.student_noisy_teacher_clean_std < 0:
                raise ValueError("student_noisy_teacher_clean_std must be non-negative.")

        # Prompt + Response from student are fed into teacher as context
        max_model_len = self.teacher_model.inference.max_model_len
        max_num_batched_tokens = self.teacher_model.inference.max_num_batched_tokens
        student_prompt_length = self.teacher_model.inference.prompt_length
        student_response_length = self.teacher_model.inference.response_length
        if self.enabled:
            required_context_len = student_prompt_length + student_response_length + 1
            if max_model_len is not None and required_context_len > max_model_len:
                raise ValueError(
                    "Distillation teacher inference requires room for the student prompt, the full student "
                    f"response, and one generated token, but got {student_prompt_length=}, "
                    f"{student_response_length=}, {required_context_len=}, {max_model_len=}."
                )
            if max_num_batched_tokens is not None and required_context_len > max_num_batched_tokens:
                raise ValueError(
                    "Distillation teacher inference requires room for the student prompt, the full student "
                    f"response, and one generated token within the engine batching budget, but got "
                    f"{student_prompt_length=}, {student_response_length=}, {required_context_len=}, "
                    f"{max_num_batched_tokens=}."
                )

        self.teacher_model.inference.prompt_length = (
            self.teacher_model.inference.prompt_length + self.teacher_model.inference.response_length
        )
        self.teacher_model.inference.response_length = 1

        # Ensure max log probs is aligned with top-k
        engine_name = self.teacher_model.inference.name
        engine_kwargs = self.teacher_model.inference.engine_kwargs
        if not self.distillation_loss.loss_settings.use_topk or self.distillation_loss.topk is None or not self.enabled:
            return
        match engine_name:
            case "vllm":
                vllm_engine_kwargs = dict(engine_kwargs.get("vllm", {}))
                max_logprobs = vllm_engine_kwargs.get("max_logprobs")
                if max_logprobs is None:
                    vllm_engine_kwargs["max_logprobs"] = self.distillation_loss.topk
                    max_logprobs = self.distillation_loss.topk
                if max_logprobs < self.distillation_loss.topk:
                    raise ValueError(
                        f"VLLM max_logprobs ({max_logprobs}) must be >= distillation_loss topk "
                        f"({self.distillation_loss.topk}) to enable distillation loss computation."
                    )
                engine_kwargs["vllm"] = vllm_engine_kwargs
            case _:
                raise NotImplementedError(
                    f"DistillationTeacherModelConfig does not support inference engine {engine_name}"
                )
