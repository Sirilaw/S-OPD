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


import torch
import torch.nn.functional as F

from verl.utils.ulysses import (
    get_ulysses_sequence_parallel_world_size,
    slice_input_tensor,
)
from verl.workers.config import DistillationConfig, DistillationLossConfig


class _MemoryEfficientTopKLogProbs(torch.autograd.Function):
    """Gather top-k log-probs without materializing full-vocabulary FP32 log-probs."""

    @staticmethod
    def forward(ctx, logits: torch.Tensor, topk_ids: torch.Tensor, chunk_size: int) -> torch.Tensor:
        topk_ids = topk_ids.long()
        log_partition = torch.full(
            (*logits.shape[:-1], 1),
            -torch.inf,
            dtype=torch.float32,
            device=logits.device,
        )
        for logits_chunk in logits.split(chunk_size, dim=-1):
            chunk_log_partition = torch.logsumexp(logits_chunk.float(), dim=-1, keepdim=True)
            log_partition = torch.logaddexp(log_partition, chunk_log_partition)

        topk_logits = torch.gather(logits, dim=-1, index=topk_ids).float()
        ctx.save_for_backward(logits, topk_ids, log_partition)
        ctx.chunk_size = chunk_size
        return topk_logits - log_partition

    @staticmethod
    def backward(ctx, grad_topk_log_probs: torch.Tensor):
        logits, topk_ids, log_partition = ctx.saved_tensors
        grad_topk_log_probs = grad_topk_log_probs.float()
        grad_log_partition = grad_topk_log_probs.sum(dim=-1, keepdim=True)
        grad_logits = torch.empty_like(logits)

        for start in range(0, logits.shape[-1], ctx.chunk_size):
            end = min(start + ctx.chunk_size, logits.shape[-1])
            probabilities = torch.exp(logits[..., start:end].float() - log_partition)
            grad_logits[..., start:end] = (-grad_log_partition * probabilities).to(logits.dtype)

        grad_logits.scatter_add_(
            dim=-1,
            index=topk_ids,
            src=grad_topk_log_probs.to(logits.dtype),
        )
        return grad_logits, None, None


def _memory_efficient_topk_log_probs(
    logits: torch.Tensor,
    topk_ids: torch.Tensor,
    *,
    chunk_size: int,
) -> torch.Tensor:
    """Compute FP32-normalized top-k log-probs with bounded temporary memory."""
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}.")
    return _MemoryEfficientTopKLogProbs.apply(logits, topk_ids, chunk_size)


def kl_divergence(log_q: torch.Tensor, log_p: torch.Tensor) -> torch.Tensor:
    """Compute KL divergence between two distributions given their log probabilities."""
    log_p = log_p.float()
    log_q = log_q.float()
    p = log_p.exp()
    kld = p * (log_p - log_q)
    return kld.sum(dim=-1)


def _append_probability_tail(log_probs: torch.Tensor) -> torch.Tensor:
    """Append log(1 - top-k mass) as a numerically stable tail bucket."""
    log_mass = torch.logsumexp(log_probs.float(), dim=-1, keepdim=True)
    log_mass = log_mass.clamp(max=-1e-7)
    tail_log_prob = torch.log(-torch.expm1(log_mass))
    return torch.cat([log_probs.float(), tail_log_prob], dim=-1)


def generalized_jsd_from_log_probs(
    student_log_probs: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    *,
    beta: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute beta KL(T||M) + (1-beta) KL(S||M) per token."""
    if student_log_probs.shape != teacher_log_probs.shape:
        raise ValueError(
            "Student and teacher reduced distributions must have identical shapes, got "
            f"{tuple(student_log_probs.shape)} and {tuple(teacher_log_probs.shape)}."
        )
    if not 0.0 < beta < 1.0:
        raise ValueError(f"beta must be in (0, 1), got {beta}.")

    student_log_probs = student_log_probs.float()
    teacher_log_probs = teacher_log_probs.float()
    beta_tensor = torch.as_tensor(beta, device=student_log_probs.device, dtype=student_log_probs.dtype)
    mixture_log_probs = torch.logaddexp(
        teacher_log_probs + torch.log(beta_tensor),
        student_log_probs + torch.log1p(-beta_tensor),
    )
    teacher_component = (
        teacher_log_probs.exp() * (teacher_log_probs - mixture_log_probs)
    ).sum(dim=-1)
    student_component = (
        student_log_probs.exp() * (student_log_probs - mixture_log_probs)
    ).sum(dim=-1)
    jsd = beta_tensor * teacher_component + (1.0 - beta_tensor) * student_component
    return jsd, teacher_component, student_component


def compute_jsd_topk(
    student_logits: torch.Tensor,
    teacher_topk_log_probs: torch.Tensor,
    teacher_topk_ids: torch.Tensor,
    config: DistillationConfig,
    data_format: str,
) -> dict[str, torch.Tensor]:
    """Generalized top-k JSD on teacher support with an optional tail.

    The teacher supplies its top-k token ids and log-probabilities. We gather
    differentiable student probabilities at the same ids. Adding the tail
    bucket makes both reduced distributions sum to one.
    """
    del data_format
    assert teacher_topk_log_probs.is_nested and teacher_topk_ids.is_nested
    teacher_topk_log_probs = teacher_topk_log_probs.values().unsqueeze(0)
    teacher_topk_ids = teacher_topk_ids.values().unsqueeze(0)

    if get_ulysses_sequence_parallel_world_size() > 1:
        teacher_topk_log_probs = slice_input_tensor(teacher_topk_log_probs, dim=1)
        teacher_topk_ids = slice_input_tensor(teacher_topk_ids, dim=1)
    if teacher_topk_log_probs.shape[:2] != student_logits.shape[:2]:
        raise ValueError(
            "Teacher top-k tensors do not align with student logits: "
            f"teacher={tuple(teacher_topk_log_probs.shape)}, student={tuple(student_logits.shape)}."
        )
    if teacher_topk_ids.numel() and (
        teacher_topk_ids.min().item() < 0 or teacher_topk_ids.max().item() >= student_logits.shape[-1]
    ):
        raise ValueError(
            "JSD top-k teacher and student must use compatible vocabularies: "
            f"teacher token id range=[{teacher_topk_ids.min().item()}, {teacher_topk_ids.max().item()}], "
            f"student vocab size={student_logits.shape[-1]}."
        )

    loss_config: DistillationLossConfig = config.distillation_loss
    student_topk_log_probs = _memory_efficient_topk_log_probs(
        student_logits,
        teacher_topk_ids,
        chunk_size=loss_config.jsd_vocab_chunk_size,
    )
    teacher_topk_log_probs = teacher_topk_log_probs.float()
    student_mass = student_topk_log_probs.exp().sum(dim=-1)
    teacher_mass = teacher_topk_log_probs.exp().sum(dim=-1)

    if loss_config.jsd_add_tail:
        student_distill_log_probs = _append_probability_tail(student_topk_log_probs)
        teacher_distill_log_probs = _append_probability_tail(teacher_topk_log_probs)
    else:
        student_distill_log_probs = student_topk_log_probs - torch.logsumexp(
            student_topk_log_probs, dim=-1, keepdim=True
        )
        teacher_distill_log_probs = teacher_topk_log_probs - torch.logsumexp(
            teacher_topk_log_probs, dim=-1, keepdim=True
        )

    jsd, teacher_component, student_component = generalized_jsd_from_log_probs(
        student_distill_log_probs,
        teacher_distill_log_probs,
        beta=loss_config.jsd_beta,
    )
    return {
        "distillation_losses": jsd.clamp_min(0.0),
        "student_mass": student_mass,
        "teacher_mass": teacher_mass,
        "jsd_teacher_kl": teacher_component,
        "jsd_student_kl": student_component,
    }


def compute_forward_kl_topk(
    student_logits: torch.Tensor,
    teacher_topk_log_probs: torch.Tensor,
    teacher_topk_ids: torch.Tensor,
    config: DistillationConfig,
    data_format: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute forward KL distillation loss using top-k log probabilities.

    Args:
        student_logits: (bsz, seqlen/sp_size, vocab_size).
        teacher_topk_log_probs: (bsz, seqlen, topk).
        teacher_topk_ids: (bsz, seqlen, topk).
        data_format: "thd" or "bshd", models not support THD format, e.g GPT-OSS, Qwen3.5

    Returns:
    - distillation_losses: (bsz, seqlen/sp_size)
    - student_mass: (bsz, seqlen/sp_size)
    - teacher_mass: (bsz, seqlen/sp_size)
    """
    assert teacher_topk_log_probs.is_nested and teacher_topk_ids.is_nested
    teacher_topk_log_probs = teacher_topk_log_probs.values().unsqueeze(0)  # (1, total_nnz, topk)
    teacher_topk_ids = teacher_topk_ids.values().unsqueeze(0)  # (1, total_nnz, topk)

    # 1. split across sp groups (bsz, seqlen, topk) => (bsz, seqlen/sp_size, topk)
    if get_ulysses_sequence_parallel_world_size() > 1:
        teacher_topk_log_probs = slice_input_tensor(teacher_topk_log_probs, dim=1)
        teacher_topk_ids = slice_input_tensor(teacher_topk_ids, dim=1)
    assert teacher_topk_log_probs.shape[:2] == teacher_topk_ids.shape[:2] == student_logits.shape[:2]

    # 2. compute token-wise KL divergence across sp groups
    student_log_probs = F.log_softmax(student_logits, dim=-1)
    student_topk_log_probs = torch.gather(student_log_probs, dim=-1, index=teacher_topk_ids)
    student_mass = student_topk_log_probs.exp().sum(dim=-1)
    teacher_mass = teacher_topk_log_probs.exp().sum(dim=-1)
    loss_config: DistillationLossConfig = config.distillation_loss
    if loss_config.log_prob_min_clamp is not None:
        student_topk_log_probs = student_topk_log_probs.clamp_min(loss_config.log_prob_min_clamp)
        teacher_topk_log_probs = teacher_topk_log_probs.clamp_min(loss_config.log_prob_min_clamp)
    distillation_losses = kl_divergence(log_q=student_topk_log_probs, log_p=teacher_topk_log_probs)

    return {"distillation_losses": distillation_losses, "student_mass": student_mass, "teacher_mass": teacher_mass}
