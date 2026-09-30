# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

import copy

import pytest
import torch
import torch.nn.functional as F
from PIL import Image

from verl.trainer.distillation.fsdp.losses import (
    _append_probability_tail,
    _memory_efficient_topk_log_probs,
    compute_jsd_topk,
    generalized_jsd_from_log_probs,
)
from verl.utils.dataset.rl_dataset import DEFAULT_STRUCTURED_VISUAL_OUTPUT_INSTRUCTION, RLHFDataset
from verl.workers.config import DistillationConfig, DistillationLossConfig


def test_jsd_topk_config_is_opt_in_and_validated():
    original = DistillationLossConfig(loss_mode="k3")
    assert original.jsd_beta == 0.5
    assert original.jsd_add_tail is True
    assert original.jsd_vocab_chunk_size == 4096

    config = DistillationLossConfig(
        loss_mode="jsd_topk",
        topk=100,
        use_policy_gradient=False,
    )
    assert config.loss_settings.use_topk

    with pytest.raises(ValueError, match="directly backpropagated"):
        DistillationLossConfig(loss_mode="jsd_topk", use_policy_gradient=True)
    with pytest.raises(ValueError, match="beta"):
        DistillationLossConfig(
            loss_mode="jsd_topk",
            use_policy_gradient=False,
            jsd_beta=1.0,
        )
    with pytest.raises(ValueError, match="chunk"):
        DistillationLossConfig(
            loss_mode="jsd_topk",
            use_policy_gradient=False,
            jsd_vocab_chunk_size=0,
        )


def test_sampled_token_and_jsd_topk_papo_modes_are_all_valid():
    sampled = DistillationLossConfig(loss_mode="k1", use_policy_gradient=True)
    jsd = DistillationLossConfig(loss_mode="jsd_topk", topk=100, use_policy_gradient=False)
    jsd_papo = DistillationLossConfig(
        loss_mode="jsd_topk",
        topk=100,
        use_policy_gradient=False,
        papo_enabled=True,
        papo_teacher_gating_enabled=True,
    )

    assert sampled.loss_settings.use_estimator
    assert jsd.loss_settings.use_topk
    assert jsd_papo.loss_settings.use_topk and jsd_papo.papo_enabled


def test_tail_bucket_normalizes_reduced_distribution():
    topk_log_probs = torch.log(torch.tensor([[0.4, 0.35], [0.1, 0.2]]))
    with_tail = _append_probability_tail(topk_log_probs)
    torch.testing.assert_close(with_tail.exp().sum(dim=-1), torch.ones(2))


def test_generalized_jsd_is_zero_for_equal_distributions_and_has_student_gradient():
    teacher = torch.log(torch.tensor([[0.6, 0.3, 0.1]]))
    equal_student = teacher.clone().requires_grad_(True)
    equal_jsd, _, _ = generalized_jsd_from_log_probs(equal_student, teacher, beta=0.5)
    torch.testing.assert_close(equal_jsd, torch.zeros_like(equal_jsd), atol=1e-7, rtol=0)

    student_logits = torch.tensor([[0.0, 1.0, -0.5]], requires_grad=True)
    student = F.log_softmax(student_logits, dim=-1)
    jsd, teacher_kl, student_kl = generalized_jsd_from_log_probs(student, teacher, beta=0.5)
    assert jsd.item() > 0
    assert teacher_kl.item() > 0
    assert student_kl.item() > 0
    jsd.sum().backward()
    assert student_logits.grad is not None
    assert student_logits.grad.abs().sum().item() > 0
    assert teacher.grad is None


def test_memory_efficient_topk_log_probs_matches_dense_forward_and_backward():
    torch.manual_seed(7)
    logits = torch.randn(2, 3, 11, dtype=torch.float32, requires_grad=True)
    reference_logits = logits.detach().clone().requires_grad_(True)
    topk_ids = torch.tensor(
        [
            [[0, 3, 8], [1, 5, 10], [2, 4, 7]],
            [[1, 2, 9], [0, 6, 8], [3, 5, 10]],
        ],
        dtype=torch.int64,
    )
    output_weights = torch.randn(2, 3, 3)

    actual = _memory_efficient_topk_log_probs(logits, topk_ids, chunk_size=4)
    expected = torch.gather(F.log_softmax(reference_logits, dim=-1), dim=-1, index=topk_ids)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)

    (actual * output_weights).sum().backward()
    (expected * output_weights).sum().backward()
    torch.testing.assert_close(logits.grad, reference_logits.grad, atol=1e-6, rtol=1e-6)


def test_memory_efficient_topk_log_probs_does_not_save_full_fp32_distribution():
    logits = torch.randn(1, 5, 17, dtype=torch.bfloat16, requires_grad=True)
    topk_ids = torch.tensor([[[0, 3], [1, 5], [2, 7], [4, 9], [6, 10]]])
    saved_tensors = []

    def pack(tensor):
        saved_tensors.append(tensor)
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
        output = _memory_efficient_topk_log_probs(logits, topk_ids, chunk_size=4)
        output.sum().backward()

    assert not any(tensor.shape == logits.shape and tensor.dtype == torch.float32 for tensor in saved_tensors)
    assert any(tensor.shape == logits.shape and tensor.dtype == torch.bfloat16 for tensor in saved_tensors)


def test_external_teacher_topk_jsd_returns_token_losses_and_backpropagates():
    total_tokens, vocab_size, topk = 3, 7, 3
    student_logits = torch.randn(1, total_tokens, vocab_size, requires_grad=True)
    teacher_ids_values = torch.tensor([[0, 2, 4], [1, 3, 5], [0, 1, 6]], dtype=torch.int64)
    teacher_probs_values = torch.tensor(
        [[0.50, 0.20, 0.10], [0.35, 0.25, 0.15], [0.40, 0.20, 0.10]],
        dtype=torch.float32,
    )
    offsets = torch.tensor([0, total_tokens], dtype=torch.int64)
    teacher_ids = torch.nested.nested_tensor_from_jagged(teacher_ids_values, offsets)
    teacher_log_probs = torch.nested.nested_tensor_from_jagged(teacher_probs_values.log(), offsets)
    config = DistillationConfig(
        distillation_loss=DistillationLossConfig(
            loss_mode="jsd_topk",
            topk=topk,
            use_policy_gradient=False,
        )
    )

    output = compute_jsd_topk(
        student_logits,
        teacher_log_probs,
        teacher_ids,
        config,
        data_format="thd",
    )

    assert set(output) == {
        "distillation_losses",
        "student_mass",
        "teacher_mass",
        "jsd_teacher_kl",
        "jsd_student_kl",
    }
    assert output["distillation_losses"].shape == (1, total_tokens)
    assert torch.all(output["distillation_losses"] >= 0)
    output["distillation_losses"].mean().backward()
    assert student_logits.grad is not None
    assert student_logits.grad.abs().sum().item() > 0


def test_dataset_builds_independent_student_and_teacher_image_messages():
    full_image = Image.new("RGB", (8, 8), color=(255, 0, 0))
    crop_image = Image.new("RGB", (4, 4), color=(0, 255, 0))
    row = {
        "data_source": "vision_opd_6k",
        "prompt": [{"role": "user", "content": "<image>\nWhat is inside the box?"}],
        "images": [full_image],
        "bbox_images": [crop_image],
        "extra_info": {},
    }
    dataset = RLHFDataset.__new__(RLHFDataset)
    dataset.dataframe = [copy.deepcopy(row)]
    dataset.prompt_key = "prompt"
    dataset.teacher_prompt_key = "teacher_prompt"
    dataset.image_key = "images"
    dataset.teacher_image_key = "bbox_images"
    dataset.video_key = "videos"
    dataset.processor = object()
    dataset.need_tools_kwargs = False

    output = dataset[0]
    student_image = output["raw_prompt"][0]["content"][0]["image"]
    teacher_image = output["teacher_raw_prompt"][0]["content"][0]["image"]

    assert student_image.size == (8, 8)
    assert teacher_image.size == (4, 4)
    assert student_image.getpixel((0, 0)) == (255, 0, 0)
    assert teacher_image.getpixel((0, 0)) == (0, 255, 0)
    # The original prompt text is shared, but neither image object is swapped in place.
    assert output["raw_prompt"][0]["content"][1] == output["teacher_raw_prompt"][0]["content"][1]


def test_structured_visual_output_is_appended_to_student_and_teacher_prompts_only():
    full_image = Image.new("RGB", (8, 8), color=(255, 0, 0))
    row = {
        "data_source": "visual_opd",
        "prompt": [{"role": "user", "content": "<image>\nStudent question"}],
        "teacher_prompt": [{"role": "user", "content": "<image>\nTeacher context"}],
        "images": [full_image],
        "extra_info": {},
    }
    dataset = RLHFDataset.__new__(RLHFDataset)
    dataset.dataframe = [copy.deepcopy(row)]
    dataset.prompt_key = "prompt"
    dataset.teacher_prompt_key = "teacher_prompt"
    dataset.image_key = "images"
    dataset.teacher_image_key = None
    dataset.video_key = "videos"
    dataset.processor = object()
    dataset.need_tools_kwargs = False
    dataset.structured_visual_output_enabled = True
    dataset.structured_visual_output_instruction = DEFAULT_STRUCTURED_VISUAL_OUTPUT_INSTRUCTION

    output = dataset[0]

    def text_content(messages):
        return "".join(
            part["text"]
            for part in messages[0]["content"]
            if part.get("type") == "text"
        )

    student_text = text_content(output["raw_prompt"])
    teacher_text = text_content(output["teacher_raw_prompt"])
    assert "Student question" in student_text
    assert "Teacher context" in teacher_text
    assert student_text.count(DEFAULT_STRUCTURED_VISUAL_OUTPUT_INSTRUCTION) == 1
    assert teacher_text.count(DEFAULT_STRUCTURED_VISUAL_OUTPUT_INSTRUCTION) == 1
    assert "<grounding>" in student_text
    assert "<reasoning>" in student_text
    assert "<answer>" in student_text
    assert row["prompt"][0]["content"] == "<image>\nStudent question"


def test_structured_visual_output_does_not_require_an_auxiliary_opd_loss():
    config = DistillationLossConfig(
        loss_mode="k1",
        use_policy_gradient=True,
        use_task_rewards=False,
        visual_grounding_enabled=False,
        token_level_va_enabled=False,
        vgd_enabled=False,
        counterfactual_residual_enabled=False,
        papo_enabled=False,
        format_reward_enabled=False,
    )

    assert config.loss_mode == "k1"
    assert config.use_policy_gradient is True
    assert config.loss_settings.use_topk is False
