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
import asyncio
from typing import Any, Optional
from uuid import uuid4

import numpy as np
import ray
import torch
from omegaconf import DictConfig
from PIL import Image
from tensordict import TensorDict
from torch.nn import functional as F

from verl.experimental.agent_loop import AsyncLLMServerManager
from verl.protocol import DataProto
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.tokenizer import normalize_token_ids
from verl.workers.config import DistillationConfig, DistillationLossConfig


def build_blank_counterfactual_multi_modal_data(
    multi_modal_data: Optional[dict[str, Any]], blank_value: int
) -> Optional[dict[str, Any]]:
    """Return a non-mutating, same-shape blank-image counterfactual.

    ``None`` means the sample has no image, so callers can reuse the positive
    teacher scores instead of issuing a duplicate request. Videos are preserved
    unchanged; the first implementation intentionally targets image grounding.
    """
    if not multi_modal_data:
        return None
    images = multi_modal_data.get("images")
    if not images:
        return None

    blank_images = []
    for image in images:
        if not isinstance(image, Image.Image):
            raise TypeError(
                "Teacher counterfactual visual grounding expects PIL images, "
                f"got {type(image)!r}."
            )
        # RGB avoids mode-specific fill semantics (for example palette indices)
        # while retaining the exact spatial resolution expected by the processor.
        blank_images.append(Image.new("RGB", image.size, color=(blank_value,) * 3))

    counterfactual = dict(multi_modal_data)
    counterfactual["images"] = blank_images
    return counterfactual


def build_pixelated_counterfactual_multi_modal_data(
    multi_modal_data: Optional[dict[str, Any]], pixelation_ratio: float
) -> Optional[dict[str, Any]]:
    """Destroy fine visual detail without changing the processor image size.

    This follows the VA-OPD counterfactual: bilinearly downsample both spatial
    dimensions, then restore the original size with nearest-neighbor
    interpolation. The input dictionary and PIL images are never mutated.
    ``None`` lets callers reuse the positive scores for samples without images.
    """
    if not 0 < pixelation_ratio <= 1:
        raise ValueError("pixelation_ratio must be in (0, 1].")
    if not multi_modal_data:
        return None
    images = multi_modal_data.get("images")
    if not images:
        return None

    pixelated_images = []
    for image in images:
        if not isinstance(image, Image.Image):
            raise TypeError(f"Token-level VA-OPD expects PIL images, got {type(image)!r}.")
        image = image.convert("RGB")
        width, height = image.size
        reduced_size = (
            max(1, round(width * pixelation_ratio)),
            max(1, round(height * pixelation_ratio)),
        )
        degraded = image.resize(reduced_size, resample=Image.Resampling.BILINEAR)
        degraded = degraded.resize(image.size, resample=Image.Resampling.NEAREST)
        pixelated_images.append(degraded)

    counterfactual = dict(multi_modal_data)
    counterfactual["images"] = pixelated_images
    return counterfactual


def build_random_mask_counterfactual_multi_modal_data(
    multi_modal_data: Optional[dict[str, Any]],
    mask_ratio: float = 0.6,
    patch_size: int = 14,
    *,
    generator: Optional[torch.Generator] = None,
) -> Optional[dict[str, Any]]:
    """Randomly black out independent image patches without mutating the input.

    This follows PAPO's random degradation: split each image into a grid of
    ``patch_size`` square pixel patches and independently blacken every patch
    with probability ``mask_ratio``. Partial patches at the right and bottom
    image boundaries are included in the same way.
    """
    if not 0.0 <= mask_ratio <= 1.0:
        raise ValueError("mask_ratio must be in [0, 1].")
    if patch_size <= 0:
        raise ValueError("patch_size must be positive.")
    if not multi_modal_data:
        return None
    images = multi_modal_data.get("images")
    if not images:
        return None

    masked_images = []
    for image in images:
        if not isinstance(image, Image.Image):
            raise TypeError(f"Counterfactual residual matching expects PIL images, got {type(image)!r}.")
        image = image.convert("RGB")
        width, height = image.size
        grid_width = (width + patch_size - 1) // patch_size
        grid_height = (height + patch_size - 1) // patch_size
        patch_mask = (
            torch.rand((grid_height, grid_width), generator=generator)
            .lt(mask_ratio)
            .mul(255)
            .to(torch.uint8)
        )
        pixel_mask = Image.fromarray(patch_mask.numpy()).resize(
            (grid_width * patch_size, grid_height * patch_size),
            resample=Image.Resampling.NEAREST,
        )
        pixel_mask = pixel_mask.crop((0, 0, width, height))
        masked_images.append(Image.composite(Image.new("RGB", image.size), image, pixel_mask))

    counterfactual = dict(multi_modal_data)
    counterfactual["images"] = masked_images
    return counterfactual


def build_gaussian_near_multi_modal_data(
    multi_modal_data: Optional[dict[str, Any]],
    std: float = 0.2,
    *,
    generator: Optional[torch.Generator] = None,
) -> Optional[dict[str, Any]]:
    """Add RGB Gaussian noise in normalized pixel space without mutating input.

    The perturbation is sampled independently for every pixel/channel, clipped
    to ``[0, 1]``, and converted back to uint8.  It is intended as a
    meaning-preserving near neighbour of the clean image.
    """
    if std < 0:
        raise ValueError("std must be non-negative.")
    if not multi_modal_data:
        return None
    images = multi_modal_data.get("images")
    if not images:
        return None

    noisy_images = []
    for image in images:
        if not isinstance(image, Image.Image):
            raise TypeError(f"Gaussian near pull expects PIL images, got {type(image)!r}.")
        clean = torch.from_numpy(
            np.array(image.convert("RGB"), dtype=np.float32, copy=True)
        ).div_(255.0)
        noise = torch.randn(clean.shape, generator=generator, dtype=clean.dtype)
        noisy = (clean + noise * std).clamp_(0.0, 1.0).mul_(255.0).round_().to(torch.uint8)
        noisy_images.append(Image.fromarray(noisy.numpy(), mode="RGB"))

    near = dict(multi_modal_data)
    near["images"] = noisy_images
    return near


def build_patch_shuffle_counterfactual_multi_modal_data(
    multi_modal_data: Optional[dict[str, Any]],
    block_size: int,
    *,
    generator: Optional[torch.Generator] = None,
) -> Optional[dict[str, Any]]:
    """Shuffle full image blocks while leaving incomplete boundaries intact."""
    if block_size <= 0:
        raise ValueError("block_size must be positive.")
    if not multi_modal_data:
        return None
    images = multi_modal_data.get("images")
    if not images:
        return None

    shuffled_images = []
    for image in images:
        if not isinstance(image, Image.Image):
            raise TypeError(f"PAPO patch shuffle expects PIL images, got {type(image)!r}.")
        image = image.convert("RGB")
        width, height = image.size
        grid_width = width // block_size
        grid_height = height // block_size
        block_count = grid_width * grid_height

        if block_count < 2:
            noise = torch.randint(
                0, 256, (height, width, 3), dtype=torch.uint8, generator=generator
            )
            shuffled_images.append(Image.fromarray(noise.numpy(), mode="RGB"))
            continue

        blocks = [
            image.crop(
                (
                    x * block_size,
                    y * block_size,
                    (x + 1) * block_size,
                    (y + 1) * block_size,
                )
            )
            for y in range(grid_height)
            for x in range(grid_width)
        ]
        permutation = torch.randperm(block_count, generator=generator)
        if torch.equal(permutation, torch.arange(block_count)):
            permutation = permutation.roll(1)

        shuffled = image.copy()
        for destination, source in enumerate(permutation.tolist()):
            x = destination % grid_width
            y = destination // grid_width
            shuffled.paste(blocks[source], (x * block_size, y * block_size))
        shuffled_images.append(shuffled)

    counterfactual = dict(multi_modal_data)
    counterfactual["images"] = shuffled_images
    return counterfactual


def build_mismatched_counterfactual_multi_modal_data(
    reference_multi_modal_data: Optional[dict[str, Any]],
    donor_multi_modal_data: Optional[dict[str, Any]],
) -> Optional[dict[str, Any]]:
    """Replace reference images with real images from another sample.

    Donor images are resized to the corresponding reference image size. This
    keeps the processor's image grid (and therefore the text/image token
    alignment) unchanged while changing only visual content. Videos are left
    untouched because this objective currently targets image grounding.
    """
    if not reference_multi_modal_data or not donor_multi_modal_data:
        return None
    reference_images = reference_multi_modal_data.get("images")
    donor_images = donor_multi_modal_data.get("images")
    if not reference_images or not donor_images:
        return None

    mismatched_images = []
    for image_index, reference_image in enumerate(reference_images):
        donor_image = donor_images[image_index % len(donor_images)]
        if not isinstance(reference_image, Image.Image) or not isinstance(donor_image, Image.Image):
            raise TypeError("Counterfactual residual matching expects PIL images.")
        donor_image = donor_image.convert("RGB")
        if donor_image.size != reference_image.size:
            donor_image = donor_image.resize(reference_image.size, resample=Image.Resampling.BICUBIC)
        mismatched_images.append(donor_image)

    counterfactual = dict(reference_multi_modal_data)
    counterfactual["images"] = mismatched_images
    return counterfactual


def _get_teacher_sampling_params(
    distillation_config: DistillationConfig,
    distillation_loss_config: DistillationLossConfig,
    num_logprobs_override: Optional[int] = None,
) -> dict[str, Any]:
    """Get sampling parameters for teacher model when computing log probabilities for distillation."""
    if distillation_config.teacher_model.inference.temperature != 1.0:
        raise NotImplementedError("vLLM does not support temperature for prompt_logprobs.")

    num_logprobs = (
        num_logprobs_override
        if num_logprobs_override is not None
        else distillation_loss_config.topk
        if distillation_loss_config.loss_settings.use_topk
        else 0
    )
    return {
        "max_tokens": 1,
        "temperature": distillation_config.teacher_model.inference.temperature,
        "prompt_logprobs": num_logprobs,
    }


def _pad_teacher_outputs(
    teacher_ids: torch.Tensor,
    teacher_logprobs: torch.Tensor,
    prompt_width: int,
    response_width: int,
    prompt_length: int,
    response_length: int,
    pad_token_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    # TODO(wuxibin): remove padding and use tensordict.
    left_pad_size = prompt_width - prompt_length
    right_pad_size = response_width - response_length
    padding = (0, 0, left_pad_size, right_pad_size)
    return (
        F.pad(teacher_ids, padding, value=pad_token_id).unsqueeze(0),
        F.pad(teacher_logprobs, padding, value=0.0).unsqueeze(0),
    )


def _unpad_teacher_inputs(data: DataProto) -> tuple[list[int], int, int]:
    """Unpad valid sequence ids and prompt/response lengths from a single sample.
    The sample is a left-padded prompt concatenated with a right-padded response.
    TODO(wuxibin): remove padding and use tensordict.
    """
    assert len(data) == 1, "Teacher logprob computation expects a single sample"

    input_ids = data.batch["input_ids"][0]
    attention_mask = data.batch["attention_mask"][0]
    prompt_width = data.batch["prompts"][0].shape[0]
    response_width = data.batch["responses"][0].shape[0]
    assert attention_mask.shape[0] == prompt_width + response_width, (
        "attention_mask sequence length must match prompt and response widths"
    )
    valid_prompt_length = int(attention_mask[:prompt_width].sum().item())
    valid_response_length = int(attention_mask[-response_width:].sum().item())
    prompt_num_padding = prompt_width - valid_prompt_length
    sequence_ids = input_ids[prompt_num_padding : prompt_width + valid_response_length]
    sequence_ids = normalize_token_ids(sequence_ids)
    return sequence_ids, valid_prompt_length, valid_response_length


def _align_teacher_outputs_to_student_layout(
    teacher_ids: torch.Tensor,
    teacher_logprobs: torch.Tensor,
    student_prompt_length: int,
    response_length: int,
    pad_token_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Project teacher outputs to student's padded layout.

    Teacher prompt can differ from student prompt (q+c vs q). We only need response-token supervision to align
    with the student's trajectory. This helper keeps response tokens intact and trims/pads teacher prompt segment
    to student_prompt_length so downstream tensors keep the expected shape.
    """
    if teacher_ids.dim() == 1:
        teacher_ids = teacher_ids.unsqueeze(-1)
    if teacher_logprobs.dim() == 1:
        teacher_logprobs = teacher_logprobs.unsqueeze(-1)

    total_len = int(teacher_ids.shape[0])
    if response_length > total_len:
        raise ValueError(
            f"response_length ({response_length}) cannot exceed teacher output length ({total_len})"
        )

    teacher_prompt_len = total_len - response_length
    teacher_prompt_ids = teacher_ids[:teacher_prompt_len]
    teacher_prompt_lps = teacher_logprobs[:teacher_prompt_len]
    teacher_response_ids = teacher_ids[teacher_prompt_len:]
    teacher_response_lps = teacher_logprobs[teacher_prompt_len:]

    if teacher_prompt_len >= student_prompt_length:
        teacher_prompt_ids = teacher_prompt_ids[-student_prompt_length:]
        teacher_prompt_lps = teacher_prompt_lps[-student_prompt_length:]
    else:
        left_pad = student_prompt_length - teacher_prompt_len
        teacher_prompt_ids = F.pad(teacher_prompt_ids, (0, 0, left_pad, 0), value=pad_token_id)
        teacher_prompt_lps = F.pad(teacher_prompt_lps, (0, 0, left_pad, 0), value=0.0)

    return (
        torch.cat([teacher_prompt_ids, teacher_response_ids], dim=0),
        torch.cat([teacher_prompt_lps, teacher_response_lps], dim=0),
    )


class AsyncTeacherLLMServerManager(AsyncLLMServerManager):
    """Teacher-specific async client used for distillation logprob computation."""

    def __init__(
        self,
        config: DictConfig,
        servers: list[tuple[str, ray.actor.ActorHandle]],
        load_balancer_handle: ray.actor.ActorHandle,
        distillation_config: DictConfig | DistillationConfig,
        pad_token_id: int,
    ):
        super().__init__(config=config, servers=servers, load_balancer_handle=load_balancer_handle)
        if isinstance(distillation_config, DistillationConfig):
            self.distillation_config = distillation_config
        else:
            self.distillation_config: DistillationConfig = omega_conf_to_dataclass(distillation_config)
        self.distillation_loss_config: DistillationLossConfig = self.distillation_config.distillation_loss
        self.pad_token_id = pad_token_id

    # async def compute_teacher_logprobs_single(
    #     self,
    #     sequence_ids: list[int],
    #     multi_modal_data: Optional[dict[str, Any]] = None,
    # ) -> tuple[torch.Tensor, torch.Tensor]:
    #     """Compute teacher log probabilities for a single unpadded sequence."""
    #     multi_modal_data = multi_modal_data or {}
    #     teacher_output = await self.generate(
    #         request_id=uuid4().hex,
    #         prompt_ids=sequence_ids,
    #         sampling_params=_get_teacher_sampling_params(self.distillation_config, self.distillation_loss_config),
    #         image_data=multi_modal_data.get("images"),
    #         video_data=multi_modal_data.get("videos"),
    #     )
    #     # Shapes: # S, (1 or K), where S is the response length, K is either 1 or topk depending on
    #     # the distillation loss settings.
    #     teacher_ids = torch.tensor(teacher_output.extra_fields["prompt_ids"], dtype=torch.int32)
    #     teacher_logprobs = torch.tensor(teacher_output.extra_fields["prompt_logprobs"])
    #     assert teacher_ids.shape[0] == teacher_logprobs.shape[0] == len(sequence_ids)
    #     return teacher_ids, teacher_logprobs
    async def compute_teacher_logprobs_single(
        self,
        sequence_ids: list[int],
        multi_modal_data: Optional[dict[str, Any]] = None,
        expected_len: Optional[int] = None,
        num_logprobs_override: Optional[int] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute teacher log probabilities for a single unpadded sequence."""
        multi_modal_data = multi_modal_data or {}
        teacher_output = await self.generate(
            request_id=uuid4().hex,
            prompt_ids=sequence_ids,
            sampling_params=_get_teacher_sampling_params(
                self.distillation_config,
                self.distillation_loss_config,
                num_logprobs_override=num_logprobs_override,
            ),
            image_data=multi_modal_data.get("images"),
            video_data=multi_modal_data.get("videos"),
        )
        teacher_ids = torch.tensor(teacher_output.extra_fields["prompt_ids"], dtype=torch.int32)
        teacher_logprobs = torch.tensor(teacher_output.extra_fields["prompt_logprobs"])

        # Validate with expected_len (length after processor expansion)
        # If not provided, fall back to sequence_ids length (compatible with non-video scenarios)
        check_len = expected_len if expected_len is not None else len(sequence_ids)
        assert teacher_ids.shape[0] == teacher_logprobs.shape[0] == check_len, (
            f"Length mismatch: teacher_ids={teacher_ids.shape[0]}, "
            f"teacher_logprobs={teacher_logprobs.shape[0]}, "
            f"expected={check_len}, sequence_ids={len(sequence_ids)}"
        )
        return teacher_ids, teacher_logprobs

    async def compute_teacher_logprobs_batch(self, data: DataProto) -> DataProto:
        """Compute teacher log probabilities for a batch of prompt-response pairs."""
        multi_modal_data_batch = data.non_tensor_batch.get("teacher_multi_modal_data")
        counterfactual_multi_modal_data_batch = data.non_tensor_batch.get(
            "teacher_counterfactual_multi_modal_data"
        )
        teacher_sequence_ids_batch = data.non_tensor_batch.get("teacher_sequence_ids")
        tasks = []
        negative_tasks = []
        papo_counterfactual_tasks = []
        sampled_positive_tasks = []
        lengths = []
        use_explicit_teacher_sequences = []
        prompt_width = data.batch["prompts"].shape[1]
        response_width = data.batch["responses"].shape[1]
        papo_teacher_contrast_enabled = (
            self.distillation_loss_config.papo_enabled
            and self.distillation_loss_config.papo_teacher_gating_enabled
            and self.distillation_loss_config.papo_teacher_gate_mode
            != "student_positive"
        )
        topk_papo_gate = (
            self.distillation_loss_config.loss_settings.use_topk
            and papo_teacher_contrast_enabled
        )

        # Compute logprobs for each sample in the batch
        for i in range(len(data)):
            item = data[i : i + 1]
            default_sequence_ids, prompt_length, response_length = _unpad_teacher_inputs(item)
            sequence_ids = default_sequence_ids
            use_explicit_sequence = False
            if teacher_sequence_ids_batch is not None and teacher_sequence_ids_batch[i] is not None:
                sequence_ids = normalize_token_ids(teacher_sequence_ids_batch[i])
                use_explicit_sequence = True
            multi_modal_data = None if multi_modal_data_batch is None else multi_modal_data_batch[i]
            sampled_logprob_override = {"num_logprobs_override": 0} if topk_papo_gate else {}
            lengths.append((prompt_length, response_length))
            use_explicit_teacher_sequences.append(use_explicit_sequence)
            tasks.append(
                asyncio.create_task(
                    self.compute_teacher_logprobs_single(
                        sequence_ids=sequence_ids,
                        expected_len=len(sequence_ids),
                        multi_modal_data=multi_modal_data,
                    )
                )
            )
            if topk_papo_gate:
                # The main JSD branch consumes top-k tensors, while PAPO's gate
                # needs the log-probability of the actually sampled token.
                sampled_positive_tasks.append(
                    asyncio.create_task(
                        self.compute_teacher_logprobs_single(
                            sequence_ids=sequence_ids,
                            expected_len=len(sequence_ids),
                            multi_modal_data=multi_modal_data,
                            num_logprobs_override=0,
                        )
                    )
                )
            if self.distillation_loss_config.visual_grounding_enabled:
                negative_multi_modal_data = build_blank_counterfactual_multi_modal_data(
                    multi_modal_data,
                    blank_value=self.distillation_loss_config.visual_grounding_blank_value,
                )
                if negative_multi_modal_data is None:
                    negative_tasks.append(None)
                else:
                    negative_tasks.append(
                        asyncio.create_task(
                            self.compute_teacher_logprobs_single(
                                sequence_ids=sequence_ids,
                                expected_len=len(sequence_ids),
                                multi_modal_data=negative_multi_modal_data,
                                **sampled_logprob_override,
                            )
                        )
                    )
            elif self.distillation_loss_config.token_level_va_enabled:
                negative_multi_modal_data = build_pixelated_counterfactual_multi_modal_data(
                    multi_modal_data,
                    pixelation_ratio=self.distillation_loss_config.token_level_va_pixelation_ratio,
                )
                if negative_multi_modal_data is None:
                    negative_tasks.append(None)
                else:
                    negative_tasks.append(
                        asyncio.create_task(
                            self.compute_teacher_logprobs_single(
                                sequence_ids=sequence_ids,
                                expected_len=len(sequence_ids),
                                multi_modal_data=negative_multi_modal_data,
                                **sampled_logprob_override,
                            )
                        )
                    )
            elif (
                self.distillation_loss_config.vgd_enabled
                or self.distillation_loss_config.counterfactual_residual_enabled
                or papo_teacher_contrast_enabled
            ):
                negative_multi_modal_data = (
                    None
                    if counterfactual_multi_modal_data_batch is None
                    else counterfactual_multi_modal_data_batch[i]
                )
                if negative_multi_modal_data is None:
                    negative_tasks.append(None)
                else:
                    negative_tasks.append(
                        asyncio.create_task(
                            self.compute_teacher_logprobs_single(
                                sequence_ids=sequence_ids,
                                expected_len=len(sequence_ids),
                                multi_modal_data=negative_multi_modal_data,
                                **sampled_logprob_override,
                            )
                        )
                    )

            # Token-level VA-OPD and PAPO require different teacher
            # counterfactuals. The primary negative above is VA-OPD's
            # pixelated image; score PAPO's random-mask/patch-shuffle image
            # separately so neither objective silently consumes the other's
            # visual intervention.
            if (
                self.distillation_loss_config.token_level_va_enabled
                and papo_teacher_contrast_enabled
            ):
                papo_multi_modal_data = (
                    None
                    if counterfactual_multi_modal_data_batch is None
                    else counterfactual_multi_modal_data_batch[i]
                )
                if papo_multi_modal_data is None:
                    papo_counterfactual_tasks.append(None)
                else:
                    papo_counterfactual_tasks.append(
                        asyncio.create_task(
                            self.compute_teacher_logprobs_single(
                                sequence_ids=sequence_ids,
                                expected_len=len(sequence_ids),
                                multi_modal_data=papo_multi_modal_data,
                                **sampled_logprob_override,
                            )
                        )
                    )

        visual_grounding_enabled = self.distillation_loss_config.visual_grounding_enabled
        token_level_va_enabled = self.distillation_loss_config.token_level_va_enabled
        vgd_enabled = self.distillation_loss_config.vgd_enabled
        residual_enabled = self.distillation_loss_config.counterfactual_residual_enabled
        separate_papo_counterfactual_enabled = (
            token_level_va_enabled and papo_teacher_contrast_enabled
        )
        counterfactual_scoring_enabled = (
            visual_grounding_enabled
            or token_level_va_enabled
            or vgd_enabled
            or residual_enabled
            or papo_teacher_contrast_enabled
        )
        if counterfactual_scoring_enabled:
            active_negative_tasks = [task for task in negative_tasks if task is not None]
            active_papo_counterfactual_tasks = [
                task for task in papo_counterfactual_tasks if task is not None
            ]
            all_outputs = await asyncio.gather(
                *tasks,
                *active_negative_tasks,
                *active_papo_counterfactual_tasks,
                *sampled_positive_tasks,
            )
            outputs = all_outputs[: len(tasks)]
            papo_counterfactual_start = len(tasks) + len(active_negative_tasks)
            sampled_positive_start = (
                papo_counterfactual_start + len(active_papo_counterfactual_tasks)
            )
            sampled_positive_outputs = (
                all_outputs[sampled_positive_start:] if topk_papo_gate else [None] * len(outputs)
            )
            negative_outputs = []
            negative_output_index = len(tasks)
            for sample_index, (positive_output, negative_task) in enumerate(
                zip(outputs, negative_tasks, strict=True)
            ):
                if negative_task is None:
                    negative_outputs.append(
                        sampled_positive_outputs[sample_index] if topk_papo_gate else positive_output
                    )
                else:
                    negative_outputs.append(all_outputs[negative_output_index])
                    negative_output_index += 1
            papo_counterfactual_outputs = []
            papo_counterfactual_output_index = papo_counterfactual_start
            for sample_index, papo_counterfactual_task in enumerate(papo_counterfactual_tasks):
                if papo_counterfactual_task is None:
                    papo_counterfactual_outputs.append(outputs[sample_index])
                else:
                    papo_counterfactual_outputs.append(
                        all_outputs[papo_counterfactual_output_index]
                    )
                    papo_counterfactual_output_index += 1
            if not separate_papo_counterfactual_enabled:
                papo_counterfactual_outputs = [None] * len(outputs)
        else:
            # Keep the original path untouched when the opt-in feature is off:
            # one teacher request and no counterfactual tensor processing.
            outputs = await asyncio.gather(*tasks)
            negative_outputs = [None] * len(outputs)
            papo_counterfactual_outputs = [None] * len(outputs)
            sampled_positive_outputs = [None] * len(outputs)

        # Pad the teacher logprobs and ids
        padded_teacher_ids = []
        padded_teacher_logprobs = []
        padded_teacher_negative_logprobs = []
        padded_teacher_counterfactual_logprobs = []
        padded_teacher_sampled_logprobs = []
        for (
            (teacher_ids, teacher_logprobs),
            negative_output,
            papo_counterfactual_output,
            sampled_positive_output,
            (prompt_length, response_length),
            use_explicit_sequence,
        ) in zip(
            outputs,
            negative_outputs,
            papo_counterfactual_outputs,
            sampled_positive_outputs,
            lengths,
            use_explicit_teacher_sequences,
            strict=True,
        ):
            if counterfactual_scoring_enabled:
                negative_teacher_ids, negative_teacher_logprobs = negative_output
                if not topk_papo_gate and not torch.equal(teacher_ids, negative_teacher_ids):
                    raise ValueError("Positive and counterfactual teacher token ids must be identical.")
            if separate_papo_counterfactual_enabled:
                (
                    papo_counterfactual_teacher_ids,
                    papo_counterfactual_teacher_logprobs,
                ) = papo_counterfactual_output
                if not torch.equal(teacher_ids, papo_counterfactual_teacher_ids):
                    raise ValueError(
                        "Positive and PAPO counterfactual teacher token ids must be identical."
                    )
            if topk_papo_gate:
                sampled_teacher_ids, sampled_teacher_logprobs = sampled_positive_output
            if use_explicit_sequence:
                teacher_ids, teacher_logprobs = _align_teacher_outputs_to_student_layout(
                    teacher_ids=teacher_ids,
                    teacher_logprobs=teacher_logprobs,
                    student_prompt_length=prompt_length,
                    response_length=response_length,
                    pad_token_id=self.pad_token_id,
                )
                if counterfactual_scoring_enabled:
                    _, negative_teacher_logprobs = _align_teacher_outputs_to_student_layout(
                        teacher_ids=negative_teacher_ids,
                        teacher_logprobs=negative_teacher_logprobs,
                        student_prompt_length=prompt_length,
                        response_length=response_length,
                        pad_token_id=self.pad_token_id,
                    )
                if separate_papo_counterfactual_enabled:
                    _, papo_counterfactual_teacher_logprobs = (
                        _align_teacher_outputs_to_student_layout(
                            teacher_ids=papo_counterfactual_teacher_ids,
                            teacher_logprobs=papo_counterfactual_teacher_logprobs,
                            student_prompt_length=prompt_length,
                            response_length=response_length,
                            pad_token_id=self.pad_token_id,
                        )
                    )
                if topk_papo_gate:
                    sampled_teacher_ids, sampled_teacher_logprobs = _align_teacher_outputs_to_student_layout(
                        teacher_ids=sampled_teacher_ids,
                        teacher_logprobs=sampled_teacher_logprobs,
                        student_prompt_length=prompt_length,
                        response_length=response_length,
                        pad_token_id=self.pad_token_id,
                    )
            padded_ids, padded_logprobs = _pad_teacher_outputs(
                teacher_ids,
                teacher_logprobs,
                prompt_width=prompt_width,
                response_width=response_width,
                prompt_length=prompt_length,
                response_length=response_length,
                pad_token_id=self.pad_token_id,
            )
            padded_teacher_ids.append(padded_ids)
            padded_teacher_logprobs.append(padded_logprobs)
            if counterfactual_scoring_enabled:
                _, padded_negative_logprobs = _pad_teacher_outputs(
                    negative_teacher_ids,
                    negative_teacher_logprobs,
                    prompt_width=prompt_width,
                    response_width=response_width,
                    prompt_length=prompt_length,
                    response_length=response_length,
                    pad_token_id=self.pad_token_id,
                )
                padded_teacher_negative_logprobs.append(padded_negative_logprobs)
            if separate_papo_counterfactual_enabled:
                _, padded_papo_counterfactual_logprobs = _pad_teacher_outputs(
                    papo_counterfactual_teacher_ids,
                    papo_counterfactual_teacher_logprobs,
                    prompt_width=prompt_width,
                    response_width=response_width,
                    prompt_length=prompt_length,
                    response_length=response_length,
                    pad_token_id=self.pad_token_id,
                )
                padded_teacher_counterfactual_logprobs.append(
                    padded_papo_counterfactual_logprobs
                )
            if topk_papo_gate:
                _, padded_sampled_logprobs = _pad_teacher_outputs(
                    sampled_teacher_ids,
                    sampled_teacher_logprobs,
                    prompt_width=prompt_width,
                    response_width=response_width,
                    prompt_length=prompt_length,
                    response_length=response_length,
                    pad_token_id=self.pad_token_id,
                )
                padded_teacher_sampled_logprobs.append(padded_sampled_logprobs)

        batch_data = {
            "teacher_ids": torch.cat(padded_teacher_ids),
            "teacher_logprobs": torch.cat(padded_teacher_logprobs),
        }
        if visual_grounding_enabled or token_level_va_enabled or vgd_enabled:
            batch_data["teacher_negative_logprobs"] = torch.cat(padded_teacher_negative_logprobs)
        elif residual_enabled or papo_teacher_contrast_enabled:
            batch_data["teacher_counterfactual_logprobs"] = torch.cat(padded_teacher_negative_logprobs)
        if separate_papo_counterfactual_enabled:
            batch_data["teacher_counterfactual_logprobs"] = torch.cat(
                padded_teacher_counterfactual_logprobs
            )
        if topk_papo_gate:
            batch_data["teacher_sampled_logprobs"] = torch.cat(padded_teacher_sampled_logprobs)
        batch = TensorDict(batch_data, batch_size=len(data))
        return DataProto(batch=batch)
