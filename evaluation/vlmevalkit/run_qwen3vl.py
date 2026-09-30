#!/usr/bin/env python3
"""Run Qwen3-VL on local visual-math benchmarks through VLMEvalKit."""

from __future__ import annotations

import argparse
import json
import os
import random
import runpy
import sys
import tempfile
from pathlib import Path


def isolate_distributed_gpu() -> None:
    """Bind a torchrun rank before importing anything that may initialize CUDA."""
    local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", "1"))
    if local_world_size <= 1:
        return

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    visible_devices = [
        device.strip()
        for device in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
        if device.strip()
    ]
    if len(visible_devices) != local_world_size:
        raise SystemExit(
            "Distributed inference requires one CUDA_VISIBLE_DEVICES entry per "
            f"local rank; got {len(visible_devices)} devices for "
            f"LOCAL_WORLD_SIZE={local_world_size}."
        )
    if not 0 <= local_rank < local_world_size:
        raise SystemExit(
            f"LOCAL_RANK={local_rank} is outside [0, {local_world_size})."
        )

    os.environ["CUDA_VISIBLE_DEVICES"] = visible_devices[local_rank]
    os.environ["VLMEVAL_GPU_ALREADY_ISOLATED"] = "1"


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vlmevalkit-root", required=True, help="Path to a VLMEvalKit checkout")
    parser.add_argument("--model-path", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--benchmark", action="append", default=[], metavar="NAME=MANIFEST")
    parser.add_argument("--builtin-benchmark", action="append", default=[], metavar="NAME")
    parser.add_argument("--mmk12-repo", default="FanqingM/MMK12")
    parser.add_argument("--mmk12-split", default="test")
    parser.add_argument("--zoombench-repo", default="inclusionAI/ZoomBench")
    parser.add_argument("--zoombench-split", default="test")
    parser.add_argument("--image-root", action="append", default=[], metavar="NAME=DIR")
    parser.add_argument("--question-field")
    parser.add_argument("--answer-field")
    parser.add_argument("--image-field")
    parser.add_argument(
        "--image-mask-ratio",
        type=float,
        default=0.0,
        help="Independently black out this fraction of image patches (0 disables ablation).",
    )
    parser.add_argument("--image-mask-patch-size", type=int, default=14)
    parser.add_argument("--image-mask-seed", type=int, default=0)
    parser.add_argument("--backend", choices=("vllm", "transformers"), default="vllm")
    parser.add_argument("--max-new-tokens", type=int, default=8192)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--max-model-len", type=int, default=16384)
    parser.add_argument("--max-num-seqs", type=int, default=8)
    parser.add_argument("--global-dynamic-queue", action="store_true")
    parser.add_argument("--dynamic-inflight-per-gpu", type=int)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--presence-penalty", type=float, default=1.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--system-prompt")
    parser.add_argument("--strip-user-suffix", action="store_true")
    parser.add_argument(
        "--require-boxed-answer",
        action="store_true",
        help="Count predictions without a complete non-empty \\boxed{...} as incorrect.",
    )
    parser.add_argument("--dataset-cache-root")
    parser.add_argument("--disable-internal-log", action="store_true")
    parser.add_argument("--work-dir", default="outputs/vlmevalkit")
    parser.add_argument("--mode", choices=("all", "infer", "eval"), default="all")
    return parser.parse_known_args()


def assignments(values: list[str], flag: str) -> dict[str, str]:
    result = {}
    for value in values:
        if "=" not in value:
            raise SystemExit(f"{flag} expects NAME=PATH, got {value!r}")
        name, path = value.split("=", 1)
        if not name or not path:
            raise SystemExit(f"Invalid {flag}: {value!r}")
        result[name] = str(Path(path).expanduser().resolve())
    return result


def builtin_dataset_class(name: str) -> str:
    """Return the VLMEvalKit dataset adapter for a built-in benchmark."""
    if name == "MMK12":
        return "MMK12Dataset"
    if name in {"ZoomBench", "ZoomBench_Crop"}:
        return "ZoomBenchDataset"
    if name == "VStarBench":
        return "VStarBenchDataset"
    if name in {"MMMU_DEV_VAL", "MMMU_TEST"}:
        return "MMMUDataset"
    if name in {"WeMath", "WeMath_COT"}:
        return "WeMath"
    if name == "LogicVista":
        return "LogicVista"
    if name == "MathVista_MINI":
        return "MathVista"
    if name.startswith("MathVerse_MINI"):
        return "MathVerse"
    if name in {"MathVision", "MathVision_MINI"}:
        return "MathVision"
    if name in {"DynaMath", "DynaMath_noprompt"}:
        return "Dynamath"
    if name in {"ZEROBench", "ZEROBench_sub"}:
        return "ZEROBench"
    if name == "VisualPuzzles":
        return "VisualPuzzles"
    if name == "VisuLogic":
        return "VisuLogic"
    if name in {"POPE", "HallusionBench"}:
        return "ImageYORNDataset"
    return "ImageMCQDataset"


def main() -> None:
    # This must precede imports from vlmeval: those imports can transitively
    # initialize CUDA, after which changing CUDA_VISIBLE_DEVICES is too late.
    isolate_distributed_gpu()
    args, passthrough = parse_args()
    if args.seed < 0:
        raise SystemExit("--seed must be non-negative")
    if args.temperature < 0:
        raise SystemExit("--temperature must be non-negative")
    if not 0 < args.top_p <= 1:
        raise SystemExit("--top-p must be in (0, 1]")
    if args.top_k == 0 or args.top_k < -1:
        raise SystemExit("--top-k must be -1 or a positive integer")
    if args.repetition_penalty <= 0:
        raise SystemExit("--repetition-penalty must be positive")
    if not -2 <= args.presence_penalty <= 2:
        raise SystemExit("--presence-penalty must be in [-2, 2]")
    if not 0.0 <= args.image_mask_ratio <= 1.0:
        raise SystemExit("--image-mask-ratio must be in [0, 1]")
    if args.image_mask_patch_size <= 0:
        raise SystemExit("--image-mask-patch-size must be positive")
    if args.image_mask_seed < 0:
        raise SystemExit("--image-mask-seed must be non-negative")
    random.seed(args.seed)
    import torch
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    if args.global_dynamic_queue and args.backend != "vllm":
        raise SystemExit("--global-dynamic-queue requires --backend vllm")
    if args.dynamic_inflight_per_gpu is not None and args.dynamic_inflight_per_gpu <= 0:
        raise SystemExit("--dynamic-inflight-per-gpu must be positive")
    if args.global_dynamic_queue:
        os.environ["VLMEVAL_VLLM_GLOBAL_QUEUE"] = "1"
        os.environ["VLMEVAL_BALANCED_SHARDING"] = "1"
    if args.dynamic_inflight_per_gpu is not None:
        os.environ["VLMEVAL_DYNAMIC_INFLIGHT_PER_GPU"] = str(args.dynamic_inflight_per_gpu)
    root = Path(args.vlmevalkit_root).expanduser().resolve()
    run_file = root / "run.py"
    if not run_file.is_file():
        raise SystemExit(f"VLMEvalKit run.py not found: {run_file}")

    benchmarks = assignments(args.benchmark, "--benchmark")
    if not benchmarks and not args.builtin_benchmark:
        raise SystemExit("Pass at least one --benchmark NAME=MANIFEST or --builtin-benchmark NAME")
    image_roots = assignments(args.image_root, "--image-root")
    sys.path.insert(0, str(root))

    # Register only for this process; the VLMEvalKit checkout remains untouched.
    import vlmeval.dataset
    import vlmeval.smp
    from local_math_dataset import LocalMathDataset
    from mmk12_dataset import MMK12Dataset
    from vstar_dataset import VStarBenchDataset
    from zoombench_dataset import ZoomBenchDataset
    vlmeval.dataset.LocalMathDataset = LocalMathDataset
    vlmeval.dataset.MMK12Dataset = MMK12Dataset
    vlmeval.dataset.VStarBenchDataset = VStarBenchDataset
    vlmeval.dataset.ZoomBenchDataset = ZoomBenchDataset
    if args.disable_internal_log:
        original_setup_logger = vlmeval.smp.setup_logger

        def console_only_setup_logger(log_file=None, *setup_args, **setup_kwargs):
            del log_file
            return original_setup_logger(None, *setup_args, **setup_kwargs)

        vlmeval.smp.setup_logger = console_only_setup_logger

    model_name = "qwen3vl-local"
    config = {
        "model": {
            model_name: {
                "class": "Qwen3VLChat",
                "model_path": (
                    str(Path(args.model_path).expanduser())
                    if Path(args.model_path).exists()
                    else args.model_path
                ),
                "use_vllm": args.backend == "vllm",
                "use_vllm_async": args.global_dynamic_queue,
                "use_custom_prompt": False,
                "max_new_tokens": args.max_new_tokens,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "top_k": args.top_k,
                "repetition_penalty": args.repetition_penalty,
                "presence_penalty": args.presence_penalty,
                "seed": args.seed,
                "system_prompt": args.system_prompt,
                "gpu_utils": args.gpu_memory_utilization,
                "max_model_len": args.max_model_len,
                "max_num_seqs": args.max_num_seqs,
            }
        },
        "data": {},
    }
    cache_root = (
        Path(args.dataset_cache_root).expanduser().resolve()
        if args.dataset_cache_root
        else Path(args.work_dir).expanduser().resolve() / "dataset_cache"
    )
    for name, manifest in benchmarks.items():
        cache_dir = str(cache_root / name)
        entry = {
            "class": "LocalMathDataset",
            "dataset": name,
            "data_file": manifest,
            "cache_dir": cache_dir,
            "strip_user_suffix": args.strip_user_suffix,
            "image_mask_ratio": args.image_mask_ratio,
            "image_mask_patch_size": args.image_mask_patch_size,
            "image_mask_seed": args.image_mask_seed,
            "require_boxed_answer": args.require_boxed_answer,
        }
        for key in ("question_field", "answer_field", "image_field"):
            value = getattr(args, key)
            if value:
                entry[key] = value
        if name in image_roots:
            entry["image_root"] = image_roots[name]
        config["data"][name] = entry
    for name in args.builtin_benchmark:
        if name == "MMK12":
            config["data"][name] = {
                "class": "MMK12Dataset",
                "dataset": name,
                "cache_dir": str(cache_root / name),
                "repo_id": args.mmk12_repo,
                "split": args.mmk12_split,
            }
            continue
        if name in {"ZoomBench", "ZoomBench_Crop"}:
            config["data"][name] = {
                "class": "ZoomBenchDataset",
                "dataset": name,
                "cache_dir": str(cache_root / "ZoomBench"),
                "repo_id": args.zoombench_repo,
                "split": args.zoombench_split,
                "image_view": "crop" if name == "ZoomBench_Crop" else "full",
            }
            continue
        dataset_class = builtin_dataset_class(name)
        config["data"][name] = {"class": dataset_class, "dataset": name}

    with tempfile.NamedTemporaryFile("w", suffix=".json", encoding="utf-8", delete=False) as handle:
        json.dump(config, handle, ensure_ascii=False, indent=2)
        config_path = handle.name
    try:
        sys.argv = [
            str(run_file),
            "--config",
            config_path,
            "--work-dir",
            args.work_dir,
            "--mode",
            args.mode,
            *passthrough,
        ]
        runpy.run_path(str(run_file), run_name="__main__")
    finally:
        os.unlink(config_path)


if __name__ == "__main__":
    main()
