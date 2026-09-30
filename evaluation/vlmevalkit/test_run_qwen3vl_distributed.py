import os
import sys

import pytest
from run_qwen3vl import isolate_distributed_gpu, parse_args


@pytest.mark.parametrize(
    ("local_rank", "expected_gpu"),
    [(0, "2"), (1, "3"), (2, "4"), (3, "5")],
)
def test_isolate_distributed_gpu_before_cuda_import(monkeypatch, local_rank, expected_gpu):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,3,4,5")
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "4")
    monkeypatch.setenv("LOCAL_RANK", str(local_rank))
    monkeypatch.delenv("VLMEVAL_GPU_ALREADY_ISOLATED", raising=False)

    isolate_distributed_gpu()

    assert os.environ["CUDA_VISIBLE_DEVICES"] == expected_gpu
    assert os.environ["VLMEVAL_GPU_ALREADY_ISOLATED"] == "1"


def test_isolate_distributed_gpu_rejects_mismatched_device_count(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,3")
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "4")
    monkeypatch.setenv("LOCAL_RANK", "0")

    with pytest.raises(SystemExit, match="one CUDA_VISIBLE_DEVICES entry per local rank"):
        isolate_distributed_gpu()


def test_sampling_penalties_are_parsed(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_qwen3vl.py",
            "--vlmevalkit-root",
            "/tmp/vlmevalkit",
            "--temperature",
            "0.7",
            "--top-p",
            "0.8",
            "--top-k",
            "20",
            "--repetition-penalty",
            "1.0",
            "--presence-penalty",
            "1.5",
        ],
    )

    args, passthrough = parse_args()

    assert args.temperature == 0.7
    assert args.top_p == 0.8
    assert args.top_k == 20
    assert args.repetition_penalty == 1.0
    assert args.presence_penalty == 1.5
    assert passthrough == []
