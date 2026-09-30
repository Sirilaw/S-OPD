from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "analyze_teacher_gate_keep_ratio.py"
SPEC = importlib.util.spec_from_file_location("teacher_gate_ratio", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_analyze_row_matches_strict_positive_teacher_gate(tmp_path):
    row = {
        "step": 10,
        "score": 1.0,
        "teacher_token_logprobs": [[-0.1], [-1.0], [-0.5], [-2.0]],
        "teacher_counterfactual_token_logprobs": [[-0.2], [-0.5], [-0.5], [-3.0]],
    }

    result = MODULE.analyze_row(
        row,
        source_file=tmp_path / "10.jsonl",
        line_number=1,
        rollout_index=0,
        clean_field=None,
        counterfactual_field=None,
        teacher_index=0,
        threshold=0.0,
    )

    assert result["response_token_count"] == 4
    assert result["kept_token_count"] == 2
    assert result["keep_fraction"] == 0.5
    assert result["mean_teacher_delta"] == pytest.approx(0.15)
    assert result["mean_teacher_gate_scale"] == pytest.approx(0.275)


def test_summarize_reports_macro_and_token_weighted_micro_fractions():
    rows = [
        {
            "step": 1,
            "response_token_count": 2,
            "kept_token_count": 2,
            "keep_fraction": 1.0,
            "mean_teacher_delta": 0.2,
            "mean_teacher_gate_scale": 0.2,
        },
        {
            "step": 1,
            "response_token_count": 8,
            "kept_token_count": 0,
            "keep_fraction": 0.0,
            "mean_teacher_delta": -0.1,
            "mean_teacher_gate_scale": 0.0,
        },
    ]

    summary = MODULE.summarize_rows(rows)

    assert summary["macro_keep_fraction"] == 0.5
    assert summary["micro_keep_fraction"] == 0.2
    assert summary["keep_fraction_median"] == 0.5


def test_discover_experiment_rollout_files_in_numeric_order(tmp_path):
    rollout_dir = tmp_path / "rollouts"
    rollout_dir.mkdir()
    for name in ("20.jsonl", "3.jsonl", "notes.jsonl"):
        (rollout_dir / name).write_text(json.dumps({"name": name}) + "\n", encoding="utf-8")

    paths = MODULE.discover_jsonl_files(tmp_path)

    assert [path.name for path in paths] == ["3.jsonl", "20.jsonl", "notes.jsonl"]
