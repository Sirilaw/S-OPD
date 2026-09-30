#!/usr/bin/env python3
"""Losslessly repair deterministic MathVerse single-choice scoring.

This is an offline migration for score artifacts produced before mcqfix_v1.
It never calls a judge and never overwrites its input unless --force is used
with an explicitly matching --output path.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
VLMEVALKIT_ROOT = REPO_ROOT / "third_party" / "VLMEvalKit"
sys.path.insert(0, str(VLMEVALKIT_ROOT))

from vlmeval.dataset.utils.mathverse import (  # noqa: E402
    MathVerse_acc,
    extract_mathverse_mcq_answer,
    is_mathverse_single_choice,
    normalize_mathverse_mcq_answer,
)
from vlmeval.smp.file import dump, load  # noqa: E402

RESCORE_VERSION = "mcqfix_v1"

LEGACY_COLUMNS = (
    "extract",
    "log_extract",
    "extract_source",
    "score",
    "log_score",
    "score_source",
)


def default_output_path(input_path: Path) -> Path:
    suffix = f"_{RESCORE_VERSION}"
    if input_path.stem.endswith(suffix):
        return input_path.with_name(f"{input_path.stem}_rescored{input_path.suffix}")
    return input_path.with_name(f"{input_path.stem}{suffix}{input_path.suffix}")


def _score_as_bool(value) -> bool:
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1"}:
            return True
        if normalized in {"false", "0", ""}:
            return False
    return bool(value)


def repair_dataframe(data: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    required = {"prediction", "answer", "question_type", "score"}
    missing = sorted(required.difference(data.columns))
    if missing:
        raise ValueError(f"Missing required MathVerse columns: {missing}")

    repaired = data.copy(deep=True)
    for column in LEGACY_COLUMNS:
        if column in repaired and f"legacy_{column}" not in repaired:
            repaired[f"legacy_{column}"] = repaired[column].copy(deep=True)

    eligible = 0
    applied = 0
    extract_changed = 0
    score_changed = 0
    false_negatives_fixed = 0
    false_positives_fixed = 0
    applied_mask = []

    for index, row in repaired.iterrows():
        if not is_mathverse_single_choice(row):
            applied_mask.append(False)
            continue
        eligible += 1

        option = extract_mathverse_mcq_answer(row.get("prediction", ""))
        if option is None:
            applied_mask.append(False)
            continue

        applied += 1
        applied_mask.append(True)
        answer = normalize_mathverse_mcq_answer(row["answer"])
        new_score = option == answer
        old_score = _score_as_bool(row["score"])

        if str(row.get("extract", "")).strip() != option:
            extract_changed += 1
        if old_score != new_score:
            score_changed += 1
            if new_score:
                false_negatives_fixed += 1
            else:
                false_positives_fixed += 1

        repaired.at[index, "extract"] = option
        repaired.at[index, "log_extract"] = (
            "Offline deterministic MCQ extraction from lossless prediction"
        )
        repaired.at[index, "extract_source"] = "rule_mcq_prediction_rescore"
        repaired.at[index, "score"] = new_score
        repaired.at[index, "log_score"] = "Offline deterministic MCQ exact match"
        repaired.at[index, "score_source"] = "rule_mcq_exact_match_rescore"

    repaired["mcqfix_applied"] = applied_mask
    repaired["mcqfix_version"] = RESCORE_VERSION

    old_scores = data["score"].map(_score_as_bool)
    new_scores = repaired["score"].map(_score_as_bool)
    summary = {
        "version": RESCORE_VERSION,
        "total_rows": int(len(data)),
        "eligible_single_choice_rows": eligible,
        "deterministically_rescored_rows": applied,
        "ambiguous_single_choice_rows_retained": eligible - applied,
        "extracts_changed": extract_changed,
        "scores_changed": score_changed,
        "false_negatives_fixed": false_negatives_fixed,
        "false_positives_fixed": false_positives_fixed,
        "legacy_correct": int(old_scores.sum()),
        "repaired_correct": int(new_scores.sum()),
        "legacy_accuracy": float(old_scores.mean() * 100),
        "repaired_accuracy": float(new_scores.mean() * 100),
    }
    return repaired, summary


def repair_file(
    input_path: Path,
    output_path: Path | None = None,
    *,
    force: bool = False,
) -> tuple[Path, dict]:
    input_path = input_path.resolve()
    output_path = (output_path or default_output_path(input_path)).resolve()
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    if output_path == input_path and not force:
        raise ValueError("Refusing to overwrite the input without --force")
    if output_path.exists() and not force:
        raise FileExistsError(f"Output already exists: {output_path}")

    data = load(str(input_path))
    if not isinstance(data, pd.DataFrame):
        raise TypeError(f"Expected a DataFrame artifact, got {type(data).__name__}")

    repaired, summary = repair_dataframe(data)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dump(repaired, str(output_path))

    # Verify that the new artifact is readable and that every original field
    # remains available either unchanged or under its legacy_* mirror.
    restored = load(str(output_path))
    if not isinstance(restored, pd.DataFrame) or len(restored) != len(data):
        raise RuntimeError("Output verification failed: invalid DataFrame round trip")
    for column in LEGACY_COLUMNS:
        if column in data and f"legacy_{column}" not in restored:
            raise RuntimeError(f"Output verification failed: missing legacy_{column}")

    accuracy_path = output_path.with_suffix(".csv")
    summary_path = output_path.with_name(f"{output_path.stem}_repair_summary.json")
    dump(MathVerse_acc(str(output_path)), str(accuracy_path))
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    summary.update(
        output=str(output_path),
        accuracy_breakdown=str(accuracy_path),
        repair_summary=str(summary_path),
    )
    return output_path, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", type=Path, nargs="+", help="Legacy MathVerse score JSON files")
    parser.add_argument(
        "--output",
        type=Path,
        help="Output path; only valid with exactly one input (default: append _mcqfix_v1)",
    )
    parser.add_argument("--force", action="store_true", help="Replace an existing output")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output is not None and len(args.inputs) != 1:
        raise SystemExit("--output can only be used with one input")

    for input_path in args.inputs:
        _, summary = repair_file(input_path, args.output, force=args.force)
        print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
