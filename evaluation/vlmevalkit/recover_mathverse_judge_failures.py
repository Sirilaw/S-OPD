#!/usr/bin/env python3
"""Recover MathVerse MCQ extraction failures from retained judge responses.

Older MathVerse artifacts treated a judge response such as ``65°`` as a parse
failure because the MCQ parser accepted only an option letter.  The artifacts
retain all five judge responses, so they can be adjudicated offline without
calling the API again.  This script requires a majority response, maps that
response to a unique option by textual or scalar equivalence, and preserves the
original fields for audit.  A majority response that is not any offered option
is a valid extraction but an incorrect MCQ answer.  Rows without a majority
remain unresolved.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
VLMEVALKIT_ROOT = REPO_ROOT / "third_party" / "VLMEvalKit"
sys.path.insert(0, str(VLMEVALKIT_ROOT))

from vlmeval.dataset.utils.mathverse import MathVerse_acc  # noqa: E402
from vlmeval.smp.file import dump, load  # noqa: E402

VERSION = "judge_response_recovery_v1"
FAIL_SOURCE = "judge_parse_failure"


def parse_choices(question: object) -> dict[str, str]:
    text = str(question)
    if "Choices:" not in text:
        return {}
    choices_text = text.split("Choices:", 1)[1]
    matches = list(
        re.finditer(r"(?m)^\s*([A-F])\s*[:：.)]\s*", choices_text)
    )
    choices: dict[str, str] = {}
    for position, match in enumerate(matches):
        end = matches[position + 1].start() if position + 1 < len(matches) else len(choices_text)
        choices[match.group(1)] = choices_text[match.end() : end].strip()
    return choices


def _latex_to_plain(value: str) -> str:
    value = re.sub(r"\\(?:d?frac)\s*\{([^{}]+)\}\s*\{([^{}]+)\}", r"(\1)/(\2)", value)
    value = re.sub(r"\\sqrt\s*\{([^{}]+)\}", r"sqrt(\1)", value)
    value = re.sub(r"\\text\s*\{([^{}]*)\}", r"\1", value)
    value = re.sub(r"\\boxed\s*\{([^{}]*)\}", r"\1", value)
    return value


def normalize_text(value: object) -> str:
    text = unicodedata.normalize("NFKC", str(value)).strip().lower()
    text = re.sub(r"^\s*(?:extracted\s+answer|answer)\s*[:：]\s*", "", text)
    text = re.sub(r"√\s*\{([^{}]+)\}", r"sqrt(\1)", text)
    text = re.sub(r"√\s*([0-9.]+)", r"sqrt(\1)", text)
    text = _latex_to_plain(text)
    text = text.replace("−", "-").replace("^\\circ", "").replace("°", "")
    text = re.sub(r"(?:degrees?|厘米|米|cm|mm|meters?|metres?)\b", "", text)
    text = re.sub(r"[\\$`*_{}\[\]\s,，。!]", "", text)
    return text.strip("'\".")


def _safe_scalar(value: object) -> float | None:
    text = normalize_text(value)
    if not text or text in {"null", "none", "nan"}:
        return None
    # A concise assignment such as ``EC = 2.0`` expresses the right-hand side.
    if "=" in text and text.count("=") == 1:
        left, right = text.split("=", 1)
        if re.fullmatch(r"[a-zα-ωθ]+", left):
            text = right
    text = re.sub(r"(?<=\d)(?=sqrt)", "*", text)
    text = text.replace("sqrt", "math.sqrt")
    if not re.fullmatch(r"[0-9.+\-*/()mathsqrt]+", text):
        return None
    try:
        value_float = float(eval(text, {"__builtins__": {}, "math": math}, {}))
    except (ArithmeticError, NameError, SyntaxError, TypeError, ValueError):
        return None
    return value_float if math.isfinite(value_float) else None


def response_candidates(response: object) -> list[str]:
    text = str(response).strip()
    candidates = [text]
    # Local judges occasionally include a derivation despite the extraction
    # prompt.  Only accept an explicitly cued final clause in that case.
    cues = re.findall(
        r"(?is)(?:therefore|thus|final\s+answer|answer\s+is|is)\s*[:：]?\s*([^\n.]+)",
        text,
    )
    if cues:
        candidates.append(cues[-1].strip())
    return list(dict.fromkeys(candidates))


def map_response_to_option(response: object, choices: dict[str, str]) -> str | None:
    normalized_choices = {key: normalize_text(value) for key, value in choices.items()}
    scalar_choices = {key: _safe_scalar(value) for key, value in choices.items()}
    matched: set[str] = set()
    for candidate in response_candidates(response):
        normalized = normalize_text(candidate)
        direct = re.fullmatch(r"(?:option)?([a-f])", normalized)
        if direct and direct.group(1).upper() in choices:
            matched.add(direct.group(1).upper())
        matched.update(key for key, value in normalized_choices.items() if normalized == value)
        scalar = _safe_scalar(candidate)
        if scalar is not None:
            matched.update(
                key
                for key, value in scalar_choices.items()
                if value is not None and math.isclose(scalar, value, rel_tol=1e-8, abs_tol=1e-8)
            )
    return next(iter(matched)) if len(matched) == 1 else None


def recover_dataframe(data: pd.DataFrame, minimum_votes: int = 3) -> tuple[pd.DataFrame, dict]:
    required = {
        "answer",
        "extract_judge_responses",
        "extract_source",
        "question_for_eval",
        "score",
    }
    missing = sorted(required.difference(data.columns))
    if missing:
        raise ValueError(f"Missing required MathVerse columns: {missing}")

    recovered = data.copy(deep=True)
    for column in ("extract", "log_extract", "extract_source", "score", "log_score", "score_source"):
        legacy = f"legacy_{column}"
        if column in recovered and legacy not in recovered:
            recovered[legacy] = recovered[column].copy(deep=True)

    stats: defaultdict[str, int] = defaultdict(int)
    split_stats: defaultdict[str, Counter] = defaultdict(Counter)
    applied: list[bool] = []

    for index, row in recovered.iterrows():
        if row.get("extract_source") != FAIL_SOURCE:
            applied.append(False)
            continue
        stats["input_failures"] += 1
        split = str(row.get("problem_version", "unknown"))
        responses = [str(value).strip() for value in row.get("extract_judge_responses", [])]
        responses = [value for value in responses if value]
        majority = Counter(responses).most_common(1)
        if not majority or majority[0][1] < minimum_votes:
            stats["unresolved_no_majority"] += 1
            split_stats[split]["unresolved"] += 1
            applied.append(False)
            continue

        majority_response, votes = majority[0]
        choices = parse_choices(row.get("question_for_eval", ""))
        option = map_response_to_option(majority_response, choices)
        answer = str(row.get("answer", "")).strip().upper()
        is_correct = option == answer if option is not None else False

        recovered.at[index, "extract"] = option if option is not None else majority_response
        recovered.at[index, "log_extract"] = (
            f"Offline recovery from retained judge responses: {votes}/{len(responses)} majority; "
            + (f"mapped uniquely to option {option}" if option else "majority answer is not an offered option")
        )
        recovered.at[index, "extract_source"] = (
            "judge_response_choice_majority" if option else "judge_response_invalid_choice_majority"
        )
        recovered.at[index, "score"] = is_correct
        recovered.at[index, "log_score"] = "Offline majority-response MCQ adjudication"
        recovered.at[index, "score_source"] = VERSION
        applied.append(True)

        stats["recovered_failures"] += 1
        split_stats[split]["recovered"] += 1
        if option is None:
            stats["majority_not_in_choices"] += 1
            split_stats[split]["not_in_choices"] += 1
        elif is_correct:
            stats["correct_recovered"] += 1
            split_stats[split]["correct_recovered"] += 1
        else:
            stats["incorrect_option_recovered"] += 1
            split_stats[split]["incorrect_option"] += 1

    recovered["judge_recovery_applied"] = applied
    recovered["judge_recovery_version"] = VERSION
    stats["remaining_failures"] = stats["input_failures"] - stats["recovered_failures"]
    summary = dict(stats)
    summary["version"] = VERSION
    summary["minimum_votes"] = minimum_votes
    summary["total_rows"] = int(len(data))
    summary["legacy_correct"] = int(data["score"].map(bool).sum())
    summary["recovered_correct"] = int(recovered["score"].map(bool).sum())
    summary["legacy_accuracy"] = float(data["score"].map(bool).mean() * 100)
    summary["recovered_accuracy"] = float(recovered["score"].map(bool).mean() * 100)
    summary["splits"] = {key: dict(value) for key, value in sorted(split_stats.items())}
    return recovered, summary


def recover_file(input_path: Path, output_path: Path, minimum_votes: int, force: bool) -> dict:
    input_path = input_path.resolve()
    output_path = output_path.resolve()
    if output_path.exists() and not force:
        raise FileExistsError(output_path)
    data = load(str(input_path))
    if not isinstance(data, pd.DataFrame):
        raise TypeError(f"Expected DataFrame, got {type(data).__name__}")
    recovered, summary = recover_dataframe(data, minimum_votes=minimum_votes)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dump(recovered, str(output_path))
    accuracy_path = output_path.with_suffix(".csv")
    summary_path = output_path.with_name(f"{output_path.stem}_summary.json")
    dump(MathVerse_acc(str(output_path)), str(accuracy_path))
    summary.update(input=str(input_path), output=str(output_path), accuracy_breakdown=str(accuracy_path))
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    restored = load(str(output_path))
    if not isinstance(restored, pd.DataFrame) or len(restored) != len(data):
        raise RuntimeError("Output round-trip verification failed")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--minimum-votes", type=int, default=3)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 1 <= args.minimum_votes <= 5:
        raise SystemExit("--minimum-votes must be in [1, 5]")
    summary = recover_file(args.input, args.output, args.minimum_votes, args.force)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
