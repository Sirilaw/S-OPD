"""ZoomBench dual-view dataset and official-style hybrid evaluator."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pandas as pd

from local_math_dataset import _as_list, _materialize_images
from vlmeval.dataset.image_base import ImageBaseDataset
from vlmeval.smp import dump, get_intermediate_file_path, load
from vlmeval.utils import track_progress_rich


ZOOMBENCH_REPO_ID = "inclusionAI/ZoomBench"
ZOOMBENCH_SPLIT = "test"
ZOOMBENCH_DEFAULT_JUDGE = "Qwen3-30B-A3B-Instruct-2507"

ZOOMBENCH_JUDGE_PROMPT = (
    "Your task is to judge whether the response expresses the same meaning as "
    "the answer of a question.\n"
    "The question is: {question}\n"
    "The answer is: {answer}\n"
    "The response is: {prediction}\n"
    "Please check and compare them and then judge. If the response is correct, "
    "your output should be Yes. Otherwise, your output should be No. Directly "
    "give me your output."
)


def resolve_zoombench_columns(column_names: list[str]) -> tuple[str, str]:
    """Resolve both the released parquet schema and the dataset-card schema."""
    columns = set(column_names)
    question_column = next(
        (name for name in ("query", "prompt", "question") if name in columns),
        None,
    )
    answer_column = next(
        (name for name in ("response", "answer") if name in columns),
        None,
    )
    missing = []
    if question_column is None:
        missing.append("question (expected one of: query, prompt, question)")
    if answer_column is None:
        missing.append("answer (expected one of: response, answer)")
    if missing:
        raise KeyError(
            f"Unable to resolve ZoomBench columns {missing}; available columns: "
            f"{sorted(columns)}"
        )
    return question_column, answer_column


def extract_zoombench_answer(value: Any) -> str:
    """Follow the answer extraction order in ZoomBench's released judge."""
    response = str(value or "").strip()
    tagged = re.search(r"<answer>(.*?)</answer>", response, flags=re.DOTALL | re.IGNORECASE)
    if tagged:
        return tagged.group(1).strip()
    answer_marker = re.search(r"answer\s*:\s*", response, flags=re.IGNORECASE)
    if answer_marker:
        return response[answer_marker.end():].strip()
    return "\n".join(response.splitlines()[-3:]).strip()


def _normalized_answer(value: Any) -> str:
    text = extract_zoombench_answer(value).casefold()
    text = re.sub(r"[\s\.,;:!?'‘’“”`]+", "", text)
    return text


def _extract_option(value: Any) -> str | None:
    text = extract_zoombench_answer(value).strip()
    patterns = (
        r"\(([A-Fa-f])\)",
        r"(?:^|\b)([A-Fa-f])(?:\.|\)|:|\s|$)",
    )
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1).upper()
    return None


def zoombench_deterministic_match(answer: Any, prediction: Any) -> bool:
    """Accept unambiguous exact or option-letter matches before LLM judging."""
    if _normalized_answer(answer) == _normalized_answer(prediction):
        return True
    gold_option = _extract_option(answer)
    pred_option = _extract_option(prediction)
    return gold_option is not None and gold_option == pred_option


def zoombench_judge_one(judge, question: Any, answer: Any, prediction: Any) -> dict[str, Any]:
    extracted = extract_zoombench_answer(prediction)
    if zoombench_deterministic_match(answer, extracted):
        return {"score": True, "judge_response": "", "judge_source": "deterministic", "judge_log": ""}

    prompt = ZOOMBENCH_JUDGE_PROMPT.format(
        question=str(question).replace("<image>", ""),
        answer=str(answer),
        prediction=extracted,
    )
    try:
        response = str(judge.generate(prompt)).strip()
    except Exception as exc:
        return {
            "score": False,
            "judge_response": "",
            "judge_source": "llm",
            "judge_log": f"ZOOMBENCH_JUDGE_FAILED: {type(exc).__name__}: {exc}",
        }

    normalized = response.casefold().strip().rstrip(".")
    if normalized == "yes":
        return {"score": True, "judge_response": response, "judge_source": "llm", "judge_log": ""}
    if normalized == "no":
        return {"score": False, "judge_response": response, "judge_source": "llm", "judge_log": ""}
    return {
        "score": False,
        "judge_response": response,
        "judge_source": "llm",
        "judge_log": "ZOOMBENCH_JUDGE_FAILED: judge did not return exactly Yes or No",
    }


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


class ZoomBenchDataset(ImageBaseDataset):
    """The full-image or crop-image view of the official ZoomBench test set."""

    TYPE = "VQA"
    MODALITY = "IMAGE"
    DEFAULT_JUDGE = ZOOMBENCH_DEFAULT_JUDGE
    JUDGE_FAIL_MARKERS = ("Failed to obtain answer", "ZOOMBENCH_JUDGE_FAILED")
    force_use_dataset_prompt = True

    def __init__(
        self,
        dataset: str = "ZoomBench",
        cache_dir: str | None = None,
        repo_id: str = ZOOMBENCH_REPO_ID,
        split: str = ZOOMBENCH_SPLIT,
        image_view: str | None = None,
    ) -> None:
        try:
            from datasets import Image, load_dataset
        except ImportError as exc:
            raise ImportError(
                "ZoomBench requires the Hugging Face `datasets` package. "
                "Install it with `pip install datasets`."
            ) from exc

        if image_view is None:
            image_view = "crop" if dataset.lower().endswith("_crop") else "full"
        if image_view not in {"full", "crop"}:
            raise ValueError("ZoomBench image_view must be 'full' or 'crop'")
        image_column = "crop_image" if image_view == "crop" else "image"

        cache = (
            Path(cache_dir).expanduser().resolve()
            if cache_dir
            else Path.cwd() / ".cache" / "vlmevalkit" / "ZoomBench"
        )
        hf_cache = cache / "huggingface"
        image_cache = cache / "images" / image_view
        hf_cache.mkdir(parents=True, exist_ok=True)
        image_cache.mkdir(parents=True, exist_ok=True)

        source = load_dataset(repo_id, split=split, cache_dir=str(hf_cache))
        question_column, answer_column = resolve_zoombench_columns(source.column_names)
        required_columns = {image_column}
        missing_columns = sorted(required_columns.difference(source.column_names))
        if missing_columns:
            raise KeyError(f"{repo_id}:{split} is missing columns: {missing_columns}")
        for column in ("image", "crop_image"):
            if column in source.column_names:
                source = source.cast_column(column, Image(decode=False))

        rows = []
        for offset, item in enumerate(source):
            image_paths = _materialize_images(
                item[image_column], root=hf_cache, cache=image_cache, row_index=offset
            )
            if not image_paths or not all(Path(path).is_file() for path in image_paths):
                raise FileNotFoundError(
                    f"Unable to materialize ZoomBench {image_view} image at row {offset}"
                )
            record = {
                key: value
                for key, value in item.items()
                if key not in {"image", "crop_image"}
                and isinstance(value, (str, int, float, bool, type(None)))
            }
            record.update(
                index=str(item.get("id", offset)),
                question=str(item[question_column]),
                answer=str(item[answer_column]),
                image_path=image_paths[0] if len(image_paths) == 1 else image_paths,
                image_view=image_view,
            )
            rows.append(record)

        self.dataset_name = dataset
        self.repo_id = repo_id
        self.split = split
        self.image_view = image_view
        self.data = pd.DataFrame(rows)
        self.meta_only = True
        self.skip_noimg = False
        self.img_root = str(image_cache)

    def build_prompt(self, line):
        if isinstance(line, int):
            line = self.data.iloc[line]
        return [
            *(dict(type="image", value=path) for path in _as_list(line["image_path"])),
            dict(type="text", value=str(line["question"])),
        ]

    def evaluate(self, eval_file, **judge_kwargs):
        from vlmeval.dataset.utils import build_judge

        eval_file = str(eval_file)
        data = load(eval_file)
        if not isinstance(data, pd.DataFrame):
            data = pd.DataFrame(data)
        required = {"index", "question", "answer", "prediction"}
        missing = sorted(required.difference(data.columns))
        if missing:
            raise KeyError(f"ZoomBench prediction file is missing columns: {missing}")

        judge_kwargs = dict(judge_kwargs)
        nproc = judge_kwargs.pop("nproc", 4)
        judge_model = judge_kwargs.pop("model", self.DEFAULT_JUDGE)
        judge_kwargs.pop("use_verifier", None)
        judge_kwargs.pop("use_vllm", None)
        judge = build_judge(model=judge_model, **judge_kwargs)
        assert judge.working(), (
            "ZoomBench open-question evaluation requires a working LLM judge; "
            f"could not use {judge_model}."
        )

        judge_tag = _safe_name(judge_model)
        checkpoint = get_intermediate_file_path(
            eval_file, f"_zoombench_{judge_tag}_official_v1", "pkl"
        )
        indices = [str(value) for value in data["index"]]
        cached = load(checkpoint) if Path(checkpoint).exists() else {}
        tasks = []
        pending_indices = []
        for _, row in data.iterrows():
            index = str(row["index"])
            if index not in cached:
                tasks.append((judge, row["question"], row["answer"], row["prediction"]))
                pending_indices.append(index)
        if tasks:
            track_progress_rich(
                zoombench_judge_one,
                tasks,
                nproc=nproc,
                chunksize=nproc,
                keys=pending_indices,
                save=checkpoint,
            )
            cached = load(checkpoint)

        results = [cached[index] for index in indices]
        data["extracted_prediction"] = data["prediction"].map(extract_zoombench_answer)
        data["judge_response"] = [result["judge_response"] for result in results]
        data["judge_source"] = [result["judge_source"] for result in results]
        data["judge_log"] = [result["judge_log"] for result in results]
        data["correct"] = [bool(result["score"]) for result in results]

        scored_file = get_intermediate_file_path(
            eval_file, f"_zoombench_{judge_tag}_scored", "xlsx"
        )
        dump(data, scored_file)
        total = int(len(data))
        correct = int(data["correct"].sum())
        metrics = {
            "Overall": round((correct / total if total else 0.0) * 100, 4),
            "accuracy": correct / total if total else 0.0,
            "correct": correct,
            "total": total,
            "image_view": self.image_view,
        }
        category_column = next(
            (name for name in ("question_type", "category") if name in data.columns),
            None,
        )
        if category_column is not None:
            for category, group in data.groupby(category_column, dropna=False):
                metrics[f"{category_column}/{category}"] = round(
                    float(group["correct"].mean()) * 100, 4
                )
        score_file = get_intermediate_file_path(
            eval_file, f"_zoombench_{judge_tag}_score", "json"
        )
        dump(metrics, score_file)
        return metrics
