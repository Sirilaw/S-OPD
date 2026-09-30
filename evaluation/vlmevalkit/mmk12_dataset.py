"""Official-style MMK12 dataset and evaluator for the local VLMEvalKit wrapper."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pandas as pd
from local_math_dataset import _as_list, _last_boxed, _materialize_images
from vlmeval.dataset.image_base import ImageBaseDataset
from vlmeval.smp import dump, get_intermediate_file_path, load
from vlmeval.utils import track_progress_rich

MMK12_REPO_ID = "FanqingM/MMK12"
MMK12_SPLIT = "test"
MMK12_DEFAULT_JUDGE = "gpt-4o"

MMK12_JUDGE_PROMPT = (
    "You are given a question, the correct answer and a model's answer.\n"
    "Please determine if the model's answer matches the correct answer.\n"
    "Focus only on the mathematical or semantic correctness of the content. "
    "Ignore any differences in formatting, such as LaTeX syntax, symbols, "
    "styles, or additional wrappers (e.g., \\boxed, $...$, or similar). "
    "Compare only the core mathematical or textual meaning of the model's "
    "answer and the correct answer.\n\n"
    "The process or reasoning leading to the Solution is irrelevant, Only the "
    "correctness of the model's answer matters.\n\n"
    'Return only "Yes" if the model\'s answer is correct or "No" if it is '
    "incorrect.\n"
    'Only return "Yes" or "No" with no additional text or formatting.\n\n'
    "Question:\n\n{question}\n\n"
    "--------------------------------\n\n"
    "Correct Answer:\n\n{answer}\n\n"
    "--------------------------------\n\n"
    "Model's Answer:\n\n{solution}\n\n"
    "--------------------------------\n"
)


def extract_mmk12_answer(value: Any) -> str:
    """Follow the official answer-tag, boxed-answer, full-response fallback."""
    response = str(value or "")
    match = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)
    if match:
        return match.group(1).strip()
    boxed = _last_boxed(response)
    return boxed.strip() if boxed is not None else response.strip()


def build_mmk12_judge_prompt(question: Any, answer: Any, prediction: Any) -> str:
    return MMK12_JUDGE_PROMPT.format(
        question=str(question),
        answer=str(answer),
        solution=extract_mmk12_answer(prediction),
    )


def mmk12_judge_one(judge, question: Any, answer: Any, prediction: Any) -> dict[str, Any]:
    """Run MMK12's Yes/No judging protocol with the shared judge settings."""
    prompt = build_mmk12_judge_prompt(question, answer, prediction)
    try:
        response = str(judge.generate(prompt)).strip()
    except Exception as exc:
        return {
            "score": False,
            "judge_response": "",
            "judge_log": f"MMK12_JUDGE_FAILED: {type(exc).__name__}: {exc}",
        }

    normalized = response.lower()
    if normalized == "yes":
        return {"score": True, "judge_response": response, "judge_log": ""}
    if normalized == "no":
        return {"score": False, "judge_response": response, "judge_log": ""}
    fail_msg = getattr(judge, "fail_msg", "")
    reason = (
        "judge API request failed"
        if fail_msg and fail_msg in response
        else "judge did not return exactly Yes or No"
    )
    return {
        "score": False,
        "judge_response": response,
        "judge_log": f"MMK12_JUDGE_FAILED: {reason}",
    }


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


class MMK12Dataset(ImageBaseDataset):
    """MMK12 test split with the evaluator released by MM-EUREKA."""

    TYPE = "VQA"
    MODALITY = "IMAGE"
    DEFAULT_JUDGE = MMK12_DEFAULT_JUDGE
    JUDGE_FAIL_MARKERS = ("Failed to obtain answer", "MMK12_JUDGE_FAILED")
    force_use_dataset_prompt = True

    def __init__(
        self,
        dataset: str = "MMK12",
        cache_dir: str | None = None,
        repo_id: str = MMK12_REPO_ID,
        split: str = MMK12_SPLIT,
    ) -> None:
        try:
            from datasets import Image, load_dataset
        except ImportError as exc:
            raise ImportError(
                "MMK12 requires the Hugging Face `datasets` package. "
                "Install it with `pip install datasets`."
            ) from exc

        cache = (
            Path(cache_dir).expanduser().resolve()
            if cache_dir
            else Path.cwd() / ".cache" / "vlmevalkit" / dataset
        )
        hf_cache = cache / "huggingface"
        image_cache = cache / "images"
        hf_cache.mkdir(parents=True, exist_ok=True)
        image_cache.mkdir(parents=True, exist_ok=True)

        source = load_dataset(repo_id, split=split, cache_dir=str(hf_cache))
        if "image" not in source.column_names:
            raise KeyError(f"{repo_id}:{split} has no `image` column")
        source = source.cast_column("image", Image(decode=False))

        rows = []
        for offset, item in enumerate(source):
            missing = [key for key in ("question", "answer", "image") if key not in item]
            if missing:
                raise KeyError(f"{repo_id}:{split} row {offset} is missing columns: {missing}")
            image_paths = _materialize_images(
                item["image"],
                root=hf_cache,
                cache=image_cache,
                row_index=offset,
            )
            if not image_paths or not all(Path(path).is_file() for path in image_paths):
                raise FileNotFoundError(f"Unable to materialize MMK12 image at row {offset}")

            identifier = str(item.get("id", offset))
            record = dict(item)
            record.pop("image", None)
            record.update(
                index=identifier,
                question=str(item["question"]),
                answer=str(item["answer"]),
                image_path=image_paths[0] if len(image_paths) == 1 else image_paths,
            )
            rows.append(record)

        self.dataset_name = dataset
        self.repo_id = repo_id
        self.split = split
        self.data = pd.DataFrame(rows)
        self.meta_only = True
        self.skip_noimg = False
        self.img_root = str(image_cache)

    def build_prompt(self, line):
        if isinstance(line, int):
            line = self.data.iloc[line]
        images = _as_list(line["image_path"])
        return [
            *(dict(type="image", value=path) for path in images),
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
            raise KeyError(f"MMK12 prediction file is missing columns: {missing}")

        judge_kwargs = dict(judge_kwargs)
        nproc = judge_kwargs.pop("nproc", 4)
        judge_model = judge_kwargs.pop("model", self.DEFAULT_JUDGE)
        judge_kwargs.pop("use_verifier", None)
        judge_kwargs.pop("use_vllm", None)
        judge = build_judge(
            model=judge_model,
            **judge_kwargs,
        )
        assert judge.working(), (
            f"MMK12 evaluation requires a working {judge_model} compatible API"
        )

        judge_tag = _safe_name(judge_model)
        checkpoint = get_intermediate_file_path(
            eval_file, f"_mmk12_{judge_tag}_official_v1", "pkl"
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
                mmk12_judge_one,
                tasks,
                nproc=nproc,
                chunksize=nproc,
                keys=pending_indices,
                save=checkpoint,
            )
            cached = load(checkpoint)

        results = [cached[index] for index in indices]
        data["extracted_prediction"] = data["prediction"].map(extract_mmk12_answer)
        data["judge_response"] = [result["judge_response"] for result in results]
        data["judge_log"] = [result["judge_log"] for result in results]
        data["correct"] = [bool(result["score"]) for result in results]

        scored_file = get_intermediate_file_path(
            eval_file, f"_mmk12_{judge_tag}_scored", "xlsx"
        )
        dump(data, scored_file)

        correct = int(data["correct"].sum())
        total = int(len(data))
        accuracy = correct / total if total else 0.0
        metrics = {
            "Overall": round(accuracy * 100, 4),
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
        }
        score_file = get_intermediate_file_path(
            eval_file, f"_mmk12_{judge_tag}_score", "json"
        )
        dump(metrics, score_file)
        return metrics
