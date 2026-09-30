import sys
import types
from pathlib import Path


HERE = Path(__file__).resolve().parent
VLM_ROOT = HERE.parents[1] / "third_party" / "VLMEvalKit"
for path in (HERE, VLM_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

# Keep the pure scoring tests independent of optional VLMEvalKit dependencies.
vlmeval = types.ModuleType("vlmeval")
dataset = types.ModuleType("vlmeval.dataset")
image_base = types.ModuleType("vlmeval.dataset.image_base")
dataset_utils = types.ModuleType("vlmeval.dataset.utils")
smp = types.ModuleType("vlmeval.smp")
utils = types.ModuleType("vlmeval.utils")
image_base.ImageBaseDataset = object
dataset_utils.build_judge = lambda **kwargs: None
smp.dump = lambda *args: None
smp.get_intermediate_file_path = lambda path, *args: path
smp.load = lambda *args: None
utils.track_progress_rich = lambda *args, **kwargs: None
sys.modules.setdefault("vlmeval", vlmeval)
sys.modules.setdefault("vlmeval.dataset", dataset)
sys.modules.setdefault("vlmeval.dataset.image_base", image_base)
sys.modules.setdefault("vlmeval.dataset.utils", dataset_utils)
sys.modules.setdefault("vlmeval.smp", smp)
sys.modules.setdefault("vlmeval.utils", utils)

from zoombench_dataset import (  # noqa: E402
    extract_zoombench_answer,
    resolve_zoombench_columns,
    zoombench_deterministic_match,
    zoombench_judge_one,
)


class FakeJudge:
    def __init__(self, response="Yes"):
        self.response = response
        self.calls = []

    def generate(self, prompt):
        self.calls.append(prompt)
        return self.response


def test_resolve_current_and_documented_zoombench_schemas():
    assert resolve_zoombench_columns(
        ["id", "query", "response", "image", "crop_image"]
    ) == ("query", "response")
    assert resolve_zoombench_columns(
        ["id", "prompt", "answer", "image", "crop_image"]
    ) == ("prompt", "answer")


def test_extract_zoombench_answer_follows_official_precedence():
    assert extract_zoombench_answer("reason <answer>blue</answer> tail") == "blue"
    assert extract_zoombench_answer("reason\nAnswer: seven") == "seven"
    assert extract_zoombench_answer("a\nb\nc\nd") == "b\nc\nd"


def test_deterministic_match_handles_exact_text_and_mcq_letters():
    assert zoombench_deterministic_match("traffic light", "Traffic light.")
    assert zoombench_deterministic_match("(C)", "The answer is (C).")
    assert not zoombench_deterministic_match("(C)", "The answer is (D).")


def test_judge_is_only_called_after_deterministic_match_fails():
    judge = FakeJudge("Yes")
    exact = zoombench_judge_one(judge, "question", "blue", "Blue.")
    assert exact["score"] is True
    assert exact["judge_source"] == "deterministic"
    assert judge.calls == []

    semantic = zoombench_judge_one(judge, "question", "two", "a pair")
    assert semantic["score"] is True
    assert semantic["judge_source"] == "llm"
    assert len(judge.calls) == 1
