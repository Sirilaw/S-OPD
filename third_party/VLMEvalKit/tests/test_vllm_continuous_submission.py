import sys
from pathlib import Path

import pandas as pd

VLMEVALKIT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(VLMEVALKIT_ROOT))

from vlmeval.inference import infer_data  # noqa: E402
from vlmeval.smp import load  # noqa: E402


class StubDataset:
    dataset_name = "StubDataset"

    def __init__(self, count):
        self.data = pd.DataFrame({"index": list(range(count))})

    def build_prompt(self, row):
        return [{"type": "text", "value": str(row["index"])}]

    def dump_image(self, line):
        del line
        return []

    def __len__(self):
        return len(self.data)


class StubVllmModel:
    use_vllm = True

    def __init__(self):
        self.batch_lengths = []

    def set_dump_image(self, dump_image):
        self.dump_image = dump_image

    def generate_batch(self, messages, dataset=None):
        del dataset
        self.batch_lengths.append(len(messages))
        return [message[0]["value"] for message in messages]


def test_zero_batch_size_submits_complete_rank_queue(monkeypatch, tmp_path):
    monkeypatch.setenv("VLMEVAL_VLLM_BATCH_SIZE", "0")
    model = StubVllmModel()
    dataset = StubDataset(137)
    output = tmp_path / "rank-output.pkl"

    infer_data(
        model=model,
        model_name="stub-model",
        work_dir=str(tmp_path),
        dataset=dataset,
        out_file=str(output),
    )

    assert model.batch_lengths == [137]
    assert load(str(output)) == {i: str(i) for i in range(137)}
