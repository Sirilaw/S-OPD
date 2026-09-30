import asyncio
import json
import pickle
import subprocess
import sys
from pathlib import Path

import pandas as pd

VLMEVALKIT_ROOT = Path(__file__).resolve().parents[2] / "third_party" / "VLMEvalKit"
sys.path.insert(0, str(VLMEVALKIT_ROOT))

from vlmeval.inference import _infer_vllm_dynamic, infer_data_job  # noqa: E402
from vlmeval.smp import load  # noqa: E402
from vlmeval.utils.workload_balance import (  # noqa: E402
    FileTaskQueue,
    balanced_partitions,
    load_output_token_history,
    order_rows_longest_first,
    write_output_token_history,
)


class _FakeDataset:
    force_use_dataset_prompt = True
    dataset_name = "fake"

    def __init__(self, total):
        self.data = pd.DataFrame({"index": list(range(total)), "question": [f"q{i}" for i in range(total)]})

    def build_prompt(self, row):
        return [{"type": "text", "value": row["question"]}]

    def dump_image(self, row):
        del row
        return []

    def __len__(self):
        return len(self.data)


class _FakeAsyncModel:
    max_num_seqs = 3
    use_vllm = True
    use_vllm_async = True

    def __init__(self):
        self.active = 0
        self.max_active = 0

    def set_dump_image(self, dump_image):
        self.dump_image = dump_image

    async def generate_async(self, struct, dataset=None, request_id=None):
        del dataset, request_id
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        await asyncio.sleep(0.001)
        self.active -= 1
        return f"answer:{struct[0]['value']}"


def test_history_drives_longest_first_order():
    rows = [
        {"index": "short", "question": "x", "image": "a.jpg"},
        {"index": "long", "question": "x", "image": "b.jpg"},
        {"index": "medium", "question": "x", "image": "c.jpg"},
    ]
    history = {"short": 4, "long": 400, "medium": 40}

    assert order_rows_longest_first(rows, history) == [1, 2, 0]


def test_balanced_partitions_cover_every_row_and_reduce_peak_load():
    rows = [{"index": i, "question": "x" * chars} for i, chars in enumerate([400, 360, 320, 40, 40, 40])]
    partitions, loads = balanced_partitions(rows, world_size=2)

    assigned = [position for partition in partitions for position in partition]
    assert sorted(assigned) == list(range(len(rows)))
    assert all(partition == sorted(partition, key=lambda pos: -len(rows[pos]["question"])) for partition in partitions)

    round_robin_loads = [
        sum(len(rows[position]["question"]) / 4 for position in range(rank, len(rows), 2))
        for rank in range(2)
    ]
    assert max(loads) < max(round_robin_loads)


def test_workload_profile_round_trip_is_lossless(tmp_path):
    path = write_output_token_history(tmp_path, "MMMU/DEV VAL", {1: 17, "case-2": 999})

    assert path is not None and path.is_file()
    assert load_output_token_history(tmp_path, "MMMU/DEV VAL") == {"1": 17, "case-2": 999}


def test_file_task_queue_claims_each_position_once_across_processes(tmp_path):
    total = 137
    queue_path = tmp_path / "queue.state"
    FileTaskQueue(queue_path, total).initialize()
    child_files = [tmp_path / f"claims-{worker}.json" for worker in range(4)]
    worker_code = """
import importlib.util
import json
import sys

spec = importlib.util.spec_from_file_location("workload_balance_standalone", sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
queue = module.FileTaskQueue(sys.argv[2], int(sys.argv[3]))
claimed = []
while True:
    position = queue.claim()
    if position is None:
        break
    claimed.append(position)
with open(sys.argv[4], "w", encoding="utf-8") as handle:
    json.dump(claimed, handle)
"""
    helper_path = VLMEVALKIT_ROOT / "vlmeval" / "utils" / "workload_balance.py"
    children = []
    for child_file in child_files:
        children.append(subprocess.Popen([
            sys.executable,
            "-c",
            worker_code,
            str(helper_path),
            str(queue_path),
            str(total),
            str(child_file),
        ]))

    for child in children:
        assert child.wait(timeout=10) == 0
    claims = []
    for child_file in child_files:
        claims.extend(json.loads(child_file.read_text(encoding="utf-8")))

    assert sorted(claims) == list(range(total))


def test_dynamic_scheduler_refills_async_slots_and_checkpoints_losslessly(tmp_path, monkeypatch):
    total = 11
    dataset = _FakeDataset(total)
    model = _FakeAsyncModel()
    queue_path = tmp_path / "dynamic.queue"
    out_file = tmp_path / "rank.pkl"
    FileTaskQueue(queue_path, total).initialize(start_position=3)
    monkeypatch.setenv("VLMEVAL_DYNAMIC_INFLIGHT_PER_GPU", "3")
    monkeypatch.setenv("VLMEVAL_DYNAMIC_CHECKPOINT_INTERVAL", "2")

    result = asyncio.run(_infer_vllm_dynamic(
        model=model,
        dataset=dataset,
        dataset_name="fake",
        data=dataset.data,
        ordered_positions=list(range(total)),
        out_file=str(out_file),
        queue_path=str(queue_path),
        rank=0,
        world_size=1,
    ))

    with out_file.open("rb") as handle:
        checkpoint = pickle.load(handle)
    expected = {index: f"answer:q{index}" for index in range(total)}
    assert result == expected
    assert checkpoint == expected
    assert model.max_active == 3


def test_dynamic_infer_job_merges_lossless_json_and_writes_profile(tmp_path, monkeypatch):
    dataset = _FakeDataset(9)
    model = _FakeAsyncModel()
    profile_dir = tmp_path / "profiles"
    monkeypatch.setenv("PRED_FORMAT", "json")
    monkeypatch.setenv("VLMEVAL_VLLM_GLOBAL_QUEUE", "1")
    monkeypatch.setenv("VLMEVAL_DYNAMIC_INFLIGHT_PER_GPU", "3")
    monkeypatch.setenv("VLMEVAL_WORKLOAD_PROFILE_DIR", str(profile_dir))

    infer_data_job(
        model=model,
        work_dir=str(tmp_path),
        model_name="fake-model",
        dataset=dataset,
        # VLMEvalKit's legacy CLI flag is False for config-defined vLLM
        # models. The runner-set global queue flag must still select async.
        use_vllm=False,
    )

    result_files = list(tmp_path.glob("fake-model_fake.json"))
    assert len(result_files) == 1
    result = load(str(result_files[0]))
    assert result["prediction"].tolist() == [f"answer:q{index}" for index in range(9)]
    assert load_output_token_history(profile_dir, "fake") == {
        str(index): max(1, (len(f"answer:q{index}") + 3) // 4)
        for index in range(9)
    }
    assert not list(tmp_path.glob("*.dynamic_queue"))
