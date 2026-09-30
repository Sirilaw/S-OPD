# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import shutil
import tempfile

import pytest


class TestCheckpointCleanupLogic:
    """Tests for checkpoint cleanup methods in BaseCheckpointManager."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up test fixtures."""
        self.test_dir = tempfile.mkdtemp()
        yield
        shutil.rmtree(self.test_dir, ignore_errors=True)

    @pytest.fixture
    def manager(self, monkeypatch):
        """Create a minimal BaseCheckpointManager for testing."""
        import torch.distributed

        monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
        monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 1)

        from verl.utils.checkpoint.checkpoint_manager import BaseCheckpointManager

        class MockModel:
            pass

        class MockOptimizer:
            pass

        return BaseCheckpointManager(
            model=MockModel(),
            optimizer=MockOptimizer(),
            lr_scheduler=None,
            processing_class=None,
            checkpoint_config=None,
        )

    def _create_checkpoint_dir(self, step: int) -> str:
        """Create a mock checkpoint directory."""
        path = os.path.join(self.test_dir, f"global_step_{step}")
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "checkpoint.txt"), "w") as f:
            f.write(f"step={step}")
        return path

    def test_max_ckpt_1_preserves_existing_before_save(self, manager):
        """
        Regression test: max_ckpt_to_keep=1 must NOT delete existing checkpoint before save.
        """
        ckpt_100 = self._create_checkpoint_dir(100)
        manager.previous_saved_paths = [ckpt_100]

        manager.ensure_checkpoint_capacity(max_ckpt_to_keep=1)

        assert os.path.exists(ckpt_100), "Bug: checkpoint deleted before save!"
        assert manager.previous_saved_paths == [ckpt_100]

    def test_max_ckpt_1_deletes_old_after_save(self, manager):
        """After save succeeds, old checkpoint should be deleted."""
        ckpt_100 = self._create_checkpoint_dir(100)
        manager.previous_saved_paths = [ckpt_100]

        ckpt_200 = self._create_checkpoint_dir(200)
        manager.register_checkpoint(ckpt_200, max_ckpt_to_keep=1)

        assert not os.path.exists(ckpt_100)
        assert os.path.exists(ckpt_200)
        assert manager.previous_saved_paths == [ckpt_200]

    def test_max_ckpt_2_keeps_one_before_save(self, manager):
        """With max_ckpt_to_keep=2, pre-save cleanup keeps 1 checkpoint."""
        ckpt_100 = self._create_checkpoint_dir(100)
        ckpt_200 = self._create_checkpoint_dir(200)
        manager.previous_saved_paths = [ckpt_100, ckpt_200]

        manager.ensure_checkpoint_capacity(max_ckpt_to_keep=2)

        assert not os.path.exists(ckpt_100)
        assert os.path.exists(ckpt_200)
        assert len(manager.previous_saved_paths) == 1

    def test_max_ckpt_0_keeps_all(self, manager):
        """max_ckpt_to_keep=0 means unlimited - no deletions."""
        ckpt_100 = self._create_checkpoint_dir(100)
        ckpt_200 = self._create_checkpoint_dir(200)
        manager.previous_saved_paths = [ckpt_100, ckpt_200]

        manager.ensure_checkpoint_capacity(max_ckpt_to_keep=0)
        ckpt_300 = self._create_checkpoint_dir(300)
        manager.register_checkpoint(ckpt_300, max_ckpt_to_keep=0)

        assert os.path.exists(ckpt_100)
        assert os.path.exists(ckpt_200)
        assert os.path.exists(ckpt_300)
        assert len(manager.previous_saved_paths) == 3

    def test_full_save_cycle_max_ckpt_1(self, manager):
        """Simulate multiple save cycles with max_ckpt_to_keep=1."""
        # First save
        manager.ensure_checkpoint_capacity(1)
        ckpt_100 = self._create_checkpoint_dir(100)
        manager.register_checkpoint(ckpt_100, 1)
        assert manager.previous_saved_paths == [ckpt_100]

        # Second save - existing checkpoint must survive pre-save
        manager.ensure_checkpoint_capacity(1)
        assert os.path.exists(ckpt_100), "Bug: checkpoint deleted before save!"

        ckpt_200 = self._create_checkpoint_dir(200)
        manager.register_checkpoint(ckpt_200, 1)
        assert not os.path.exists(ckpt_100)
        assert manager.previous_saved_paths == [ckpt_200]

        # Third save
        manager.ensure_checkpoint_capacity(1)
        assert os.path.exists(ckpt_200), "Bug: checkpoint deleted before save!"

        ckpt_300 = self._create_checkpoint_dir(300)
        manager.register_checkpoint(ckpt_300, 1)
        assert not os.path.exists(ckpt_200)
        assert manager.previous_saved_paths == [ckpt_300]

    def test_cleanup_removes_data_with_rotated_actor_checkpoint(self):
        from verl.utils.checkpoint.checkpoint_manager import cleanup_global_step_dirs_without_actor

        stale = self._create_checkpoint_dir(100)
        with open(os.path.join(stale, "data.pt"), "w") as f:
            f.write("dataloader state")

        retained = self._create_checkpoint_dir(200)
        os.makedirs(os.path.join(retained, "actor"))
        with open(os.path.join(retained, "data.pt"), "w") as f:
            f.write("dataloader state")

        removed = cleanup_global_step_dirs_without_actor(self.test_dir)

        assert removed == [stale]
        assert not os.path.exists(stale)
        assert os.path.isfile(os.path.join(retained, "data.pt"))

    def test_old_actor_checkpoints_are_reduced_to_hf_only(self):
        from verl.utils.checkpoint.checkpoint_manager import prune_actor_checkpoints_to_hf_only

        def create_full_checkpoint(step: int):
            checkpoint = self._create_checkpoint_dir(step)
            actor = os.path.join(checkpoint, "actor")
            hf = os.path.join(actor, "huggingface")
            os.makedirs(hf)
            for filename in (
                "model_world_size_2_rank_0.pt",
                "optim_world_size_2_rank_0.pt",
                "extra_state_world_size_2_rank_0.pt",
                "fsdp_config.json",
            ):
                with open(os.path.join(actor, filename), "w") as f:
                    f.write(filename)
            with open(os.path.join(hf, "model.safetensors"), "w") as f:
                f.write("hf model")
            with open(os.path.join(checkpoint, "data.pt"), "w") as f:
                f.write("dataloader state")
            return checkpoint

        old = create_full_checkpoint(100)
        latest = create_full_checkpoint(200)

        removed = prune_actor_checkpoints_to_hf_only(self.test_dir, max_full_to_keep=1)

        assert removed
        assert os.listdir(os.path.join(old, "actor")) == ["huggingface"]
        assert os.path.isfile(os.path.join(old, "actor", "huggingface", "model.safetensors"))
        assert not os.path.exists(os.path.join(old, "data.pt"))
        assert os.path.isfile(os.path.join(latest, "actor", "model_world_size_2_rank_0.pt"))
        assert os.path.isfile(os.path.join(latest, "actor", "optim_world_size_2_rank_0.pt"))
        assert os.path.isfile(os.path.join(latest, "data.pt"))

    def test_hf_only_pruning_skips_checkpoint_without_hf_export(self):
        from verl.utils.checkpoint.checkpoint_manager import prune_actor_checkpoints_to_hf_only

        old = self._create_checkpoint_dir(100)
        actor = os.path.join(old, "actor")
        os.makedirs(actor)
        shard = os.path.join(actor, "model_world_size_1_rank_0.pt")
        with open(shard, "w") as f:
            f.write("only model copy")
        latest = self._create_checkpoint_dir(200)
        os.makedirs(os.path.join(latest, "actor", "huggingface"))

        removed = prune_actor_checkpoints_to_hf_only(self.test_dir, max_full_to_keep=1)

        assert removed == []
        assert os.path.isfile(shard)

    def test_hf_frequency_keeps_milestones_and_latest_full_checkpoint(self):
        from verl.utils.checkpoint.checkpoint_manager import prune_actor_checkpoints_to_hf_only

        def create_full_checkpoint(step: int):
            checkpoint = self._create_checkpoint_dir(step)
            actor = os.path.join(checkpoint, "actor")
            hf = os.path.join(actor, "huggingface")
            os.makedirs(hf)
            with open(os.path.join(actor, "model_world_size_1_rank_0.pt"), "w") as f:
                f.write("sharded model")
            with open(os.path.join(actor, "optim_world_size_1_rank_0.pt"), "w") as f:
                f.write("optimizer")
            with open(os.path.join(hf, "model.safetensors"), "w") as f:
                f.write("hf model")
            with open(os.path.join(checkpoint, "data.pt"), "w") as f:
                f.write("dataloader")
            return checkpoint

        non_milestones = [create_full_checkpoint(step) for step in (10, 20, 30, 40)]
        milestone = create_full_checkpoint(50)
        latest = create_full_checkpoint(60)

        removed = prune_actor_checkpoints_to_hf_only(
            self.test_dir,
            max_full_to_keep=1,
            huggingface_save_freq=50,
        )

        assert all(not os.path.exists(path) for path in non_milestones)
        assert milestone not in removed
        assert os.listdir(os.path.join(milestone, "actor")) == ["huggingface"]
        assert not os.path.exists(os.path.join(milestone, "data.pt"))
        assert os.path.isfile(os.path.join(latest, "actor", "model_world_size_1_rank_0.pt"))
        assert os.path.isfile(os.path.join(latest, "actor", "optim_world_size_1_rank_0.pt"))
        assert os.path.isfile(os.path.join(latest, "data.pt"))

    def test_best_checkpoint_snapshot_is_replaced_atomically(self):
        from verl.utils.checkpoint.checkpoint_manager import save_best_checkpoint_snapshot

        first = self._create_checkpoint_dir(100)
        os.makedirs(os.path.join(first, "actor"))
        with open(os.path.join(first, "actor", "model.pt"), "w") as f:
            f.write("first")
        best = save_best_checkpoint_snapshot(
            self.test_dir, first, step=100, metric_name="val/acc", metric_value=0.5
        )

        second = self._create_checkpoint_dir(200)
        os.makedirs(os.path.join(second, "actor"))
        with open(os.path.join(second, "actor", "model.pt"), "w") as f:
            f.write("second")
        save_best_checkpoint_snapshot(self.test_dir, second, step=200, metric_name="val/acc", metric_value=0.8)

        with open(os.path.join(best, "actor", "model.pt")) as f:
            assert f.read() == "second"
        with open(os.path.join(best, "best_checkpoint_info.json")) as f:
            import json

            assert json.load(f) == {
                "global_step": 200,
                "metric_name": "val/acc",
                "metric_value": 0.8,
            }
