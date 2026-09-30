import os
import sys
from pathlib import Path

import pytest

VLMEVALKIT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(VLMEVALKIT_ROOT))

from vlmeval.utils.distributed_env import (  # noqa: E402
    TORCHRUN_ENV_KEYS,
    isolated_model_build_environment,
)


def test_model_build_environment_hides_and_restores_torchrun_state(monkeypatch):
    original = {key: f'value-{key}' for key in TORCHRUN_ENV_KEYS}
    for key, value in original.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '4')

    with isolated_model_build_environment():
        assert all(key not in os.environ for key in TORCHRUN_ENV_KEYS)
        assert os.environ['CUDA_VISIBLE_DEVICES'] == '4'

    assert {key: os.environ[key] for key in TORCHRUN_ENV_KEYS} == original


def test_model_build_environment_restores_state_after_failure(monkeypatch):
    monkeypatch.setenv('RANK', '3')

    with pytest.raises(RuntimeError, match='model init failed'):
        with isolated_model_build_environment():
            raise RuntimeError('model init failed')

    assert os.environ['RANK'] == '3'
