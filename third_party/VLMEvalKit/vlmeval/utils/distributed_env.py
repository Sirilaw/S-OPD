import os
from contextlib import contextmanager

# A local model replica launched by torchrun must not pass the outer process
# group's rendezvous identity to subprocesses created by inference backends
# such as vLLM. CUDA_VISIBLE_DEVICES is intentionally preserved.
TORCHRUN_ENV_KEYS = (
    'RANK',
    'WORLD_SIZE',
    'LOCAL_RANK',
    'LOCAL_WORLD_SIZE',
    'GROUP_RANK',
    'ROLE_RANK',
    'ROLE_WORLD_SIZE',
    'MASTER_ADDR',
    'MASTER_PORT',
    'TORCHELASTIC_RESTART_COUNT',
    'TORCHELASTIC_MAX_RESTARTS',
    'TORCHELASTIC_RUN_ID',
    'TORCHELASTIC_USE_AGENT_STORE',
)


@contextmanager
def isolated_model_build_environment():
    """Hide the outer torchrun rendezvous while constructing a model replica."""
    saved = {key: os.environ.pop(key) for key in TORCHRUN_ENV_KEYS if key in os.environ}
    try:
        yield
    finally:
        os.environ.update(saved)
