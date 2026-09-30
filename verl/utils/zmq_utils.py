"""Helpers for collision-free ZeroMQ IPC endpoints."""

import hashlib
import os


def get_zmq_handle(device_uuid: str) -> str:
    """Return a per-user, per-run IPC address for colocated weight transfer."""
    namespace = os.getenv("VERL_ZMQ_NAMESPACE", f"uid-{os.getuid()}")
    namespace_hash = hashlib.sha256(namespace.encode()).hexdigest()[:12]
    return f"ipc:///tmp/rl-colocate-zmq-{os.getuid()}-{namespace_hash}-{device_uuid}.sock"
