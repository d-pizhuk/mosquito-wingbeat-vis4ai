"""Atomic tensor-only checkpoints; do not unpickle arbitrary downloaded code."""
from __future__ import annotations
import os
from pathlib import Path
import random
import tempfile

import numpy as np
import torch

from common.class_names import CLASS_NAMES
from common.config import SCHEMA_VERSION


def atomic_torch_save(path: str | Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def load_checkpoint(path: str | Path) -> dict:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Checkpoint schema is not supported")
    if checkpoint.get("class_names") != list(CLASS_NAMES):
        raise ValueError("Checkpoint class order does not match this project")
    return checkpoint


def rng_state() -> dict:
    numpy_state = np.random.get_state()
    return {"python": random.getstate(),
            "numpy": [numpy_state[0], numpy_state[1].tolist(), int(numpy_state[2]),
                      int(numpy_state[3]), float(numpy_state[4])],
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state: dict) -> None:
    random.setstate(state["python"])
    name, keys, position, has_gauss, cached_gaussian = state["numpy"]
    np.random.set_state((name, np.asarray(keys, dtype=np.uint32), position, has_gauss, cached_gaussian))
    torch.set_rng_state(state["torch_cpu"])
    if state["torch_cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])
