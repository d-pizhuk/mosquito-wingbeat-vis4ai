"""Read-only memory-mapped data and reproducible sample selection."""
from __future__ import annotations
import json
from pathlib import Path
import random
import shutil
import tempfile

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from common.class_names import CLASS_NAMES
from common.config import DataConfig
from common.io_utils import emit, file_sha256


def read_metadata(directory: str | Path) -> dict:
    directory = Path(directory)
    metadata = json.loads((directory / "metadata.json").read_text())
    if metadata.get("schema_version") != 1 or metadata.get("class_names") != list(CLASS_NAMES):
        raise ValueError("Unsupported dataset schema or class order")
    DataConfig.from_dict(metadata["data_config"])
    if (directory / "waveforms.npy").stat().st_size != metadata["waveform_file_bytes"]:
        raise ValueError("Waveform cache is incomplete or has changed")
    if file_sha256(directory / "manifest.csv") != metadata["manifest_sha256"]:
        raise ValueError("Manifest checksum does not match the prepared dataset")
    return metadata


def stage_locally(directory: str | Path, cache_root: str | Path = "/tmp/wingbeat-cache") -> Path:
    """One sequential copy avoids many small reads against a remote Volume."""
    directory = Path(directory)
    metadata = read_metadata(directory)
    root = Path(cache_root)
    root.mkdir(parents=True, exist_ok=True)
    target = root / metadata["dataset_fingerprint"]
    if target.exists():
        read_metadata(target)
        return target
    free = shutil.disk_usage(root).free
    if free < metadata["waveform_file_bytes"] * 1.1:
        raise OSError("Not enough local scratch disk to stage the waveform cache")
    temporary = Path(tempfile.mkdtemp(prefix="stage-", dir=root))
    try:
        emit("stage_dataset_start", bytes=metadata["waveform_file_bytes"])
        for name in ("waveforms.npy", "manifest.csv", "metadata.json", "duplicates.csv"):
            shutil.copyfile(directory / name, temporary / name)
        if file_sha256(temporary / "waveforms.npy") != metadata["waveform_file_sha256"]:
            raise ValueError("Local waveform copy failed checksum verification")
        temporary.rename(target)
        emit("stage_dataset_complete", directory=str(target))
        return target
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def select_rows(manifest: pd.DataFrame, split: str, limit: int | None, seed: int) -> np.ndarray:
    """Proportional stratified subset of one split, retaining every species."""
    if split not in {"train", "validation", "test"}:
        raise ValueError("Unknown split")
    part = manifest.loc[manifest["split"] == split]
    if part.empty:
        raise ValueError(f"Empty {split} split")
    if limit is None or limit >= len(part):
        return part["row_idx"].to_numpy(dtype=np.int64)
    counts = part.groupby("label").size().sort_index()
    if limit < len(counts):
        raise ValueError("The subset is too small to include every class")
    desired = counts.to_numpy() * limit / len(part)
    quotas = np.minimum(counts.to_numpy(), np.maximum(1, np.floor(desired).astype(int)))
    while quotas.sum() > limit:
        eligible = np.where(quotas > 1)[0]
        quotas[eligible[np.argmax(quotas[eligible] - desired[eligible])]] -= 1
    while quotas.sum() < limit:
        eligible = np.where(quotas < counts.to_numpy())[0]
        quotas[eligible[np.argmax(desired[eligible] - quotas[eligible])]] += 1
    rng = np.random.default_rng(seed)
    chosen = []
    for label, quota in zip(counts.index, quotas):
        candidates = part.loc[part["label"] == label, "row_idx"].to_numpy(dtype=np.int64)
        chosen.extend(rng.choice(candidates, int(quota), replace=False).tolist())
    return np.sort(np.asarray(chosen, dtype=np.int64))


class WaveformDataset(Dataset):
    def __init__(self, directory: str | Path, manifest: pd.DataFrame, rows: np.ndarray):
        self.directory = Path(directory)
        self.rows = np.asarray(rows, dtype=np.int64)
        indexed = manifest.set_index("row_idx")
        self.labels = indexed.loc[self.rows, "label"].to_numpy(dtype=np.int64)
        self._waveforms = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_waveforms"] = None  # Workers open their own read-only map.
        return state

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        if self._waveforms is None:
            self._waveforms = np.load(self.directory / "waveforms.npy", mmap_mode="r", allow_pickle=False)
        row = int(self.rows[index])
        # Copy prevents read-only-buffer warnings and accidental cache mutation.
        waveform = torch.from_numpy(np.array(self._waveforms[row], dtype=np.float32, copy=True))
        return waveform, int(self.labels[index]), row


def _seed_worker(worker_id: int) -> None:
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)
    torch.set_num_threads(1)


def make_loader(directory: str | Path, manifest: pd.DataFrame, rows: np.ndarray, *,
                batch_size: int, num_workers: int, seed: int, epoch: int = 0,
                shuffle: bool = False, start_batch: int = 0, cuda: bool = False) -> DataLoader:
    ordered = np.asarray(rows, dtype=np.int64).copy()
    if shuffle:
        ordered = ordered[np.random.default_rng(seed + epoch).permutation(len(ordered))]
    if start_batch < 0:
        raise ValueError("start_batch cannot be negative")
    ordered = ordered[start_batch * batch_size:]
    generator = torch.Generator().manual_seed(seed + epoch)
    kwargs = {}
    if num_workers:
        kwargs.update(multiprocessing_context="spawn", prefetch_factor=2, persistent_workers=False)
    return DataLoader(WaveformDataset(directory, manifest, ordered), batch_size=batch_size,
                      shuffle=False, drop_last=False, num_workers=num_workers, pin_memory=cuda,
                      worker_init_fn=_seed_worker, generator=generator, **kwargs)
