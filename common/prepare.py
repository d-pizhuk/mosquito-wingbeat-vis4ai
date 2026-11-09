"""CPU-only preparation: inspect WAVs, deduplicate inputs, split, and cache arrays.

This is a NEW experimental split, not a reproduction of either paper's results.
Existing completed datasets are immutable. No source code from the papers is copied.
"""
from __future__ import annotations
from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import shutil
import stat
import tempfile
import zipfile

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

from common.audio import decode_audio
from common.class_names import CLASS_NAMES, NUM_CLASSES, label_from_path
from common.config import DataConfig, SCHEMA_VERSION, fingerprint
from common.io_utils import atomic_json, emit, file_sha256


def safe_extract_zip(archive: str | Path, destination: str | Path,
                     max_uncompressed_bytes: int = 60 * 1024**3) -> None:
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as source:
        total = 0
        for item in source.infolist():
            name = item.filename.replace("\\", "/")
            target = (destination / name).resolve()
            if not target.is_relative_to(destination) or ":" in name:
                raise ValueError(f"Unsafe archive path: {item.filename}")
            if stat.S_ISLNK(item.external_attr >> 16):
                raise ValueError(f"Archive symlinks are not accepted: {item.filename}")
            total += item.file_size
            if total > max_uncompressed_bytes:
                raise ValueError("Archive exceeds the extraction size limit")
        source.extractall(destination)


def _decode_task(task: tuple) -> tuple:
    idx, path, config = task
    waveform, stats = decode_audio(path, config)
    return idx, waveform, stats


def split_manifest(frame: pd.DataFrame, config: DataConfig) -> pd.DataFrame:
    frame = frame.copy()
    labels = frame["label"].to_numpy()
    positions = np.arange(len(frame))
    if set(labels.tolist()) != set(range(NUM_CLASSES)):
        raise ValueError("All six species must be represented")
    if config.split_mode == "recording":
        trainval, test = train_test_split(positions, test_size=config.test_fraction,
                                        stratify=labels, random_state=config.seed)
        train, validation = train_test_split(
            trainval, test_size=config.validation_fraction / (1 - config.test_fraction),
            stratify=labels[trainval], random_state=config.seed + 1)
        frame["split"] = "train"
        frame.loc[validation, "split"] = "validation"
        frame.loc[test, "split"] = "test"
    else:
        if frame["group_id"].nunique() < 10:
            raise ValueError("Group splitting needs at least ten independent groups")
        # Group identities may span species; they must not cross any split.
        folds = np.empty(len(frame), dtype=np.int64)
        splitter = StratifiedGroupKFold(10, shuffle=True, random_state=config.seed)
        for fold, (_, held) in enumerate(splitter.split(positions, labels, frame["group_id"])):
            folds[held] = fold
        frame["split"] = np.where(folds < 2, "test", np.where(folds == 2, "validation", "train"))
    for split in ("train", "validation", "test"):
        part = frame[frame["split"] == split]
        if set(part["label"].tolist()) != set(range(NUM_CLASSES)):
            raise ValueError(f"{split} does not contain all classes; supply more data/groups")
    if frame.groupby("waveform_sha256")["split"].nunique().max() != 1:
        raise AssertionError("Exact duplicate input leaked across splits")
    if config.split_mode == "group" and frame.groupby("group_id")["split"].nunique().max() != 1:
        raise AssertionError("Group leaked across splits")
    return frame


def prepare_dataset(raw_root: str | Path, output: str | Path, config: DataConfig,
                    source: dict | None = None, groups_csv: str | Path | None = None) -> dict:
    raw_root, output = Path(raw_root).resolve(), Path(output).resolve()
    source = source or {"kind": "local", "root": str(raw_root)}
    if output.exists():
        marker = output / "metadata.json"
        if not marker.is_file():
            raise FileExistsError(f"Incomplete dataset at {output}; use a new output directory")
        metadata = json.loads(marker.read_text())
        if metadata["data_config"] != config.to_dict():
            raise ValueError("Prepared dataset configuration differs; create a new preparation ID")
        return metadata
    wavs = sorted(p for p in raw_root.rglob("*") if p.is_file() and p.suffix.lower() == ".wav")
    if not wavs:
        raise FileNotFoundError(f"No .wav files found under {raw_root}")
    paths = [p.relative_to(raw_root).as_posix() for p in wavs]
    labels = [label_from_path(p) for p in paths]
    group_map: dict[str, str] = {}
    if config.split_mode == "group":
        if groups_csv is None:
            raise ValueError("Group splitting requires a CSV with relative_path and group_id")
        groups = pd.read_csv(groups_csv, dtype=str, keep_default_na=False)
        if not {"relative_path", "group_id"} <= set(groups.columns):
            raise ValueError("Group CSV columns must include relative_path, group_id")
        if groups["relative_path"].duplicated().any() or (groups["group_id"].str.strip() == "").any():
            raise ValueError("Group CSV has duplicate paths or empty group IDs")
        group_map = dict(zip(groups["relative_path"].str.replace("\\", "/", regex=False), groups["group_id"]))
        missing = set(paths) - group_map.keys()
        if missing:
            raise ValueError(f"Missing group metadata for {len(missing)} files, e.g. {next(iter(missing))}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=output.name + ".building-", dir=output.parent))
    emit("prepare_start", files=len(wavs), samples_per_recording=config.audio.num_samples)
    waveform_file = staging / "waveforms.npy"
    cache = np.lib.format.open_memmap(waveform_file, mode="w+", dtype=np.float32,
                                     shape=(len(wavs), config.audio.num_samples))
    records, duplicate_records = [], []
    seen: dict[str, tuple[int, str]] = {}
    try:
        executor_context = ProcessPoolExecutor(max_workers=config.decode_workers,
                                              mp_context=mp.get_context("spawn")) if config.decode_workers else nullcontext(None)
        with executor_context as executor:
            # Submit bounded blocks, never all 280k decode futures simultaneously.
            for start in range(0, len(wavs), 1024):
                tasks = [(i, str(wavs[i]), config.audio) for i in range(start, min(start + 1024, len(wavs)))]
                decoded = executor.map(_decode_task, tasks, chunksize=16) if executor else map(_decode_task, tasks)
                for i, waveform, stats in decoded:
                    cache[i] = waveform
                    digest = stats["waveform_sha256"]
                    if digest in seen:
                        old_label, old_path = seen[digest]
                        if old_label != labels[i]:
                            raise ValueError(f"Identical model inputs have different labels: {old_path}, {paths[i]}")
                        duplicate_records.append({"relative_path": paths[i], "kept_path": old_path,
                                                  "waveform_sha256": digest})
                        continue
                    seen[digest] = (labels[i], paths[i])
                    sample_id = hashlib.sha256((paths[i] + ":" + digest).encode()).hexdigest()[:24]
                    records.append({"row_idx": i, "sample_id": sample_id, "relative_path": paths[i],
                                    "label": labels[i], "species": CLASS_NAMES[labels[i]],
                                    "group_id": group_map.get(paths[i], sample_id), **stats})
                emit("prepare_progress", decoded=min(start + 1024, len(wavs)), total=len(wavs),
                     duplicates=len(duplicate_records))
        cache.flush()
        del cache
        manifest = split_manifest(pd.DataFrame(records).reset_index(drop=True), config)
        manifest.to_csv(staging / "manifest.csv", index=False)
        pd.DataFrame(duplicate_records, columns=["relative_path", "kept_path", "waveform_sha256"]).to_csv(
            staging / "duplicates.csv", index=False)
        counts = {split: {CLASS_NAMES[i]: int(n) for i, n in group["label"].value_counts().sort_index().items()}
                  for split, group in manifest.groupby("split")}
        manifest_hash = file_sha256(staging / "manifest.csv")
        metadata = {
            "schema_version": SCHEMA_VERSION, "data_config": config.to_dict(), "source": source,
            "class_names": list(CLASS_NAMES), "recordings_discovered": len(wavs),
            "recordings_retained": len(manifest), "duplicates_removed": len(duplicate_records),
            "waveform_array_shape": [len(wavs), config.audio.num_samples],
            "waveform_dtype": "float32", "waveform_file_bytes": waveform_file.stat().st_size,
            "waveform_file_sha256": file_sha256(waveform_file), "manifest_sha256": manifest_hash,
            "dataset_fingerprint": fingerprint({"config": config.to_dict(), "manifest": manifest_hash}),
            "class_counts": counts,
            "split_scope": "recording-level; not an unseen-individual/session guarantee" if config.split_mode == "recording"
                           else "group-disjoint according to the supplied metadata",
            "preparation": "new split; center crop/zero pad; DC removal; exact canonical-input deduplication",
        }
        atomic_json(staging / "metadata.json", metadata)
        staging.rename(output)
        emit("prepare_complete", fingerprint=metadata["dataset_fingerprint"], counts=counts)
        return metadata
    except BaseException:
        # A completed dataset is published only after all files pass preparation.
        if "cache" in locals():
            del cache
        shutil.rmtree(staging, ignore_errors=True)
        raise
