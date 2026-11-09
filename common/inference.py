"""Portable checkpoint loading, held-out evaluation and native attention exports."""
from __future__ import annotations
import csv
import json
from pathlib import Path
import time

import numpy as np
import librosa
import pandas as pd
import torch
from torch.nn import functional as F

from common.audio import decode_audio
from common.class_names import CLASS_NAMES, CLASS_SLUGS
from common.config import DataConfig, TrainConfig
from common.io_utils import atomic_json, emit, file_sha256
from common.load_data import make_loader, read_metadata, select_rows
from common.model import build_model
from training.checkpoints import load_checkpoint
from training.engine import amp_context, resolve_device
from training.metrics import Meter


def load_model(checkpoint_path: str | Path, device: str | None = None):
    checkpoint = load_checkpoint(checkpoint_path)
    device_obj = resolve_device(device)
    data = DataConfig.from_dict(checkpoint["data_config"])
    config = TrainConfig.from_dict(checkpoint["train_config"])
    # Loading a trained checkpoint never downloads ImageNet weights again.
    model = build_model(data.audio, config.model, load_imagenet=False).to(device_obj)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    return model, checkpoint, device_obj


@torch.inference_mode()
def predict_wav(checkpoint_path: str | Path, wav_path: str | Path, device: str | None = None) -> dict:
    model, checkpoint, device_obj = load_model(checkpoint_path, device)
    waveform, stats = decode_audio(wav_path, model.audio_config)
    probabilities = model(torch.from_numpy(waveform).unsqueeze(0).to(device_obj)).softmax(dim=1)[0].cpu().numpy()
    predicted = int(probabilities.argmax())
    return {"predicted_class": predicted, "predicted_species": CLASS_NAMES[predicted],
            "confidence": float(probabilities[predicted]),
            "probabilities": dict(zip(CLASS_NAMES, probabilities.astype(float).tolist())),
            "audio": stats, "checkpoint_epoch": checkpoint.get("epoch")}


@torch.inference_mode()
def evaluate_checkpoint(checkpoint_path: str | Path, dataset_directory: str | Path,
                        output_directory: str | Path, *, split: str = "test", limit: int | None = None,
                        batch_size: int = 128, num_workers: int = 4, device: str | None = None,
                        max_seconds: float = 3600) -> dict:
    started = time.monotonic()
    directory, output = Path(dataset_directory), Path(output_directory)
    metadata = read_metadata(directory)
    model, checkpoint, device_obj = load_model(checkpoint_path, device)
    if checkpoint["dataset_fingerprint"] != metadata["dataset_fingerprint"]:
        raise ValueError("Checkpoint and evaluation split do not belong to the same prepared dataset")
    config = TrainConfig.from_dict(checkpoint["train_config"])
    manifest = pd.read_csv(directory / "manifest.csv", keep_default_na=False)
    rows = select_rows(manifest, split, limit, config.seed + 2)
    # Re-read provenance, not just labels. Never export an augmented input.
    indexed = manifest.set_index("row_idx")
    loader = make_loader(directory, manifest, rows, batch_size=batch_size, num_workers=num_workers,
                         seed=config.seed + 2, cuda=device_obj.type == "cuda")
    output.mkdir(parents=True, exist_ok=True)
    temporary = output / "predictions.csv.partial"
    columns = ["sample_id", "row_idx", "relative_path", "split", "true_class", "true_species",
               "predicted_class", "predicted_species", "correct", "confidence", "entropy",
               "raw_rms", "spectral_peak_hz"] + [f"prob_{slug}" for slug in CLASS_SLUGS]
    meter = Meter()
    try:
        with temporary.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            for batch, (waveform, target, row_indices) in enumerate(loader):
                if time.monotonic() - started > max_seconds:
                    raise TimeoutError("Evaluation time limit reached; partial CSV was not published")
                waveform, target = waveform.to(device_obj, non_blocking=True), target.to(device_obj, non_blocking=True)
                with amp_context(device_obj, config.precision):
                    logits = model(waveform)
                    losses = F.cross_entropy(logits, target, reduction="none")
                probabilities = logits.float().softmax(dim=1).cpu().numpy()
                if not np.isfinite(probabilities).all() or not np.allclose(probabilities.sum(1), 1, atol=1e-5):
                    raise FloatingPointError("Invalid model probabilities")
                meter.update(logits, target, losses)
                for row_idx, probability in zip(row_indices.tolist(), probabilities):
                    original = indexed.loc[row_idx]
                    predicted = int(probability.argmax())
                    record = {"sample_id": original["sample_id"], "row_idx": row_idx,
                              "relative_path": original["relative_path"], "split": split,
                              "true_class": int(original["label"]), "true_species": original["species"],
                              "predicted_class": predicted, "predicted_species": CLASS_NAMES[predicted],
                              "correct": predicted == int(original["label"]), "confidence": float(probability[predicted]),
                              "entropy": float(-(probability * np.log(probability.clip(1e-12))).sum()),
                              "raw_rms": original["raw_rms"], "spectral_peak_hz": original["spectral_peak_hz"]}
                    record.update({f"prob_{slug}": float(p) for slug, p in zip(CLASS_SLUGS, probability)})
                    writer.writerow(record)
                if batch % 25 == 0:
                    emit("evaluate_progress", completed=meter.count, total=len(rows))
        if meter.count != len(rows):
            raise AssertionError("Evaluation count mismatch")
        temporary.replace(output / "predictions.csv")
    finally:
        temporary.unlink(missing_ok=True)
    metrics = {**meter.result(), "split": split, "checkpoint_epoch": checkpoint.get("epoch"),
               "checkpoint_sha256": file_sha256(checkpoint_path),
               "dataset_fingerprint": metadata["dataset_fingerprint"],
               "seconds": time.monotonic() - started,
               "warning": "Softmax confidence is uncalibrated; spectral_peak_hz is not necessarily the fundamental frequency."}
    atomic_json(output / "metrics.json", metrics)
    pd.DataFrame(metrics["confusion_matrix"], index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(output / "confusion_counts.csv")
    atomic_json(output / "class_mapping.json", {str(i): name for i, name in enumerate(CLASS_NAMES)})
    emit("evaluate_complete", samples=meter.count, accuracy=metrics["accuracy"], macro_f1=metrics["macro_f1"])
    return metrics


@torch.inference_mode()
def attention_examples(checkpoint_path: str | Path, dataset_directory: str | Path,
                       sample_ids: list[str], output_directory: str | Path,
                       device: str | None = None) -> dict:
    """Phase II utility. Saves raw gate values, not normalized display heatmaps.

    Spatial gates are not class-specific explanations or causal attributions.
    Retain native resolution; interpolation cannot create finer evidence.
    """
    if not 1 <= len(sample_ids) <= 64 or len(sample_ids) != len(set(sample_ids)):
        raise ValueError("Request 1-64 unique sample IDs")
    model, checkpoint, device_obj = load_model(checkpoint_path, device)
    metadata = read_metadata(dataset_directory)
    if checkpoint["dataset_fingerprint"] != metadata["dataset_fingerprint"]:
        raise ValueError("Dataset/checkpoint mismatch")
    frame = pd.read_csv(Path(dataset_directory) / "manifest.csv", keep_default_na=False).set_index("sample_id")
    unknown = set(sample_ids) - set(frame.index)
    if unknown:
        raise KeyError(f"Unknown sample IDs: {unknown}")
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    cache = np.load(Path(dataset_directory) / "waveforms.npy", mmap_mode="r", allow_pickle=False)
    records = []
    for sample_id in sample_ids:
        row = frame.loc[sample_id]
        waveform = torch.from_numpy(np.array(cache[int(row["row_idx"])], copy=True)).unsqueeze(0).to(device_obj)
        logits, maps = model(waveform, return_attention=True)
        arrays = {"log_mel_db": model.frontend.spectrogram(waveform)[0].cpu().numpy(),
                  "waveform": waveform[0].cpu().numpy(), "probabilities": logits.softmax(1)[0].cpu().numpy(),
                  "mel_centers_hz": librosa.mel_frequencies(n_mels=model.audio_config.n_mels + 2,
                      fmin=model.audio_config.f_min, fmax=model.audio_config.f_max, htk=False)[1:-1],
                  "frame_times_seconds": np.arange(model.audio_config.num_frames) * model.audio_config.hop_length / model.audio_config.sample_rate}
        for stage, gates in maps.items():
            arrays[f"{stage}_spatial"] = gates["spatial"][0, 0].float().cpu().numpy()
            arrays[f"{stage}_channel"] = gates["channel"][0, :, 0, 0].float().cpu().numpy()
        np.savez_compressed(output / f"{sample_id}.npz", **arrays)
        records.append({"sample_id": sample_id, "species": row["species"], "split": row["split"],
                        "predicted_species": CLASS_NAMES[int(logits.argmax(1).item())]})
    result = {"examples": records, "audio_config": metadata["data_config"]["audio"],
              "class_names": list(CLASS_NAMES), "dataset_fingerprint": metadata["dataset_fingerprint"],
              "attention_semantics": "CBAM sigmoid channel/spatial gates; not class-specific; not a causal explanation"}
    atomic_json(output / "metadata.json", result)
    return result
