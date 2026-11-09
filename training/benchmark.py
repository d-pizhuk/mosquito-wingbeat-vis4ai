"""Measure the actual pipeline before committing to a long paid training run."""
from __future__ import annotations
import time
from pathlib import Path

import pandas as pd
import torch
from torch.nn import functional as F

from common.config import DataConfig, TrainConfig
from common.io_utils import emit
from common.load_data import make_loader, read_metadata, select_rows
from common.model import build_model
from training.engine import (amp_context, class_weights, configure_epoch, make_optimizer,
                             make_scaler, resolve_device, seed_everything, train_batch)


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _batches(loader):
    while True:
        for batch in loader:
            yield batch


def benchmark_training(dataset_directory: str | Path, config: TrainConfig, *,
                       device: str | None = None, warmup_steps: int = 8, measured_steps: int = 30) -> dict:
    if not 1 <= warmup_steps <= 100 or not 2 <= measured_steps <= 200:
        raise ValueError("Benchmark steps outside the permitted range")
    started = time.monotonic()
    device_obj = resolve_device(device)
    metadata = read_metadata(dataset_directory)
    data = DataConfig.from_dict(metadata["data_config"])
    manifest = pd.read_csv(Path(dataset_directory) / "manifest.csv", keep_default_na=False)
    train_rows = select_rows(manifest, "train", config.max_train_samples, config.seed)
    val_rows = select_rows(manifest, "validation", config.max_validation_samples, config.seed + 1)
    seed_everything(config.seed, config.deterministic)
    model = build_model(data.audio, config.model).to(device_obj)
    if device_obj.type == "cuda":
        model.to(memory_format=torch.channels_last)
        torch.cuda.reset_peak_memory_stats(device_obj)
    optimizer, scaler = make_optimizer(model, config), make_scaler(device_obj, config.precision)
    weights = class_weights(manifest, train_rows, device_obj) if config.class_weighting else None
    loader = make_loader(dataset_directory, manifest, train_rows, batch_size=config.batch_size,
                         num_workers=config.num_workers, seed=config.seed, shuffle=True,
                         cuda=device_obj.type == "cuda")
    iterator = _batches(loader)
    results = {}
    for name, epoch in (("warmup", 0), ("finetune", config.warmup_epochs)):
        configure_epoch(model, optimizer, config, epoch)
        model.train()
        samples = 0
        for step in range(warmup_steps + measured_steps):
            if step == warmup_steps:
                _sync(device_obj)
                clock = time.monotonic()
            waveform, target, _ = next(iterator)
            waveform, target = waveform.to(device_obj, non_blocking=True), target.to(device_obj, non_blocking=True)
            train_batch(model, optimizer, scaler, waveform, target, config, device_obj, weights)
            if step >= warmup_steps:
                samples += len(target)
        _sync(device_obj)
        seconds = time.monotonic() - clock
        results[name + "_samples_per_second"] = samples / seconds
        emit("benchmark_stage", stage=name, samples=samples, seconds=round(seconds, 3),
             samples_per_second=round(samples / seconds, 2))
    model.eval()
    val_loader = make_loader(dataset_directory, manifest, val_rows, batch_size=config.batch_size,
                             num_workers=config.num_workers, seed=config.seed + 1,
                             cuda=device_obj.type == "cuda")
    _sync(device_obj)
    clock, samples = time.monotonic(), 0
    with torch.inference_mode():
        for step, (waveform, target, _) in enumerate(val_loader):
            waveform, target = waveform.to(device_obj, non_blocking=True), target.to(device_obj, non_blocking=True)
            with amp_context(device_obj, config.precision):
                logits = model(waveform)
                F.cross_entropy(logits, target).item()
            samples += len(target)
            if step + 1 >= measured_steps:
                break
    _sync(device_obj)
    results["validation_samples_per_second"] = samples / max(1e-6, time.monotonic() - clock)
    training_seconds = len(train_rows) * (
        config.warmup_epochs / results["warmup_samples_per_second"] +
        (config.epochs - config.warmup_epochs) / results["finetune_samples_per_second"])
    validation_seconds = len(val_rows) * config.epochs / results["validation_samples_per_second"]
    projected_seconds = training_seconds + validation_seconds
    results.update({"training_samples": len(train_rows), "validation_samples": len(val_rows),
                    "epochs": config.epochs, "projected_train_validation_hours": projected_seconds / 3600,
                    "projected_compute_usd": projected_seconds * config.estimated_usd_per_second,
                    "projected_compute_usd_with_20_percent_margin": projected_seconds * config.estimated_usd_per_second * 1.2,
                    "within_configured_time_budget_with_margin": projected_seconds * 1.2 <= config.allowed_seconds,
                    "peak_gpu_memory_gib": torch.cuda.max_memory_allocated(device_obj) / 1024**3 if device_obj.type == "cuda" else None,
                    "benchmark_seconds_excluding_staging": time.monotonic() - started,
                    "dataset_fingerprint": metadata["dataset_fingerprint"],
                    "note": "Actual measured throughput, but the extrapolation excludes initial staging, checkpoint writes, cold starts and other jobs. Benchmark weights are discarded."})
    emit("benchmark_complete", **results)
    return results
