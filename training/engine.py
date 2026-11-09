"""Budget-aware single-GPU fine-tuning with resumable mid-epoch checkpoints."""
from __future__ import annotations
from contextlib import nullcontext
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import random
import signal
import time
from typing import Callable

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

from common.class_names import CLASS_NAMES, NUM_CLASSES
from common.config import DataConfig, SCHEMA_VERSION, TrainConfig, fingerprint
from common.io_utils import atomic_json, emit
from common.load_data import make_loader, read_metadata, select_rows
from common.model import build_model
from training.checkpoints import atomic_torch_save, load_checkpoint, restore_rng, rng_state
from training.metrics import Meter


class TrainingStopped(Exception):
    """A safe stop requested by the local budget or an external signal."""


def seed_everything(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = not deterministic
        torch.backends.cuda.matmul.allow_tf32 = not deterministic
        torch.backends.cudnn.allow_tf32 = not deterministic
    torch.use_deterministic_algorithms(deterministic, warn_only=False)


def resolve_device(value: str | None = None) -> torch.device:
    if value:
        device = torch.device(value)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but no working CUDA device is available")
    return device


def amp_context(device: torch.device, precision: str):
    if device.type != "cuda" or precision == "fp32":
        return nullcontext()
    dtype = torch.bfloat16 if precision == "bf16" and torch.cuda.is_bf16_supported() else torch.float16
    return torch.autocast("cuda", dtype=dtype)


def make_scaler(device: torch.device, precision: str):
    needs_scaling = device.type == "cuda" and (
        precision == "fp16" or (precision == "bf16" and not torch.cuda.is_bf16_supported()))
    return torch.amp.GradScaler("cuda", enabled=needs_scaling)


def configure_epoch(model, optimizer, config: TrainConfig, epoch: int) -> None:
    warmup = epoch < config.warmup_epochs
    model.set_training_stage(warmup=warmup, unfreeze_from_stage=config.unfreeze_from_stage,
                             freeze_bn_stats=config.freeze_bn_stats)
    progress = max(0, epoch - config.warmup_epochs) / max(1, config.epochs - config.warmup_epochs - 1)
    factor = 0.1 + 0.9 * (1 + math.cos(math.pi * progress)) / 2
    for group in optimizer.param_groups:
        if group["name"] == "backbone":
            group["lr"] = 0 if warmup else config.backbone_lr * factor
        elif group["name"] == "attention":
            group["lr"] = config.head_lr if warmup else config.attention_lr * factor
        else:
            group["lr"] = config.head_lr if warmup else config.attention_lr * factor


def make_optimizer(model, config: TrainConfig):
    # Keep all groups present from the start, so checkpoint state remains stable
    # when previously frozen layers become trainable.
    return torch.optim.AdamW([
        {"params": list(model.backbone.parameters()), "name": "backbone", "lr": config.backbone_lr},
        {"params": list(model.attention.parameters()), "name": "attention", "lr": config.head_lr},
        {"params": list(model.classifier.parameters()), "name": "classifier", "lr": config.head_lr},
    ], weight_decay=config.weight_decay)


def class_weights(manifest: pd.DataFrame, rows: np.ndarray, device: torch.device) -> torch.Tensor:
    labels = manifest.set_index("row_idx").loc[rows, "label"].to_numpy()
    counts = np.bincount(labels, minlength=NUM_CLASSES).astype(np.float64)
    if (counts == 0).any():
        raise ValueError("Every class must have training examples")
    weights = np.sqrt(counts.mean() / counts)
    weights = np.clip(weights, 0.25, 3.0)
    return torch.tensor(weights / weights.mean(), dtype=torch.float32, device=device)


def train_batch(model, optimizer, scaler, waveforms, target, config, device, weights=None):
    optimizer.zero_grad(set_to_none=True)
    with amp_context(device, config.precision):
        logits = model(waveforms, augment=config.augment,
                       frequency_mask_bins=config.frequency_mask_bins, time_mask_frames=config.time_mask_frames)
        losses = F.cross_entropy(logits, target, weight=weights,
                                 label_smoothing=config.label_smoothing, reduction="none")
        loss = losses.mean()
    if not torch.isfinite(loss):
        raise FloatingPointError("Training loss is not finite; refusing to overwrite a valid checkpoint")
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad),
                            config.gradient_clip, error_if_nonfinite=not scaler.is_enabled())
    scaler.step(optimizer)
    scaler.update()
    return logits.detach(), losses.detach()


@torch.inference_mode()
def validate(model, loader, device, config, should_stop: Callable[[], bool] | None = None) -> dict:
    model.eval()
    meter = Meter()
    for waveforms, target, _ in loader:
        if should_stop and should_stop():
            raise TrainingStopped("Stopped during validation; it will be rerun on resume")
        waveforms = waveforms.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        with amp_context(device, config.precision):
            logits = model(waveforms)
            # Validation is unsmoothed, unweighted cross-entropy.
            losses = F.cross_entropy(logits, target, reduction="none")
        meter.update(logits, target, losses)
    if meter.count == 0:
        raise ValueError("Validation loader is empty")
    return meter.result()


def environment_info() -> dict:
    packages = {}
    for package in ("torch", "torchvision", "numpy", "scipy", "pandas", "librosa", "soundfile", "modal"):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    source_root = Path(__file__).resolve().parents[1]
    sources = {str(p.relative_to(source_root)): p.read_text()
               for package in ("common", "training") for p in sorted((source_root / package).glob("*.py"))}
    return {"python": platform.python_version(), "platform": platform.platform(), "packages": packages,
            "source_fingerprint": fingerprint(sources), "cuda_version": torch.version.cuda,
            "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None}


def run_training(dataset_directory: str | Path, run_directory: str | Path, config: TrainConfig, *,
                 resume: bool = False, device: str | None = None,
                 commit: Callable[[], None] | None = None, entry_started: float | None = None,
                 stop_after_steps: int | None = None) -> dict:
    """Train one run. stop_after_steps is a local testing/debugging facility.

    Resuming restores optimizer, scaler, RNG, epoch permutation and batch cursor.
    Dataset/model/training settings cannot silently change during resume. Budgets
    may be increased explicitly. This timer is NOT a Modal account spending cap.
    """
    started = time.monotonic() if entry_started is None else entry_started
    device_obj = resolve_device(device)
    directory, output = Path(dataset_directory), Path(run_directory)
    metadata = read_metadata(directory)
    data_config = DataConfig.from_dict(metadata["data_config"])
    output.mkdir(parents=True, exist_ok=True)
    last_path = output / "last.pt"
    checkpoint = None
    if last_path.exists():
        if not resume:
            raise FileExistsError("Run already exists. Use resume=True, or choose a new run_id")
        checkpoint = load_checkpoint(last_path)
        if checkpoint["dataset_fingerprint"] != metadata["dataset_fingerprint"]:
            raise ValueError("Cannot resume against a different dataset or split")
        old_config = TrainConfig.from_dict(checkpoint["train_config"])
        if old_config.training_identity() != config.training_identity():
            raise ValueError("Training/model settings changed. Only budget, logging and worker settings may change on resume")
        if checkpoint["state"]["status"] in {"completed", "early_stopped"}:
            emit("already_finished", run=str(output), status=checkpoint["state"]["status"])
            return json.loads((output / "status.json").read_text())
    elif resume:
        raise FileNotFoundError("No last.pt exists for this run; start with resume=False")
    elif (output / "config.json").exists():
        raise FileExistsError("An incomplete run without a checkpoint exists; use a new run_id")

    seed_everything(config.seed, config.deterministic)
    manifest = pd.read_csv(directory / "manifest.csv", keep_default_na=False)
    train_rows = select_rows(manifest, "train", config.max_train_samples, config.seed)
    validation_rows = select_rows(manifest, "validation", config.max_validation_samples, config.seed + 1)
    if len(train_rows) < 2:
        raise ValueError("Training requires at least two examples")
    row_fingerprint = fingerprint({"train": train_rows.tolist(), "validation": validation_rows.tolist()})
    if checkpoint and checkpoint["row_fingerprint"] != row_fingerprint:
        raise ValueError("Selected training/validation rows changed")
    if not checkpoint:
        np.save(output / "train_rows.npy", train_rows, allow_pickle=False)
        np.save(output / "validation_rows.npy", validation_rows, allow_pickle=False)
    atomic_json(output / "config.json", config.to_dict())
    atomic_json(output / "dataset_metadata.json", metadata)
    atomic_json(output / "environment.json", environment_info())
    model = build_model(data_config.audio, config.model, load_imagenet=False if checkpoint else None).to(device_obj)
    if device_obj.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
    optimizer = make_optimizer(model, config)
    scaler = make_scaler(device_obj, config.precision)
    state = {"epoch": 0, "next_batch": 0, "global_step": 0, "best_macro_f1": -1.0,
             "best_epoch": None, "bad_epochs": 0, "history": [], "train_meter": None,
             "elapsed_seconds": 0.0, "status": "running"}
    if checkpoint:
        model.load_state_dict(checkpoint["model_state"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scaler.load_state_dict(checkpoint["scaler_state"])
        state.update(checkpoint["state"])
        restore_rng(checkpoint["rng_state"])
    base_elapsed = float(state["elapsed_seconds"])
    # A crash can lose time since the last save. Conservatively reserve one save
    # interval when the previous checkpoint still marked the call as running.
    if checkpoint and state["status"] == "running":
        base_elapsed += config.checkpoint_every_seconds
    weights = class_weights(manifest, train_rows, device_obj) if config.class_weighting else None
    total_batches = math.ceil(len(train_rows) / config.batch_size)
    step_count_this_call = 0
    stop_flag = {"requested": False}
    original_handler = None
    try:
        original_handler = signal.signal(signal.SIGTERM, lambda *_: stop_flag.update(requested=True))
    except ValueError:
        pass  # Local notebooks can invoke this from a non-main thread.

    def elapsed() -> float:
        return base_elapsed + time.monotonic() - started

    def should_stop() -> bool:
        return (stop_flag["requested"] or elapsed() >= max(0, config.allowed_seconds - 30)
                or (stop_after_steps is not None and step_count_this_call >= stop_after_steps))

    def log(event: str, **values) -> None:
        payload = {"event": event, "elapsed_seconds": round(elapsed(), 3), **values}
        with (output / "events.jsonl").open("a") as stream:
            stream.write(json.dumps(payload, allow_nan=False) + "\n")
        emit(event, **{k: v for k, v in payload.items() if k != "event"})

    def common_payload() -> dict:
        return {"schema_version": SCHEMA_VERSION, "architecture": "resnet50_cbam",
                "class_names": list(CLASS_NAMES), "train_config": config.to_dict(),
                "data_config": data_config.to_dict(), "dataset_fingerprint": metadata["dataset_fingerprint"],
                "row_fingerprint": row_fingerprint,
                "model_state": model.state_dict()}

    def save(status: str = "running") -> dict:
        state["status"] = status
        state["elapsed_seconds"] = elapsed()
        payload = {**common_payload(), "state": state, "optimizer_state": optimizer.state_dict(),
                   "scaler_state": scaler.state_dict(), "rng_state": rng_state()}
        atomic_torch_save(last_path, payload)
        summary = {k: v for k, v in state.items() if k not in {"history", "train_meter"}}
        summary.update({"training_samples": len(train_rows), "validation_samples": len(validation_rows),
                        "dataset_fingerprint": metadata["dataset_fingerprint"],
                        "estimated_compute_usd": state["elapsed_seconds"] * config.estimated_usd_per_second,
                        "allowed_seconds": config.allowed_seconds,
                        "budget_note": "estimate only; excludes other jobs, image builds, cold starts, storage and pricing changes"})
        atomic_json(output / "status.json", summary)
        if state["history"]:
            temporary = output / "history.csv.tmp"
            pd.DataFrame(state["history"]).to_csv(temporary, index=False)
            temporary.replace(output / "history.csv")
        if commit:
            commit()
        return summary

    last_saved = time.monotonic()
    log("train_start", train_samples=len(train_rows), validation_samples=len(validation_rows),
        epochs=config.epochs, resume=resume, device=str(device_obj),
        allowed_hours=round(config.allowed_seconds / 3600, 3),
        precision=config.precision if device_obj.type == "cuda" else "fp32")
    if not checkpoint:
        save()  # Persist a known-good initial state before the first update.
    try:
        while state["epoch"] < config.epochs:
            epoch = state["epoch"]
            if should_stop():
                raise TrainingStopped("Wall-time/budget or explicit stop requested")
            configure_epoch(model, optimizer, config, epoch)
            model.train()
            train_meter = Meter(state["train_meter"])
            epoch_started = time.monotonic()
            seen_this_epoch_call = 0
            if state["next_batch"] < total_batches:
                loader = make_loader(directory, manifest, train_rows, batch_size=config.batch_size,
                                     num_workers=config.num_workers, seed=config.seed, epoch=epoch,
                                     shuffle=True, start_batch=state["next_batch"], cuda=device_obj.type == "cuda")
                offset = state["next_batch"]
                for local_batch, (waveforms, target, _) in enumerate(loader):
                    if should_stop():
                        raise TrainingStopped("Wall-time/budget or explicit stop requested")
                    waveforms = waveforms.to(device_obj, non_blocking=True)
                    target = target.to(device_obj, non_blocking=True)
                    logits, losses = train_batch(model, optimizer, scaler, waveforms, target,
                                                 config, device_obj, weights)
                    train_meter.update(logits, target, losses)
                    state["next_batch"] = offset + local_batch + 1
                    state["global_step"] += 1
                    step_count_this_call += 1
                    seen_this_epoch_call += len(target)
                    state["train_meter"] = train_meter.state()
                    if state["global_step"] % config.log_every_steps == 0:
                        result = train_meter.result()
                        log("train_progress", epoch=epoch + 1, epochs=config.epochs,
                            batch=state["next_batch"], batches=total_batches,
                            loss=round(result["loss"], 5), accuracy=round(result["accuracy"], 5),
                            samples_per_second=round(seen_this_epoch_call / max(.001, time.monotonic() - epoch_started), 2),
                            estimated_compute_usd=round(elapsed() * config.estimated_usd_per_second, 3))
                    if time.monotonic() - last_saved >= config.checkpoint_every_seconds:
                        save()
                        last_saved = time.monotonic()
                del loader
            if should_stop():
                raise TrainingStopped("Stopping before validation; the completed training batches are saved")
            validation_loader = make_loader(directory, manifest, validation_rows, batch_size=config.batch_size,
                                            num_workers=config.num_workers, seed=config.seed + 1,
                                            cuda=device_obj.type == "cuda")
            validation = validate(model, validation_loader, device_obj, config, should_stop)
            del validation_loader
            train_result = train_meter.result()
            score = validation["macro_f1"]
            old_best = state["best_macro_f1"]
            state["bad_epochs"] = 0 if score > old_best + config.min_delta else state["bad_epochs"] + 1
            if score > old_best:
                state["best_macro_f1"] = score
                state["best_epoch"] = epoch + 1
                atomic_torch_save(output / "best.pt", {**common_payload(), "epoch": epoch + 1,
                                                       "validation_metrics": validation})
            record = {"epoch": epoch + 1, "train_loss": train_result["loss"],
                      "train_accuracy": train_result["accuracy"], "validation_loss": validation["loss"],
                      "validation_accuracy": validation["accuracy"], "validation_macro_f1": score,
                      "elapsed_seconds": elapsed(), "epoch_seconds_this_call": time.monotonic() - epoch_started,
                      "best_epoch": state["best_epoch"]}
            state["history"].append(record)
            state["epoch"], state["next_batch"], state["train_meter"] = epoch + 1, 0, None
            log("epoch_complete", **{k: round(v, 5) if isinstance(v, float) else v
                                     for k, v in record.items() if k != "elapsed_seconds"})
            status = "completed" if state["epoch"] == config.epochs else (
                "early_stopped" if state["bad_epochs"] >= config.patience else "running")
            summary = save(status)
            last_saved = time.monotonic()
            if status != "running":
                log("train_finished", status=status, best_epoch=state["best_epoch"],
                    best_validation_macro_f1=state["best_macro_f1"])
                if commit:
                    commit()
                return summary
        return save("completed")
    except TrainingStopped as error:
        log("train_paused", reason=str(error), best_checkpoint_exists=(output / "best.pt").exists())
        return save("paused")
    except BaseException as error:
        # Do not replace a valid checkpoint with a possibly half-finished update.
        atomic_json(output / "failure.json", {"type": type(error).__name__, "message": str(error),
                                              "last_completed_step_in_memory": state["global_step"],
                                              "elapsed_seconds_this_call": time.monotonic() - started})
        if commit:
            commit()
        raise
    finally:
        if original_handler is not None:
            signal.signal(signal.SIGTERM, original_handler)
