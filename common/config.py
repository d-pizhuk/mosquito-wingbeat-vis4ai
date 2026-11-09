"""Validated, JSON-serializable configuration; safe to import without PyTorch."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any

SCHEMA_VERSION = 1


def safe_name(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}", value):
        raise ValueError("Use a 1-80 character identifier containing letters, digits, _, - or .")
    return value


def fingerprint(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True)
class AudioConfig:
    sample_rate: int = 8000
    duration_seconds: float = 0.65
    n_fft: int = 512
    hop_length: int = 32
    n_mels: int = 128
    f_min: float = 0.0
    f_max: float = 4000.0
    top_db: float = 80.0

    def __post_init__(self) -> None:
        if self.sample_rate < 1000 or not 0 < self.duration_seconds <= 10:
            raise ValueError("Invalid audio sample rate or duration")
        if not (16 <= self.n_fft <= self.num_samples and 1 <= self.hop_length <= self.n_fft):
            raise ValueError("Require 16 <= n_fft <= waveform length and hop_length <= n_fft")
        if not (8 <= self.n_mels <= self.n_fft // 2 + 1):
            raise ValueError("Invalid number of Mel bands")
        if not 0 <= self.f_min < self.f_max <= self.sample_rate / 2:
            raise ValueError("Frequency bounds must lie inside the Nyquist range")
        if not math.isfinite(self.top_db) or self.top_db <= 0:
            raise ValueError("top_db must be positive and finite")

    @property
    def num_samples(self) -> int:
        return round(self.sample_rate * self.duration_seconds)

    @property
    def num_frames(self) -> int:
        return self.num_samples // self.hop_length + 1


@dataclass(frozen=True)
class DataConfig:
    audio: AudioConfig = field(default_factory=AudioConfig)
    seed: int = 42
    validation_fraction: float = 0.10
    test_fraction: float = 0.20
    split_mode: str = "recording"  # or "group": supplied recording/session metadata
    decode_workers: int = 4

    def __post_init__(self) -> None:
        if not 0 < self.validation_fraction < 1 or not 0 < self.test_fraction < 1:
            raise ValueError("Validation and test fractions must be between zero and one")
        if self.validation_fraction + self.test_fraction >= 0.8:
            raise ValueError("At least 20% of the data must remain for training")
        if self.split_mode not in {"recording", "group"}:
            raise ValueError("split_mode must be recording or group")
        if self.split_mode == "group" and (self.validation_fraction, self.test_fraction) != (0.1, 0.2):
            raise ValueError("Group mode uses 10 stratified group folds: 7 train / 1 val / 2 test")
        if not 0 <= self.decode_workers <= 16:
            raise ValueError("decode_workers must be between 0 and 16")
        if not 0 <= self.seed < 2**32:
            raise ValueError("seed must fit a nonnegative uint32")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DataConfig":
        data = dict(data)
        data["audio"] = AudioConfig(**data.get("audio", {}))
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ModelConfig:
    pretrained: bool = True
    attention_stages: tuple[int, ...] = (2, 3)
    reduction: int = 16
    spatial_kernel: int = 7
    dropout: float = 0.20

    def __post_init__(self) -> None:
        object.__setattr__(self, "attention_stages", tuple(self.attention_stages))
        if not self.attention_stages or len(set(self.attention_stages)) != len(self.attention_stages):
            raise ValueError("Specify at least one unique attention stage")
        if not set(self.attention_stages).issubset({1, 2, 3, 4}):
            raise ValueError("ResNet attention stages must be between 1 and 4")
        if self.reduction < 1 or self.spatial_kernel not in {3, 7}:
            raise ValueError("Invalid CBAM configuration")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")


@dataclass(frozen=True)
class TrainConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    seed: int = 42
    epochs: int = 10
    warmup_epochs: int = 1
    batch_size: int = 128
    num_workers: int = 4
    max_train_samples: int | None = None
    max_validation_samples: int | None = None
    head_lr: float = 0.001
    attention_lr: float = 0.0002
    backbone_lr: float = 0.00003
    weight_decay: float = 0.0001
    label_smoothing: float = 0.05
    class_weighting: bool = True
    unfreeze_from_stage: int = 2
    freeze_bn_stats: bool = True
    patience: int = 3
    min_delta: float = 0.001
    precision: str = "bf16"
    gradient_clip: float = 1.0
    augment: bool = True
    frequency_mask_bins: int = 8
    time_mask_frames: int = 8
    log_every_steps: int = 25
    checkpoint_every_seconds: float = 180.0
    max_hours: float = 4.0
    compute_budget_usd: float = 18.0
    estimated_usd_per_second: float = 0.00078192
    budget_safety_factor: float = 1.15
    deterministic: bool = False

    def __post_init__(self) -> None:
        if not 1 <= self.epochs <= 1000 or not 0 <= self.warmup_epochs < self.epochs:
            raise ValueError("Require 0 <= warmup_epochs < epochs")
        if self.batch_size < 2 or not 0 <= self.num_workers <= 16:
            raise ValueError("batch_size must be >= 2; num_workers must be in [0, 16]")
        for name in ("max_train_samples", "max_validation_samples"):
            value = getattr(self, name)
            if value is not None and value < 6:
                raise ValueError(f"{name} must be None or >= 6")
        for name in ("head_lr", "attention_lr", "backbone_lr", "gradient_clip", "max_hours",
                     "compute_budget_usd", "estimated_usd_per_second", "checkpoint_every_seconds"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.max_hours > 7:
            raise ValueError("max_hours is limited to 7; Modal function timeout is 8 hours")
        if self.precision not in {"bf16", "fp16", "fp32"}:
            raise ValueError("precision must be bf16, fp16 or fp32")
        if self.unfreeze_from_stage not in {1, 2, 3, 4}:
            raise ValueError("Invalid unfreeze stage")
        if self.patience < 1 or self.log_every_steps < 1 or self.min_delta < 0:
            raise ValueError("Invalid stopping or logging configuration")
        if self.budget_safety_factor < 1 or not 0 <= self.label_smoothing < 1:
            raise ValueError("Invalid budget safety factor or label smoothing")
        if min(self.weight_decay, self.frequency_mask_bins, self.time_mask_frames) < 0:
            raise ValueError("Weight decay and masking widths cannot be negative")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TrainConfig":
        data = dict(data)
        data["model"] = ModelConfig(**data.get("model", {}))
        return cls(**data)

    @classmethod
    def load(cls, path: str | Path) -> "TrainConfig":
        return cls.from_dict(json.loads(Path(path).read_text()))

    def to_dict(self) -> dict[str, Any]:
        # Round-trip normalizes tuples to JSON lists for checkpoint comparisons.
        return json.loads(json.dumps(asdict(self)))

    @property
    def allowed_seconds(self) -> float:
        estimated_budget_seconds = self.compute_budget_usd / (
            self.estimated_usd_per_second * self.budget_safety_factor)
        return min(self.max_hours * 3600, estimated_budget_seconds)

    def training_identity(self) -> dict[str, Any]:
        """These settings cannot change when resuming an existing optimizer state."""
        data = self.to_dict()
        for key in ("max_hours", "compute_budget_usd", "estimated_usd_per_second",
                    "budget_safety_factor", "log_every_steps", "checkpoint_every_seconds", "num_workers"):
            data.pop(key)
        return data
