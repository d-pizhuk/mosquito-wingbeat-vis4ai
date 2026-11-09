"""Deterministic CPU audio decoding, shared by preparation and WAV inference."""
from __future__ import annotations
import hashlib
import math
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly
import soundfile as sf

from common.config import AudioConfig


def decode_audio(path: str | Path, config: AudioConfig) -> tuple[np.ndarray, dict]:
    audio, original_rate = sf.read(str(path), dtype="float32", always_2d=True)
    if len(audio) == 0 or not np.isfinite(audio).all():
        raise ValueError(f"Empty or non-finite waveform: {path}")
    original_duration = len(audio) / original_rate
    mono = audio.mean(axis=1, dtype=np.float32)
    if original_rate != config.sample_rate:
        divisor = math.gcd(int(original_rate), config.sample_rate)
        mono = resample_poly(mono, config.sample_rate // divisor, original_rate // divisor).astype(np.float32)
    original_rms = float(np.sqrt(np.mean(mono.astype(np.float64) ** 2)))
    n = config.num_samples
    was_cropped = len(mono) > n
    was_padded = len(mono) < n
    if was_cropped:
        start = (len(mono) - n) // 2
        mono = mono[start:start + n]
    elif was_padded:
        left = (n - len(mono)) // 2
        mono = np.pad(mono, (left, n - len(mono) - left))
    # No peak normalization and no reflection/repetition of the short recording.
    mono = np.asarray(mono, dtype=np.float32)
    mono = np.ascontiguousarray(mono - mono.mean(), dtype=np.float32)
    canonical_rms = float(np.sqrt(np.mean(mono.astype(np.float64) ** 2)))
    if canonical_rms < 1e-10:
        raise ValueError(f"Silent/constant waveform after DC removal: {path}")
    spectrum = np.abs(np.fft.rfft(mono * np.hanning(n)))
    spectrum[0] = 0
    peak_hz = float(np.argmax(spectrum) * config.sample_rate / n)
    return mono, {
        "original_sample_rate": int(original_rate),
        "original_duration_seconds": original_duration,
        "original_channels": int(audio.shape[1]),
        "cropped": was_cropped,
        "padded": was_padded,
        "raw_rms": original_rms,
        "canonical_rms": canonical_rms,
        "spectral_peak_hz": peak_hz,  # strongest FFT bin, NOT guaranteed fundamental wingbeat
        "waveform_sha256": hashlib.sha256(mono.astype("<f4", copy=False).tobytes()).hexdigest(),
    }
