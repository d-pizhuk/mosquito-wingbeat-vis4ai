"""Recompute wingbeat-band spectral peaks for all test predictions.

The CSV column `spectral_peak_hz` is the global FFT argmax and is often ~6 Hz
junk for Anopheles. This script recomputes the strongest bin restricted to
[F_LO, F_HI] Hz after the same decode / centre-crop pipeline as training.
"""
from __future__ import annotations

import io
import time
import zipfile
from math import gcd
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import resample_poly

ROOT = Path(__file__).resolve().parents[2]
PREDS_CSV = ROOT / "results" / "predictions.csv"
OUT_CSV = ROOT / "results" / "f0_band_peak.csv"
ZIP_PATH = Path(r"D:\AI and Vis\archive.zip")

SR = 8000
N = round(SR * 0.65)  # 5200
F_LO, F_HI = 200.0, 1500.0
FREQS = np.fft.rfftfreq(N, d=1.0 / SR)
LO = int(np.searchsorted(FREQS, F_LO, side="left"))
HI = int(np.searchsorted(FREQS, F_HI, side="right"))
WINDOW = np.hanning(N).astype(np.float32)


def load_mono(zf: zipfile.ZipFile, relative_path: str) -> np.ndarray:
    data = zf.read(relative_path)
    audio, orig_sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
    mono = audio.mean(axis=1)
    if orig_sr != SR:
        d = gcd(int(orig_sr), SR)
        mono = resample_poly(mono, SR // d, int(orig_sr) // d).astype(np.float32)
    if len(mono) > N:
        start = (len(mono) - N) // 2
        mono = mono[start : start + N]
    elif len(mono) < N:
        left = (N - len(mono)) // 2
        mono = np.pad(mono, (left, N - len(mono) - left))
    mono = np.asarray(mono, dtype=np.float32)
    return mono - mono.mean()


def band_peak_hz(mono: np.ndarray) -> float:
    spec = np.abs(np.fft.rfft(mono * WINDOW))
    peak_bin = LO + int(np.argmax(spec[LO:HI]))
    return float(FREQS[peak_bin])


def main() -> None:
    if not ZIP_PATH.exists():
        raise FileNotFoundError(ZIP_PATH)
    df = pd.read_csv(PREDS_CSV)
    print(f"Predictions: {len(df):,}  zip: {ZIP_PATH}")
    print(f"Band peak search: {F_LO:.0f}-{F_HI:.0f} Hz  (bins {LO}:{HI})")

    f0 = np.empty(len(df), dtype=np.float64)
    ok = np.zeros(len(df), dtype=bool)
    t0 = time.perf_counter()
    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        for i, path in enumerate(df["relative_path"].tolist()):
            try:
                f0[i] = band_peak_hz(load_mono(zf, path))
                ok[i] = True
            except Exception as exc:  # noqa: BLE001 — log and continue
                f0[i] = np.nan
                ok[i] = False
                print(f"FAIL {path}: {exc}")
            if (i + 1) % 2000 == 0 or i + 1 == len(df):
                elapsed = time.perf_counter() - t0
                rate = (i + 1) / elapsed
                eta = (len(df) - i - 1) / rate
                print(f"  {i+1:,}/{len(df):,}  {rate:.0f}/s  ETA {eta/60:.1f} min")

    out = df[["sample_id", "relative_path", "true_species", "spectral_peak_hz"]].copy()
    out["f0_hz"] = f0
    out["f0_ok"] = ok
    out["f0_method"] = f"rfft_argmax_{int(F_LO)}_{int(F_HI)}hz"
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)
    print(f"Wrote {OUT_CSV}  ok={ok.mean():.1%}")

    print("\nPer-species f0_hz summary (valid only):")
    valid = out[out["f0_ok"]]
    print(
        valid.groupby("true_species")["f0_hz"]
        .agg(["count", "median", "mean", "std"])
        .round(1)
        .to_string()
    )


if __name__ == "__main__":
    main()
