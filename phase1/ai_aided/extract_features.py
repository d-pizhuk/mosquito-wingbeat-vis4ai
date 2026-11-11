"""Phase I (AI-aided): decode the test recordings and measure their acoustics.

Run from the repository root:

    python -m phase1.ai_aided.extract_features --audio-root ../archive

Outputs
    data/prepared/phase1_test_waveforms.npy   model-ready waveforms (git-ignored, ~1.2 GB)
    phase1/ai_aided/cache/features.csv        one row of acoustic measurements per test clip
    phase1/ai_aided/cache/spectra.npz         long-term spectrum (0-4 kHz) per test clip

Every measurement is taken from the exact waveform the model receives
(`common.audio.decode_audio`), so a feature always describes the model input.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt, hilbert

from common.audio import decode_audio
from common.config import AudioConfig

AUDIO = AudioConfig()
SR = AUDIO.sample_rate
N = AUDIO.num_samples
NFFT = 16384                                   # zero-padded: 0.49 Hz per bin
FREQS = np.fft.rfftfreq(NFFT, 1 / SR)
F0_GRID = np.arange(200.0, 1000.0, 1.0)        # candidate wingbeat fundamentals (Hz)
N_HARMONICS = 6
HIGHPASS = butter(4, 120, "highpass", fs=SR, output="sos")
SPECTRUM_BINS = 400                            # 10 Hz bins for the stored long-term spectrum


def _band_max(magnitude: np.ndarray, centre: float, half_width: float) -> float:
    lo, hi = np.searchsorted(FREQS, [centre - half_width, centre + half_width])
    return float(magnitude[lo:max(hi, lo + 1)].max())


def estimate_f0(magnitude: np.ndarray) -> tuple[float, float]:
    """Harmonic-comb fundamental estimate.

    For each candidate f0 the score adds the spectral magnitude at its first
    four harmonics and subtracts the magnitude half-way between them, which
    penalises the classic "one octave too low" error. Returns (f0, salience),
    where salience in [0, 1] says how much of the comb sits on the harmonics.
    """
    index = lambda f: np.clip(np.rint(f / (SR / NFFT)).astype(int), 0, len(magnitude) - 1)
    on = sum(magnitude[index(F0_GRID * k)] / k for k in range(1, 5))
    off = sum(magnitude[index(F0_GRID * (k - 0.5))] / k for k in range(1, 5))
    score = on - off
    best = int(score.argmax())
    f0 = F0_GRID[best]
    # refine on the true spectral peak near the winning candidate
    lo, hi = np.searchsorted(FREQS, [f0 - 4, f0 + 4])
    f0 = float(FREQS[lo + magnitude[lo:hi].argmax()])
    return f0, float(score[best] / (on[best] + off[best] + 1e-12))


def measure(waveform: np.ndarray) -> tuple[dict, np.ndarray]:
    x = sosfiltfilt(HIGHPASS, waveform.astype(np.float64))      # drop sensor drift / mains hum
    magnitude = np.abs(np.fft.rfft(x * np.hanning(len(x)), NFFT))
    # light smoothing (about 6 Hz) so a harmonic is not missed by one bin
    smooth = np.convolve(magnitude, np.ones(13) / 13, mode="same")
    f0, salience = estimate_f0(smooth)

    power = magnitude ** 2
    band = (FREQS >= 150) & (FREQS <= 3900)
    harmonic_mask = np.zeros_like(band)
    harmonics = []
    for k in range(1, N_HARMONICS + 1):
        centre = k * f0
        if centre > 3900:
            harmonics.append(np.nan)
            continue
        harmonic_mask |= np.abs(FREQS - centre) <= max(12.0, 0.03 * centre)
        harmonics.append(_band_max(smooth, centre, max(8.0, 0.02 * centre)))
    harmonic_power = power[band & harmonic_mask].sum()
    residual_power = power[band & ~harmonic_mask].sum()
    snr_db = 10 * np.log10((harmonic_power + 1e-20) / (residual_power + 1e-20))
    h = np.array(harmonics)
    in_band = power[band]
    centroid = float((FREQS[band] * in_band).sum() / (in_band.sum() + 1e-20))
    flatness = float(np.exp(np.mean(np.log(in_band + 1e-20))) / (in_band.mean() + 1e-20))

    # envelope: where inside the 0.65 s window does the insect actually fly past?
    envelope = np.abs(hilbert(x))
    envelope = np.convolve(envelope, np.ones(160) / 160, mode="same")       # 20 ms
    peak = envelope.max() + 1e-20
    active = envelope > 0.25 * peak
    time = np.arange(len(x)) / SR
    weights = envelope ** 2
    event_centre = float((time * weights).sum() / weights.sum())
    edge = max(envelope[100:260].mean(), envelope[-260:-100].mean()) / peak  # 100-sample pad each side
    features = {
        "f0_hz": f0, "f0_salience": salience, "snr_db": float(snr_db),
        "h2_db": float(20 * np.log10((h[1] + 1e-20) / (h[0] + 1e-20))),
        "h3_db": float(20 * np.log10((h[2] + 1e-20) / (h[0] + 1e-20))),
        "h4_db": float(20 * np.log10((h[3] + 1e-20) / (h[0] + 1e-20))),
        "centroid_hz": centroid, "flatness": flatness,
        "event_seconds": float(active.sum() / SR), "event_centre_s": event_centre,
        "edge_level": float(edge), "rms_highpassed": float(np.sqrt(np.mean(x ** 2))),
    }
    spectrum = 10 * np.log10(power[: SPECTRUM_BINS * 20].reshape(SPECTRUM_BINS, 20).mean(1) + 1e-20)
    return features, (spectrum - spectrum.max()).astype(np.float32)


def _task(path: str):
    waveform, _ = decode_audio(path, AUDIO)
    features, spectrum = measure(waveform)
    return waveform, features, spectrum


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio-root", required=True, help="folder that contains the outer Wingbeats/ directory")
    parser.add_argument("--predictions", default="results/predictions.csv")
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()

    predictions = pd.read_csv(args.predictions)
    paths = [str(Path(args.audio_root) / p) for p in predictions["relative_path"]]
    cache_dir = Path("phase1/ai_aided/cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    Path("data/prepared").mkdir(parents=True, exist_ok=True)
    waveforms = np.lib.format.open_memmap("data/prepared/phase1_test_waveforms.npy", mode="w+",
                                          dtype=np.float32, shape=(len(paths), N))
    spectra = np.empty((len(paths), SPECTRUM_BINS), dtype=np.float32)
    rows = []
    with ProcessPoolExecutor(args.workers) as pool:
        for i, (waveform, features, spectrum) in enumerate(pool.map(_task, paths, chunksize=256)):
            waveforms[i], spectra[i] = waveform, spectrum
            rows.append(features)
            if i % 5000 == 0:
                print(f"{i}/{len(paths)}", flush=True)
    waveforms.flush()
    frame = pd.DataFrame(rows)
    frame.insert(0, "sample_id", predictions["sample_id"].to_numpy())
    frame.to_csv(cache_dir / "features.csv", index=False, float_format="%.5g")
    np.savez_compressed(cache_dir / "spectra.npz", spectra=spectra,
                        frequency_hz=np.arange(SPECTRUM_BINS) * 20 * SR / NFFT + 10 * SR / NFFT)
    print("done", frame.describe().T.to_string())


if __name__ == "__main__":
    main()
