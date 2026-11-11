"""Phase I (AI-aided): black-box probes. Only inputs are changed, only outputs are read.

Run from the repository root after `extract_features`:

    python -m phase1.ai_aided.run_probes

Probes (all written to phase1/ai_aided/cache/)
    sanity.json          local model reproduces results/predictions.csv
    projection.csv       UMAP of the *inputs* (time-averaged log-Mel spectrum), all test clips
    pitch_shift.csv      species probabilities when a clip is resampled to move its pitch
    ablation.csv         predictions when parts of the signal are removed (harmonics, background, ...)
    band_occlusion.csv   accuracy when one 250 Hz band is silenced
    noise.csv            predictions when white noise is mixed in at a fixed signal-to-noise ratio
    cue_conflict.csv     harmonics of one clip + background of a clip from another species

Run a subset with e.g. `python -m phase1.ai_aided.run_probes cue_conflict noise`.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import hilbert, resample
import torch

from common.class_names import CLASS_NAMES, CLASS_SLUGS
from common.inference import load_model
from training.engine import amp_context

CACHE = Path("phase1/ai_aided/cache")
CHECKPOINT = "results/runs/r50-cbam-full-001/best.pt"
SR, N, PAD = 8000, 5200, 100          # 5000 recorded samples, zero-padded by 100 on each side
RNG = np.random.default_rng(42)


class BlackBox:
    def __init__(self):
        self.model, checkpoint, self.device = load_model(CHECKPOINT)
        self.precision = checkpoint["train_config"]["precision"]

    @torch.inference_mode()
    def __call__(self, waveforms: np.ndarray, batch: int = 512) -> np.ndarray:
        out = []
        for start in range(0, len(waveforms), batch):
            x = torch.from_numpy(np.ascontiguousarray(waveforms[start:start + batch], dtype=np.float32)).to(self.device)
            with amp_context(self.device, self.precision):
                logits = self.model(x)
            out.append(logits.float().softmax(1).cpu().numpy())
        return np.concatenate(out)

    @torch.inference_mode()
    def mean_log_mel(self, waveforms: np.ndarray, batch: int = 1024) -> np.ndarray:
        out = []
        for start in range(0, len(waveforms), batch):
            x = torch.from_numpy(np.ascontiguousarray(waveforms[start:start + batch], dtype=np.float32)).to(self.device)
            out.append(self.model.frontend.spectrogram(x).mean(dim=2).cpu().numpy())
        return np.concatenate(out)


def fit_length(x: np.ndarray) -> np.ndarray:
    """Centre crop / zero pad to the model input length (same rule as decode_audio)."""
    length = x.shape[1]
    if length > N:
        start = (length - N) // 2
        return x[:, start:start + N]
    left = (N - length) // 2
    return np.pad(x, ((0, 0), (left, N - length - left)))


def spectral_edit(x: np.ndarray, keep: np.ndarray) -> np.ndarray:
    """Multiply the spectrum of every clip by a per-clip 0/1 mask (zero-phase, no ringing tricks)."""
    spectrum = np.fft.rfft(x, axis=1)
    return np.fft.irfft(spectrum * keep, n=x.shape[1], axis=1).astype(np.float32)


def stratified(frame: pd.DataFrame, per_class: int) -> np.ndarray:
    picks = [RNG.choice(group.index.to_numpy(), min(per_class, len(group)), replace=False)
             for _, group in frame.groupby("true_class")]
    return np.sort(np.concatenate(picks))


def summarise(frame: pd.DataFrame, rows: np.ndarray, probabilities: np.ndarray, **extra) -> pd.DataFrame:
    out = pd.DataFrame({"sample_id": frame.loc[rows, "sample_id"].to_numpy(),
                        "true_class": frame.loc[rows, "true_class"].to_numpy(),
                        "f0_hz": frame.loc[rows, "f0_hz"].to_numpy(),
                        "predicted_class": probabilities.argmax(1)})
    for slug, column in zip(CLASS_SLUGS, probabilities.T):
        out[f"prob_{slug}"] = column.astype(np.float32)
    for key, value in extra.items():
        out[key] = value
    return out


def harmonic_mask(f0: np.ndarray, frequencies: np.ndarray, first: int = 1, last: int = 12) -> np.ndarray:
    """True on the bins that belong to harmonics `first..last` of each clip's fundamental.

    The wingbeat pitch drifts during a pass, so each harmonic is a band, not a line:
    +-7% of its frequency (at least 30 Hz), capped at +-0.3 f0 so that the gaps
    between neighbouring harmonics never close.
    """
    mask = np.zeros((len(f0), len(frequencies)), dtype=bool)
    base = f0[:, None]
    for k in range(first, last + 1):
        half_width = np.minimum(np.maximum(30.0, 0.07 * k * base), 0.3 * base)
        mask |= np.abs(frequencies[None, :] - k * base) <= half_width
    return mask


def probe_sanity(frame, waveforms, box, frequencies) -> None:
    probabilities = box(waveforms)
    agreement = float((probabilities.argmax(1) == frame["predicted_class"].to_numpy()).mean())
    stored = frame[[f"prob_{s}" for s in CLASS_SLUGS]].to_numpy()
    sanity = {"clips": len(frame), "prediction_agreement": agreement,
              "max_probability_difference": float(np.abs(probabilities - stored).max()),
              "local_accuracy": float((probabilities.argmax(1) == frame["true_class"].to_numpy()).mean())}
    (CACHE / "sanity.json").write_text(json.dumps(sanity, indent=2))
    print("sanity", sanity, flush=True)


def probe_projection(frame, waveforms, box, frequencies) -> None:
    from sklearn.decomposition import PCA
    import umap
    mel = box.mean_log_mel(waveforms)
    np.save(CACHE / "mean_log_mel.npy", mel.astype(np.float16))
    reduced = PCA(30, random_state=0).fit_transform(mel)
    xy = umap.UMAP(n_neighbors=30, min_dist=0.15, random_state=42).fit_transform(reduced)
    pd.DataFrame({"sample_id": frame["sample_id"], "umap_x": xy[:, 0], "umap_y": xy[:, 1]}).to_csv(
        CACHE / "projection.csv", index=False, float_format="%.4f")
    print("projection done", flush=True)


def probe_pitch_shift(frame, waveforms, box, frequencies) -> None:
    rows = stratified(frame, 1000)
    core = np.asarray(waveforms[rows])[:, PAD:N - PAD]
    parts = []
    for ratio in np.round(np.arange(0.60, 1.6001, 0.05), 2):
        shifted = fit_length(resample(core, int(round(core.shape[1] / ratio)), axis=1).astype(np.float32))
        parts.append(summarise(frame, rows, box(shifted), ratio=ratio))
    pd.concat(parts).to_csv(CACHE / "pitch_shift.csv", index=False, float_format="%.5g")
    print("pitch shift done", flush=True)


def probe_ablation(frame, waveforms, box, frequencies) -> None:
    rows = stratified(frame, 2500)
    clips = np.asarray(waveforms[rows])
    variants = ablation_variants(clips, frame.loc[rows, "f0_hz"].to_numpy(), frequencies)
    parts = [summarise(frame, rows, box(x), variant=name) for name, x in variants.items()]
    pd.concat(parts).to_csv(CACHE / "ablation.csv", index=False, float_format="%.5g")
    print("ablation done", flush=True)


def ablation_variants(clips: np.ndarray, f0: np.ndarray, frequencies: np.ndarray) -> dict[str, np.ndarray]:
    """Edited copies of the same clips; each removes or isolates one kind of evidence."""
    harmonic = harmonic_mask(f0, frequencies)
    fundamental = harmonic_mask(f0, frequencies, last=1)
    envelope = np.abs(hilbert(clips, axis=1))
    kernel = np.ones(160) / 160
    envelope = np.apply_along_axis(lambda v: np.convolve(v, kernel, mode="same"), 1, envelope)
    envelope_noise = (envelope * RNG.standard_normal(clips.shape)).astype(np.float32)
    # quietest 150 ms of the recorded part: everything else is zeroed
    window = 1200
    energy = np.cumsum(np.pad(clips[:, PAD:N - PAD] ** 2, ((0, 0), (1, 0))), axis=1)
    start = (energy[:, window:] - energy[:, :-window]).argmin(1) + PAD
    quiet = np.zeros_like(clips)
    for i, s in enumerate(start):
        quiet[i, N // 2 - window // 2: N // 2 + window // 2] = clips[i, s:s + window]
    return {
        "original": clips,
        "harmonics only": spectral_edit(clips, harmonic),
        "fundamental only": spectral_edit(clips, fundamental),
        "fundamental removed": spectral_edit(clips, ~fundamental),
        "harmonics removed": spectral_edit(clips, ~harmonic),
        "below 150 Hz only": spectral_edit(clips, np.broadcast_to(frequencies < 150, harmonic.shape)),
        "envelope-shaped noise": envelope_noise,
        "quietest 150 ms": quiet,
        "white noise": RNG.standard_normal(clips.shape).astype(np.float32),
    }


def probe_band_occlusion(frame, waveforms, box, frequencies) -> None:
    rows = stratified(frame, 2500)
    clips = np.asarray(waveforms[rows])
    parts = []
    for low in range(0, 4000, 250):
        keep = ~((frequencies >= low) & (frequencies < low + 250))[None, :]
        parts.append(summarise(frame, rows, box(spectral_edit(clips, keep)), band_low_hz=low))
    pd.concat(parts).to_csv(CACHE / "band_occlusion.csv", index=False, float_format="%.5g")
    print("band occlusion done", flush=True)


def probe_noise(frame, waveforms, box, frequencies) -> None:
    rows = stratified(frame, 1500)
    clips = np.asarray(waveforms[rows])
    power = (clips[:, PAD:N - PAD] ** 2).mean(1, keepdims=True)
    noise = RNG.standard_normal(clips.shape).astype(np.float32)
    parts = []
    for snr_db in (40, 30, 20, 15, 10, 5, 0, -5, -10, -15, -20, -30):
        mixed = clips + noise * np.sqrt(power / 10 ** (snr_db / 10))
        parts.append(summarise(frame, rows, box(mixed.astype(np.float32)), snr_db=snr_db))
    pd.concat(parts).to_csv(CACHE / "noise.csv", index=False, float_format="%.5g")
    print("noise done", flush=True)


def probe_cue_conflict(frame, waveforms, box, frequencies) -> None:
    """Harmonic comb of clip A (species X) + everything-but-the-comb of clip B (species Y).

    Both donors are clips the model classifies correctly, so each part carries
    a cue the model is known to accept. Which donor does the prediction follow?
    """
    correct = frame[frame["correct"]]
    rows = stratified(correct, 1500)
    partner = np.empty_like(rows)
    labels = frame.loc[rows, "true_class"].to_numpy()
    for i, label in enumerate(labels):                      # random partner of a different species
        while True:
            j = RNG.integers(len(rows))
            if labels[j] != label:
                partner[i] = rows[j]
                break
    a, b = np.asarray(waveforms[rows]), np.asarray(waveforms[partner])
    comb_a = harmonic_mask(frame.loc[rows, "f0_hz"].to_numpy(), frequencies)
    comb_b = harmonic_mask(frame.loc[partner, "f0_hz"].to_numpy(), frequencies)
    mixed = spectral_edit(a, comb_a) + spectral_edit(b, ~comb_b)
    out = summarise(frame, rows, box(mixed))
    out = out.rename(columns={"true_class": "harmonic_class", "sample_id": "harmonic_sample_id"})
    out["background_class"] = frame.loc[partner, "true_class"].to_numpy()
    out["background_sample_id"] = frame.loc[partner, "sample_id"].to_numpy()
    out["background_f0_hz"] = frame.loc[partner, "f0_hz"].to_numpy()
    out.to_csv(CACHE / "cue_conflict.csv", index=False, float_format="%.5g")
    print("cue conflict done", flush=True)


PROBES = {"sanity": probe_sanity, "projection": probe_projection, "pitch_shift": probe_pitch_shift,
          "ablation": probe_ablation, "band_occlusion": probe_band_occlusion, "noise": probe_noise,
          "cue_conflict": probe_cue_conflict}


def main() -> None:
    frame = pd.read_csv("results/predictions.csv").merge(pd.read_csv(CACHE / "features.csv"), on="sample_id")
    waveforms = np.load("data/prepared/phase1_test_waveforms.npy", mmap_mode="r")
    box = BlackBox()
    frequencies = np.fft.rfftfreq(N, 1 / SR)
    global RNG
    for name in sys.argv[1:] or PROBES:
        RNG = np.random.default_rng(42)          # every probe is reproducible on its own
        PROBES[name](frame, waveforms, box, frequencies)


if __name__ == "__main__":
    main()
