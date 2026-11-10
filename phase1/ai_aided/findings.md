# Phase I Findings – AI-aided pair

## Central question
Which spectrograms does the model confuse, and why?

---

## Finding 1: *An. arabiensis* is the hardest class – but not because of class size alone

- Lowest test F1: 0.782 (vs. next worst 0.884 for *Ae. albopictus*)
- *An. arabiensis* has 13 507 training samples; *Ae. albopictus* has 14 162 (similar size, 10 F1 points higher)
- **Conclusion:** Class imbalance is a contributing factor but cannot be the whole explanation.

## Finding 2: The *Anopheles* confusion is genus-locked and asymmetric

Confusion matrix reveals:
- *An. gambiae* → *An. arabiensis*: 619/9894 = **6.3%** of *An. gambiae* test samples mislabelled
- *An. arabiensis* → *An. gambiae*: 643/3860 = **16.7%** of *An. arabiensis* test samples mislabelled
- Errors between *Anopheles* and any other genus are negligible (<1%)

**Conclusion:** The model correctly identifies the genus but struggles to separate the two sibling species. The asymmetry means *An. arabiensis* is the harder species to identify — it is more frequently mistaken for *An. gambiae* than vice versa.

## Finding 3: The model is overconfident on its errors

- CDF of confidence shows incorrect predictions still have median confidence ~0.6–0.7
- For *An. arabiensis* errors, the model is not uncertain — it confidently predicts *An. gambiae*
- **Implication:** In a deployment context (e.g., disease-vector monitoring), a high-confidence wrong answer looks identical to a high-confidence right answer. Confidence scores alone cannot be used to flag unreliable predictions.

## Finding 4: Wingbeat frequency overlap explains the spectrogram confusion

- `spectral_peak_hz` (dominant FFT bin) distributions for *An. gambiae* and *An. arabiensis* overlap almost completely in the 400–650 Hz range
- Other species are more separable: *C. quinquefasciatus* is concentrated at ~300–400 Hz, explaining its high F1 of 0.974
- **At 400–650 Hz, accuracy for *An. arabiensis* drops below 80%** — this is the zone where both species sound the same
- **Conclusion:** The core failure is acoustic — the two sibling species have nearly identical wingbeat frequencies, so their log-Mel spectrograms look nearly identical to the model.

## Finding 5: The confused samples are not randomly distributed

- Synthetic spectrogram illustrations (using the model's exact processing pipeline) confirm: at the same fundamental frequency (e.g., 500 Hz), the spectrograms of *An. gambiae* and *An. arabiensis* are visually indistinguishable
- Species identification at these frequencies would require sub-150 Hz discriminating features — possibly harmonic amplitude ratios, temporal modulation, or recording-session-specific noise

---

## Open questions for Phase II (internals)

1. Does CBAM attention activate on the same harmonic lines for both Anopheles species in the overlap zone?
2. Are confused samples clustered in the penultimate feature layer, or scattered?
3. Would training with a contrastive loss specifically on the *Anopheles* pair improve separation?

---

## AI assistance log

This analysis was conducted with Claude (Sonnet 4.6) assistance:
- Suggested the frequency-overlap hypothesis as the main structural question
- Designed the per-frequency-bin accuracy plot to quantify where in the spectrum the model fails
- Wrote the mel-filterbank and spectrogram synthesis code (pure numpy/scipy, mirroring `common/audio.py`)
- Identified the asymmetry in the confusion matrix (An. arabiensis harder than An. gambiae)
