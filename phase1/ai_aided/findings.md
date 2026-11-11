# Phase I Findings – AI-aided pair

## Central question
Which spectrograms does the model confuse, and why?

**Setup:** ResNet-50 + CBAM, 55 914 test clips, overall accuracy **93.1%**. Frequency analyses use `f0_hz` (strongest FFT bin in **200–1500 Hz**), not the original `spectral_peak_hz` (often ~6 Hz junk for Anopheles).

---

## Finding 1: *An. arabiensis* is the hardest class – not just class size
*(class-balance / F1 plot)*

| Species | Train *n* | Test F1 |
|---|---:|---:|
| Ae. aegypti | 59 887 | 0.941 |
| Ae. albopictus | 14 162 | 0.884 |
| An. gambiae | 34 630 | 0.923 |
| **An. arabiensis** | **13 507** | **0.782** |
| C. pipiens | 21 290 | 0.938 |
| C. quinquefasciatus | 52 219 | 0.974 |

- *An. arabiensis* and *Ae. albopictus* have similar training sizes, but arabiensis is ~10 F1 points worse.
- **Conclusion:** Imbalance contributes, but does not fully explain the gap.

---

## Finding 2: Confusion is genus-locked and asymmetric
*(confusion matrix)*

- *An. gambiae* → *An. arabiensis*: **619 / 9894 = 6.3%**
- *An. arabiensis* → *An. gambiae*: **643 / 3860 = 16.7%**
- Cross-genus Anopheles errors are rare (~307 of 13 754 Anopheles clips); most mistakes stay inside Anopheles (**1262** mutual confusions).
- **Conclusion:** The model usually gets the genus right, but struggles on the sibling pair — especially *arabiensis*.

---

## Finding 3: The model is overconfident when wrong
*(confidence CDF / boxplots; probability scatter)*

- Incorrect predictions still sit at median confidence ~0.6–0.7.
- Decision margin `|P(gambiae) − P(arabiensis)|`: correct **0.916** vs incorrect **0.340**.
- Still, **54.1%** of Anopheles errors have margin **> 0.3** (confidently wrong).
- **Implication:** Softmax confidence alone is a weak reject signal for deployment.

---

## Finding 4: Wingbeat peaks overlap – with a trustworthy frequency feature
*(f0 histograms; per-bin accuracy)*

- Old `spectral_peak_hz` medians for both Anopheles were **~6 Hz** (unusable). Recomputed `f0_hz` medians: *gambiae* **801.5 Hz**, *arabiensis* **758.5 Hz**.
- Distributions still overlap heavily. In the mass-overlap band **550–900 Hz**:
  - *An. gambiae*: n=7569, accuracy **0.937**
  - *An. arabiensis*: n=2115, accuracy **0.710**
- *C. quinquefasciatus* sits lower (~480 Hz median) and keeps the highest F1 (0.974).
- **Conclusion:** Frequency overlap remains a real issue for Anopheles, but *arabiensis* is weak even where both species have mass — not only in a tiny filtered subset.

---

## Finding 5: H2 (atypical f0 → more errors) is mixed
*(Charts A–C / lift table on full test set)*

| Species | *n* | Typical err | Atypical err | Lift |
|---|---:|---:|---:|---:|
| Ae. aegypti | 17 111 | 0.067 | 0.045 | **0.68** |
| Ae. albopictus | 4 046 | 0.037 | 0.233 | **6.32** |
| An. gambiae | 9 894 | 0.061 | 0.089 | **1.45** |
| An. arabiensis | 3 860 | 0.285 | 0.144 | **0.51** |
| C. pipiens | 6 083 | 0.041 | 0.080 | **1.95** |
| C. quinquefasciatus | 14 920 | 0.023 | 0.031 | **1.36** |

- Strong support for *Ae. albopictus*; mild for *gambiae* / *pipiens*.
- **Reversed for *An. arabiensis***: errors are high near its own median, not mainly in the tails.
- **Conclusion:** “Far from species median ⇒ harder” is **not** the main *arabiensis* story.

---

## Finding 6: Average spectrograms do not reveal an obvious cue
*(real spectrograms; listen; mean / Δ maps)*

- Side-by-side correct vs confused clips in the overlap zone look very similar.
- Mean log-Mel spectrograms (50 clips/category) and `confused − correct` maps show only weak harmonic stripes — nothing visually decisive.
- Listening examples: wrong predictions can still have high confidence (e.g. gambiae→arabiensis at conf 0.91).
- **Conclusion:** Input-space averages alone do not explain the swap; need finer features or model internals (Phase II).

---

## Finding 7: Confused samples partially separate in probability space
*(PCA on 6 class probabilities)*

- Anopheles test *n*=13 754; confused within Anopheles = **1262 (9.2%)**.
- Centroid distance (confused vs correct in PC space): **0.281**.
- **61.9%** of confused clips are closer to the confused centroid than the correct one.
- **Verdict:** Partial sub-cluster — not random, but not a clean separable blob either.

---

## Finding 8: Errors concentrate in some recording sessions (species never share sessions)
*(session bar charts; violin; Lorenz / Gini)*

- Session ID = `D_*` parent folder. *Gambiae* and *arabiensis* share **0** recording sessions.
- Sessions with n≥10: arabiensis 192, gambiae 493.
- Top **10%** of sessions hold **23.3%** of *arabiensis* errors and **37.0%** of *gambiae* errors.
- Gini (error concentration): *gambiae* **0.61**, *arabiensis* **0.33**.
- Session size vs error rate Spearman ≈ 0 (not a small-session artifact).
- Worst *arabiensis* sessions reach error rates ~0.45–0.65 (often late Jan / early Feb mornings in the top-10 list).
- **Conclusion:** Some sessions are much harder; *gambiae* errors are more session-concentrated. This is a recording/batch effect candidate for Phase II — not proof of background leakage by itself.

---

## Finding 9: Time-of-day is not a clear driver for *arabiensis*
*(hour / time-bucket plots)*

- Kruskal–Wallis across night/morning/afternoon/evening: **p=0.40** (n.s.).
- Morning vs other Mann–Whitney: **p=0.14**; medians both **0.20**.
- Hour vs error Spearman: **ρ=-0.05, p=0.46**.
- **Conclusion:** No significant time-of-day effect in this test split. Session identity matters more than clock hour.

---

## Finding 10: *Ae. albopictus* has a few very bad calendar days
*(day-of-month error plot)*

- Days with mean error ≥0.4 for any species include *Ae. albopictus* day **2** (mean err **0.53**, 8 sessions) and day **3** (**0.41**, 20 sessions).
- Fits the strong H2 lift for albopictus: atypical / batch-odd clips hurt that class a lot.
- **Conclusion:** Albopictus failures look more “session/day + atypical f0” than the Anopheles sibling problem.

---

## Bottom line

| Question | Answer from Phase I |
|---|---|
| What is confused? | Mostly *An. gambiae* ↔ *An. arabiensis* (asymmetric). |
| Why acoustically? | Overlapping wingbeat-band peaks; mean spectrograms nearly alike. |
| Is it just atypical f0? | No for *arabiensis* (H2 reversed); partly for other species. |
| Is it random? | No — partial prob-space cluster + high-error sessions. |
| Can confidence save us? | Not reliably (many confident errors). |

---

## Open questions for Phase II (internals)

1. Does CBAM attention fire on the same harmonic lines for both Anopheles in the overlap zone?
2. Are confused samples a tight cluster in the penultimate layer?
3. Would a paired contrastive / hard-negative loss on the Anopheles pair help?
4. Do high-error sessions share background / SNR / fly-through structure (H6/H7)?

---

## AI assistance log

Analysis assisted by Claude / Cursor:
- Frequency-overlap framing and per-bin accuracy design
- Diagnosed broken `spectral_peak_hz`; recomputed band-limited `f0_hz` for all 55 914 clips
- H2 atypicality tests, session uniformity (Gini/Lorenz), spectrogram tooling aligned with `common/audio.py`
- Interpretation of null/weak plots (mean Δ maps, time-of-day) so negative results stay in the report
