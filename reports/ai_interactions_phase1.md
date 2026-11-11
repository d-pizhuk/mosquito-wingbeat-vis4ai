# Phase I AI Interaction Record – Model Setup

## AI Interaction

**User request:** Develop a ResNet-50 model with attention for the WINGBEATS dataset, using Modal for training and Jupyter notebooks for monitoring. The goal is to achieve good accuracy and support visualizations, with a budget of around $30.

**AI contribution:** Generated code for dataset preparation, ResNet-50 with CBAM attention, training, evaluation, Modal integration, notebooks, and tests.

**Main design decisions:**
- Use ImageNet-pretrained ResNet-50 with CBAM attention.
- Train the model using the full WINGBEATS training dataset.
- Prepare data on CPUs and train on an NVIDIA A100 GPU.
- Split data into training (70%), validation (10%), and test (20%) sets.
- Save checkpoints and support attention visualizations for Phase II.

**Verification:** The implementation was initially tested locally using synthetic data. We successfully prepared the WINGBEATS dataset, verified training on Modal, measured GPU performance, and tested model evaluation.

**Issues and limitations:** We fixed a Modal disk configuration error. Duplicate recordings were detected and removed. Attention maps also need careful interpretation, as they do not necessarily explain why the model makes a prediction.

## Next Steps

Both pairs will use the same trained model and dataset but investigate its behavior separately. The AI-aided pair will document further AI interactions, suggestions, corrections, and findings.

---

# Phase I AI Interaction Record – Behaviour Analysis (AI-aided pair, Andrew)

Tool: Claude Code (Claude Fable 5.2), 10–11 November 2026, one session with four requests.

## Interaction 1 – starting point

**User request:** Complete Phase I of the assignment (the task text was pasted) using the repository.

**AI contribution:** Read the repository, found that only predictions and the checkpoint were present (no audio), parsed recording session and time from file names and reported a first lead: *Ae. albopictus* accuracy rises from 46% to 96% across recording days. Started installing an analysis environment; the user interrupted and redirected.

## Interaction 2 – ideas

**User request:** The audio is in `../archive/Wingbeats`; give ideas that can be visualized for Phase I.

**AI contribution:** Eight ideas, each checked against the predictions first (sub-populations by session, sibling-species confusion, "almost right" errors, quiet recordings, *Ae. aegypti* as default answer, input map, background-versus-insect probe, pitch-shift probe) and a suggested storyline.

## Interaction 3 – measurable hypotheses

**User request:** Pick 7–8 measurable ideas that can be visualized and proved with spectrograms and audio characteristics.

**AI contribution:** Eight hypotheses in a table: what to measure, which chart, and what would count as proof. The AI stated which three already had support and which five were untested.

## Interaction 4 – visualizations

**User request:** Create 5–10 visualizations with code in `analysis.ipynb`; be creative; decode audio with `decode_audio` if needed.

**AI contribution:**
- `phase1/ai_aided/extract_features.py`: decodes the 55,914 test clips and measures pitch, harmonics, noise and event duration.
- `phase1/ai_aided/run_probes.py`: runs the checkpoint on edited audio (pitch shift, harmonic removal, band occlusion, added noise) and computes a UMAP of the inputs.
- `phase1/ai_aided/analysis.ipynb`: nine figures with commentary; `findings.md`: summary table.

**Design decisions made by the AI:** own pitch estimator instead of the unusable `spectral_peak_hz` column; pairing each observed pattern with an intervention; one fixed colour per species; finding stated in each figure title.

**Verification:** the local model reproduces the saved predictions on 99.95% of clips; the notebook executes end to end without errors; every figure was rendered and inspected.

**Corrections the AI made to its own work:**
- The first harmonic filter was too narrow and left the harmonics visible in the spectrogram. This had suggested that four species are recognised from the background alone. With a wider filter, accuracy without harmonics is 33%, and the claim was withdrawn.
- A figure title said the model "does not hedge" on noisy input. Checking confidence showed it does (median 0.94 falls to 0.48), so the title and text were changed.
- A cue-conflict experiment gave an unclear result (39% of answers matched neither source clip) and was left out of the figures; it is listed under limits in `findings.md`.

**Points for the pair to check by hand:** the reading of the two pitch modes as female/male is an interpretation (no sex labels); the pitch estimate has no ground truth; hypothesis 6 from interaction 3 (truncated passes) was only covered through event duration, not edge truncation.

---

# Phase I AI Interaction Record – Analysis (AI-aided pair, Paul)

## Overview

We designed and led the Phase I black-box analysis, using Claude (Anthropic) as an implementation tool to accelerate coding and run analyses in parallel. The investigation questions, hypotheses, and analytical decisions were ours; the AI handled code generation and execution so we could focus on interpretation and direction.

---

## Interaction 1 – Initial exploratory analysis

**Our approach:** We structured the initial exploration around four questions: which species are hardest to classify, where does confusion concentrate in the confusion matrix, how confident is the model when it is wrong, and do wingbeat frequencies separate the problem species? We directed the AI to implement visualisations for each.

**AI assistance:** Implemented notebook cells for class balance / F1 per species, an annotated confusion matrix, a confidence CDF split by correct/incorrect, and frequency histograms.

**Key decision by us:** Inspecting the frequency histograms, we noticed `spectral_peak_hz` values were clustered around ~6 Hz for both Anopheles species — clearly meaningless for wingbeat analysis. We identified this as a broken feature and decided to recompute a reliable wingbeat peak (`f0_hz`) restricted to the 200–1500 Hz band. The AI implemented the recomputation; we validated the resulting medians (~758 Hz arabiensis, ~802 Hz gambiae) against known biology before using the feature in any further analysis.

---

## Interaction 2 – Hypothesis testing on frequency atypicality (H2)

**Our approach:** We hypothesised that clips with an unusual wingbeat frequency (far from their species median) would be harder to classify — a natural extension of the overlap-zone finding. We asked the AI to test this per species using an IQR-based atypicality threshold.

**AI assistance:** Implemented the atypicality split (|f0_hz − median| > 1.5 × IQR), computed error rate lift per species, and produced a summary table and bar chart.

**Our interpretation:** The result for An. arabiensis was the opposite of the hypothesis — atypical clips were *easier*, not harder. We recognised this as a meaningful negative result: arabiensis errors concentrate near its own median frequency, not in the tails. Rather than discarding the finding, we decided to include it explicitly (Finding 5 in findings.md) because it rules out a simple frequency-based explanation for the arabiensis problem.

---

## Interaction 3 – Deeper analysis in four directions (parallel subagents)

**Our approach:** After the initial exploration we identified four directions worth pursuing in depth: the structure of the probability space (PCA), whether errors cluster in specific recording sessions, the decision boundary between the two Anopheles species, and how uniformly errors are distributed across sessions. We designed all four directions and ran them simultaneously using parallel AI subagents to save time.

**AI assistance:** Implemented each direction as a separate notebook section — PCA via numpy SVD, session error bar charts and violin plots, probability scatter with decision margin histogram, and Lorenz curves with Gini coefficients.

**Our interpretation:** Comparing the Gini coefficients (gambiae 0.61, arabiensis 0.33) we recognised that the two species have fundamentally different failure modes — not just different error rates. This distinction — chronic vs session-concentrated — was our framing, and it became the central structural insight of the analysis.

**Critical check by us:** The gambiae violin plot appeared heavily squashed toward zero, which looked like a rendering issue. We investigated and confirmed it was not a bug but the actual distribution — most gambiae sessions have near-zero errors, with a handful of extreme outliers pulling the mean up. Keeping this result rather than dismissing it as a visual artefact was our call.

---

## Interaction 4 – Session metadata and statistical validation

**Our approach:** We noticed the session IDs encode a timestamp (`D_YY_MM_DD_HH_MM_SS`) and suspected a time-of-day effect might explain why some arabiensis sessions are worse. We asked the AI to extract the timestamps and compare the worst and best sessions. When the initial metadata table appeared to show a morning cluster, we explicitly decided that visual inspection was not sufficient — we needed proper statistical tests before claiming a time-of-day effect.

**AI assistance:** Implemented a regex parser for session IDs, produced a metadata comparison table (worst vs best sessions by error rate), and ran three statistical tests: Kruskal–Wallis across four time buckets, Mann–Whitney U (morning vs other), and Spearman correlation (hour vs error rate).

**Our interpretation:** All three tests returned non-significant results (p=0.40, p=0.14, ρ=–0.05). We accepted this as a genuine null result — the visual pattern at hours 11–12 did not survive statistical scrutiny. We included the null result in findings.md rather than dropping it, because ruling out time-of-day as a cause is itself informative.

---

## Interaction 5 – Per-species session size analysis and day-of-month plot

**Our approach:** A pooled session plot obscures species-level patterns, so we decided each species should be shown individually. We also wanted to test whether errors could be explained by a specific bad recording day — an equipment failure or unusual conditions on one date — rather than a persistent species-level problem.

**AI assistance:** Produced a 2×3 grid of session size vs error rate scatterplots (one per species, top 3 worst sessions labelled) and a day-of-month vs error rate plot with labels for days where error ≥ 0.4 or > 2× the species mean.

**Our interpretation:** Only 2 days received labels (Ae. albopictus days 2–3, n=8 and n=20 clips). We recognised that these tiny sample sizes make the apparent error spike meaningless — a single confused batch inflates a small-n day to 50% error. More importantly, arabiensis sat consistently elevated across every single day of recording. We concluded the arabiensis problem is chronic and structural, not an artefact of one bad recording session.

---

## Summary of human vs AI contribution

| Area | Our contribution | AI role |
|---|---|---|
| Investigation design | Defined all hypotheses, chose analysis directions, decided what to pursue deeper | None — we set the agenda |
| Feature engineering | Identified `spectral_peak_hz` as broken; decided to recompute `f0_hz` with a biologically motivated frequency band | Implemented the recomputation |
| Statistical rigour | Decided visual inspection was insufficient; insisted on formal tests before claiming a time-of-day effect | Ran the tests and reported results |
| Findings synthesis | Interpreted the Gini asymmetry as two different failure modes; framed arabiensis as chronic vs gambiae as session-specific | No framing — produced numbers |
| Null results | Decided to keep three null results (H2 arabiensis, time-of-day, no bad day) as genuine findings rather than discarding them | Flagged the results |
| Critical checks | Caught the gambiae violin skew as a real finding, not a bug; validated `f0_hz` medians against biology | Implemented what we asked |
| Code and figures | — | Wrote all notebook cells and produced all figures |

The AI did not access model weights, attention maps, or training logs — all analysis was black-box (predictions.csv only).