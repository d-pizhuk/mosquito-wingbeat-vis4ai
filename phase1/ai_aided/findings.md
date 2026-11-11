# Phase I findings (AI-aided pair): how does the wingbeat classifier behave?

Model: ResNet-50 + CBAM on log-Mel spectrograms, run `r50-cbam-full-001`. Data: the 55,914 test
recordings (93.1% accuracy, 3,834 errors). The model is treated as a black box. All figures and
their code are in `analysis.ipynb`; image files are in `figures/`.

## Method

1. **Observation.** Every test clip was decoded with `common.audio.decode_audio` and measured:
   wingbeat pitch (fundamental frequency, harmonic-comb estimate), harmonic levels, harmonic-to-noise
   ratio, how long the insect is audible, plus recording day and session from the file name. These
   were joined with `results/predictions.csv`.
2. **Intervention.** The checkpoint was run locally on edited copies of test clips: pitch shifted
   by -40% to +60%, harmonics removed or isolated, single 250 Hz bands silenced, white noise added.
   The local model reproduces the saved predictions on 99.95% of clips.

The `spectral_peak_hz` column of `predictions.csv` was not used: it is below 50 Hz for 84% of the
clips (sensor drift, not wingbeat).

## Findings

| # | Behaviour | Key numbers | Figure |
|---|---|---|---|
| 1 | Errors form two systems, not a uniform spread | *An. gambiae* and *An. arabiensis* exchange 1,262 clips (33% of errors); *Ae. aegypti* sends or receives 2,008 (52%) | 1 |
| 2 | Three species have two pitch modes (near 500-600 Hz and 750-850 Hz) and the model serves the majority mode | *Ae. albopictus*: 95% recall in the high mode, 66% in the low mode. *An. arabiensis*: 86% low, 71% high | 2 |
| 3 | For *Aedes* and *Culex*, pitch drives the answer | Shifting *C. quinquefasciatus* up 15% gives 66% *Ae. aegypti* answers; *C. pipiens* up 25% gives 84% *Ae. albopictus*, down 30% gives 74% *C. quinquefasciatus* | 3 |
| 4 | The weak *Ae. albopictus* mode is a specific sub-population: two recording days (2-3 January 2017), 55 days before the rest | 567 clips (14% of the species), median pitch 554 Hz inside the *Ae. aegypti* range, 56% recall, 38% answered as *Ae. aegypti*; these are the cleanest recordings of the species | 4 |
| 5 | Input space is one continuous pitch gradient with no island per species; errors sit where species share a pitch region | Low-pitch *Ae. albopictus* 34% wrong, high-pitch *An. arabiensis* 29% wrong | 5 |
| 6 | The model does not need the fundamental, but needs both the harmonics and what lies between them | Fundamental removed: 91% to 87% accuracy. Harmonics removed: 33%. Harmonics only: 33% | 6 |
| 7 | Each species depends on particular harmonics | Silencing 750-1000 Hz costs *Ae. albopictus* 27 points (its fundamental); 1000-1250 Hz costs *C. pipiens* 28 and 750-1000 Hz costs *C. quinquefasciatus* 20 (their second harmonics). *An. arabiensis* is sensitive across the whole spectrum | 7 |
| 8 | Weak or missing signal gets a default answer | With added noise at -10 dB, 72% of answers are *Ae. aegypti*. A 150 ms fragment is called *Ae. albopictus* 79% of the time with median confidence 0.84 | 6, 8 |
| 9 | The *Anopheles* pair is separable by pitch only in the low mode; in the high mode the model sides with the majority species | Pitch-only AUC 0.87 (low) against 0.60 (high); model AUC 0.98 against 0.94; high-mode recall 93% for *An. gambiae*, 71% for *An. arabiensis* | 9 |

## What this adds beyond the benchmark table

- The headline recall of *Ae. albopictus* (87%) and *An. arabiensis* (79%) each hides two groups
  with very different reliability. The groups are consistent with the usual female and male
  wingbeat ranges, but the dataset has no sex label, so this is an interpretation.
- Class weighting balanced the species during training, not the sub-populations inside a species.
  Where a minority group overlaps a larger species in pitch, the model follows the majority.
- *Ae. aegypti* looks like a strong class (94% recall), but part of that comes from being the
  model's default for degraded input: it keeps 70% recall at -10 dB noise while every other
  species is below 25%.
- We looked for evidence that the model recognises the recording session instead of the insect
  and did not find it: with the harmonics filtered out, accuracy falls to 33%.

## Limits

- The split is random per recording, so the same session is in training and test. All numbers
  are optimistic for unseen sessions, and finding 4 would probably be worse on a session-level split.
- Pitch shifting by resampling also changes how long the pass lasts.
- Spectral edits create inputs unlike any real recording; they show what the model can be pushed
  to do, not how often it happens.
- A cue-conflict probe (harmonics of one clip mixed with the residue of a clip from another
  species, `cache/cue_conflict.csv`) was inconclusive: 39% of answers matched neither donor. It is
  not shown in the notebook.
- The pitch estimate is ours and was checked only against the expected species ranges, not against
  ground truth.

## Reproduce

```bash
pip install -r phase1/ai_aided/requirements.txt
python -m phase1.ai_aided.extract_features --audio-root <folder containing Wingbeats/>   # about 15 s
python -m phase1.ai_aided.run_probes                                                      # about 6 min on a GPU
jupyter lab phase1/ai_aided/analysis.ipynb
```

`phase1/ai_aided/cache/` (155 MB) and `data/prepared/` (1.1 GB) are generated and git-ignored.
