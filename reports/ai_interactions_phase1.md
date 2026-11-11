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
# Phase I AI Interaction Record – Behaviour Analysis (AI-aided pair)

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
