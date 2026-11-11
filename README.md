# Mosquito Wingbeat Classification

This project classifies six mosquito species from their wingbeat recordings. It is part of the Vis4AI / AI4Vis course. We use the trained model to study its predictions, errors and, later, attention maps.

## Dataset

We use the [WINGBEATS dataset](https://www.kaggle.com/datasets/potamitis/wingbeats). After removing duplicate recordings, we have **279,566 recordings** from six mosquito species.

| Split | Recordings |
|---|---:|
| Training | 195,695 (70%) |
| Validation | 27,957 (10%) |
| Test | 55,914 (20%) |

The split is made by recording, so recordings from the same mosquito or recording session may appear in different splits. The prepared dataset is stored in a Modal Volume, not in this repository.

## Model

We use **ResNet-50 with CBAM attention**:

- The audio is converted into log-Mel spectrograms.
- ResNet-50 starts with ImageNet-pretrained weights.
- CBAM attention is added after ResNet stages 2 and 3.
- The output is one of the six mosquito species.

The model is trained on an NVIDIA A100 GPU using [Modal](https://modal.com/). The goal is a working classifier for visualization, not the highest possible accuracy. This is a separate implementation from the original ResNet-9 and WbNet models.

## Training and evaluation

The full training run used **10 epochs** and a batch size of **128**. The best checkpoint was selected using validation macro-F1. It was evaluated on the test set after training.

Results for `r50-cbam-full-001`:

| Test metric | Result |
|---|---:|
| Accuracy | 93.14% |
| Macro-F1 | 90.71% |
| Balanced accuracy | 90.55% |

The selected checkpoint is from epoch 9.

The main outputs are:

- `results/runs/r50-cbam-full-001/best.pt` - trained model.
- `results/predictions.csv` - test predictions, confidence scores and class probabilities.
- `results/runs/r50-cbam-full-001/evaluation/test/metrics.json` - test metrics.
- `results/runs/r50-cbam-full-001/evaluation/test/confusion_counts.csv` - confusion matrix.
- `results/runs/r50-cbam-full-001/history.csv` - training history.

## How to run

Use Python 3.11 and a Modal account with GPU access.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m modal setup
jupyter lab notebooks/01_train_on_modal.ipynb
```

Follow the notebook cells to prepare the dataset, benchmark the GPU, train the model, evaluate it and download the results. Remote jobs can cost money, so check the Modal settings before starting them.

The settings are in `configs/data.json` and `configs/full.json`. The complete training run is already saved, so you do **not** need to train again to start Phase I analysis.

## Repository structure

| Folder | Contents |
|---|---|
| `common/` | Audio processing, model, data loading and inference |
| `training/` | Training, benchmarking, checkpoints and metrics |
| `modal_jobs/` | Modal functions for preparation, training and evaluation |
| `configs/` | Dataset and full-training settings |
| `notebooks/` | Notebook for running training on Modal |
| `results/` | Trained model, predictions and evaluation results |
| `phase1/ai_aided/` | Phase I work with GenAI |
| `phase1/human_only/` | Phase I work without GenAI |
| `reports/` | AI interaction record and reflection |
| `presentation/` | Progress video link |

## Phase I - midpoint analysis

In Phase I, both pairs treated the model as a **black box**, using recordings and predictions to understand its behavior beyond overall accuracy. Both pairs worked with the same six-class model and **55,914 test recordings**, but investigated different questions.

### Human-only pair

We investigated species confusion, prediction confidence, recording characteristics and differences between recording sessions.

- The most common confusion is *An. arabiensis* predicted as *An. gambiae* (**16.7%** of *An. arabiensis* test recordings). The reverse confusion is **6.3%**.
- Wrong predictions generally have lower confidence, but some mistakes are still made with high confidence.
- For *An. arabiensis*, lower **estimated session-based SNR** is associated with more mistakes (**about 18-20%** in lower-SNR groups versus **about 11%** in the highest group). This does not prove that background noise causes the errors.
- Signal strength shows a weaker relationship; edge activity has no clear connection to the mistakes. Performance also varies between recording sessions.

[Analysis notebook](phase1/human_only/analysis.ipynb) | [Detailed findings](phase1/human_only/findings.md)

### AI-aided pair

We explored wingbeat-frequency patterns, confidence, recording sessions and how the model responds to modified audio inputs.

- Some species show different wingbeat-frequency groups with noticeably different classification performance. In particular, the frequencies of *An. gambiae* and *An. arabiensis* overlap strongly.
- Pitch-shifting and frequency-band removal experiments suggest that the model uses both wingbeat pitch and harmonic information, rather than a single frequency alone.
- Errors are concentrated in some recording sessions, and the model can produce confident but incorrect answers. The analyses also investigated how weak or noisy signals affect predictions.
- These experiments show possible reasons for errors, but they do not establish a definite causal explanation.

[Analysis notebook](phase1/ai_aided/analysis.ipynb) | [Additional analysis](phase1/ai_aided/analysis_paul.ipynb) | [Findings](phase1/ai_aided/findings.md) | [Additional findings](phase1/ai_aided/findings_paul.md)

### Midpoint materials

- [AI interaction record](reports/ai_interactions_phase1.md)
- [GenAI reflection](reports/genai_reflection_phase1.md)
- [Progress video link](presentation/video_link.md)

In **Phase II**, we plan to investigate what the model learned by examining its internal representations and CBAM attention maps, focusing on the error patterns found in Phase I. Attention maps alone do not prove why the model made a prediction.

## References

- [WINGBEATS dataset](https://www.kaggle.com/datasets/potamitis/wingbeats)
- [TorchVision ResNet-50](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet50.html)
- [CBAM paper](https://arxiv.org/abs/1807.06521)