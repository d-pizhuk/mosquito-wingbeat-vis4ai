"""Sample-weighted loss and fixed-six-class metrics, including absent classes."""
from __future__ import annotations
import numpy as np
import torch
from common.class_names import CLASS_NAMES, NUM_CLASSES


def metrics_from_confusion(matrix) -> dict:
    matrix = np.asarray(matrix, dtype=np.int64)
    if matrix.shape != (NUM_CLASSES, NUM_CLASSES):
        raise ValueError("Expected a six-class confusion matrix")
    true_positive = matrix.diagonal().astype(np.float64)
    support = matrix.sum(axis=1)
    predicted = matrix.sum(axis=0)
    precision = np.divide(true_positive, predicted, out=np.zeros(NUM_CLASSES), where=predicted > 0)
    recall = np.divide(true_positive, support, out=np.zeros(NUM_CLASSES), where=support > 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros(NUM_CLASSES), where=(precision + recall) > 0)
    return {
        "accuracy": float(true_positive.sum() / max(1, matrix.sum())),
        "macro_f1": float(f1.mean()), "balanced_accuracy": float(recall.mean()),
        "samples": int(matrix.sum()), "confusion_matrix": matrix.tolist(),
        "per_class": {CLASS_NAMES[i]: {"precision": float(precision[i]), "recall": float(recall[i]),
                                      "f1": float(f1[i]), "support": int(support[i])}
                      for i in range(NUM_CLASSES)},
    }


class Meter:
    def __init__(self, state: dict | None = None):
        state = state or {}
        self.loss_sum = float(state.get("loss_sum", 0))
        self.count = int(state.get("count", 0))
        self.matrix = np.asarray(state.get("matrix", np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)), dtype=np.int64)

    def update(self, logits: torch.Tensor, target: torch.Tensor, losses: torch.Tensor) -> None:
        n = len(target)
        if not torch.isfinite(losses).all():
            raise FloatingPointError("Non-finite loss")
        self.loss_sum += float(losses.detach().sum().item())
        self.count += n
        flat = (target.detach() * NUM_CLASSES + logits.detach().argmax(dim=1)).cpu().numpy()
        self.matrix += np.bincount(flat, minlength=NUM_CLASSES**2).reshape(NUM_CLASSES, NUM_CLASSES)

    def result(self) -> dict:
        return {**metrics_from_confusion(self.matrix), "loss": self.loss_sum / max(1, self.count)}

    def state(self) -> dict:
        return {"loss_sum": self.loss_sum, "count": self.count, "matrix": self.matrix.tolist()}
