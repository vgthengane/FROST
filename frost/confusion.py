"""Confusion-matrix helpers for train-side conflict replay (paper path)."""

from __future__ import annotations

import numpy as np
import torch


def build_confusion_matrix(
    labels: torch.Tensor | np.ndarray,
    preds: torch.Tensor | np.ndarray,
    num_classes: int,
) -> np.ndarray:
    """Row-normalized confusion matrix ``[num_classes, num_classes]``."""
    labels_np = labels.cpu().numpy() if torch.is_tensor(labels) else np.asarray(labels)
    preds_np = preds.cpu().numpy() if torch.is_tensor(preds) else np.asarray(preds)
    cm = np.zeros((num_classes, num_classes), dtype=np.float64)
    for true_l, pred_l in zip(labels_np, preds_np):
        if 0 <= int(true_l) < num_classes and 0 <= int(pred_l) < num_classes:
            cm[int(true_l), int(pred_l)] += 1.0
    row_sum = cm.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1.0
    return cm / row_sum


def per_class_recall(cm: np.ndarray) -> np.ndarray:
    """Diagonal of a row-normalized confusion matrix (per-class recall)."""
    return np.diag(cm)
