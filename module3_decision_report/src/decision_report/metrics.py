"""Evaluation metrics (numpy only -- no scikit-learn dependency).

All functions are pure and return ``None`` for undefined cases (e.g. AUROC when
only one class is present) rather than raising, so the evaluation panel degrades
gracefully on sparse per-drug slices. These REPORT performance on the test
split; nothing here tunes a threshold.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ClassificationMetrics:
    n: int
    balanced_accuracy: float | None
    recall_resistant: float | None  # sensitivity (positive = resistant)
    recall_susceptible: float | None  # specificity
    precision_resistant: float | None
    f1_resistant: float | None
    accuracy: float | None


def _confusion(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[int, int, int, int]:
    tp = int(np.sum(y_true & y_pred))
    fp = int(np.sum(~y_true & y_pred))
    tn = int(np.sum(~y_true & ~y_pred))
    fn = int(np.sum(y_true & ~y_pred))
    return tp, fp, tn, fn


def classification_metrics(y_true, y_pred) -> ClassificationMetrics:
    """Discrete-label metrics. Inputs are boolean-like (True = resistant)."""
    yt = np.asarray(y_true, dtype=bool)
    yp = np.asarray(y_pred, dtype=bool)
    n = int(yt.size)
    if n == 0:
        return ClassificationMetrics(0, None, None, None, None, None, None)
    tp, fp, tn, fn = _confusion(yt, yp)
    recall_r = tp / (tp + fn) if (tp + fn) > 0 else None
    recall_s = tn / (tn + fp) if (tn + fp) > 0 else None
    precision_r = tp / (tp + fp) if (tp + fp) > 0 else None
    bal_acc = (recall_r + recall_s) / 2 if (recall_r is not None and recall_s is not None) else None
    if precision_r is not None and recall_r is not None and (precision_r + recall_r) > 0:
        f1 = 2 * precision_r * recall_r / (precision_r + recall_r)
    else:
        f1 = None
    accuracy = (tp + tn) / n
    return ClassificationMetrics(n, bal_acc, recall_r, recall_s, precision_r, f1, accuracy)


def _average_ranks(sorted_vals: np.ndarray) -> np.ndarray:
    """Average ranks (1-based) for values already sorted ascending."""
    n = sorted_vals.size
    ranks = np.empty(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sorted_vals[j + 1] == sorted_vals[i]:
            j += 1
        avg = (i + j) / 2.0 + 1.0  # 1-based average rank for the tie group
        ranks[i : j + 1] = avg
        i = j + 1
    return ranks


def auroc(scores, labels) -> float | None:
    """Area under ROC via the rank (Mann-Whitney U) statistic, tie-aware."""
    s = np.asarray(scores, dtype=float)
    y = np.asarray(labels, dtype=bool)
    n_pos = int(y.sum())
    n_neg = int((~y).sum())
    if n_pos == 0 or n_neg == 0 or s.size == 0:
        return None
    order = np.argsort(s, kind="mergesort")
    ranks_sorted = _average_ranks(s[order])
    ranks = np.empty_like(ranks_sorted)
    ranks[order] = ranks_sorted
    sum_ranks_pos = ranks[y].sum()
    return float((sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def average_precision(scores, labels) -> float | None:
    """PR-AUC as average precision (step, no interpolation)."""
    s = np.asarray(scores, dtype=float)
    y = np.asarray(labels, dtype=bool)
    n_pos = int(y.sum())
    if n_pos == 0 or s.size == 0:
        return None
    order = np.argsort(-s, kind="mergesort")
    y_sorted = y[order]
    tp = np.cumsum(y_sorted)
    fp = np.cumsum(~y_sorted)
    precision = tp / np.maximum(tp + fp, 1)
    # AP = sum over positive positions of precision, divided by n_pos.
    return float(precision[y_sorted].sum() / n_pos)


def brier_score(probs, labels) -> float | None:
    p = np.asarray(probs, dtype=float)
    y = np.asarray(labels, dtype=float)
    if p.size == 0:
        return None
    return float(np.mean((p - y) ** 2))


@dataclass(frozen=True)
class ReliabilityBin:
    bin_low: float
    bin_high: float
    mean_predicted: float
    fraction_positive: float
    count: int


def reliability_curve(probs, labels, n_bins: int = 10) -> list[ReliabilityBin]:
    """Binned calibration curve: mean predicted prob vs observed frequency."""
    p = np.asarray(probs, dtype=float)
    y = np.asarray(labels, dtype=float)
    out: list[ReliabilityBin] = []
    if p.size == 0:
        return out
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        if i == n_bins - 1:
            mask = (p >= lo) & (p <= hi)
        else:
            mask = (p >= lo) & (p < hi)
        if not mask.any():
            continue
        out.append(
            ReliabilityBin(
                bin_low=float(lo),
                bin_high=float(hi),
                mean_predicted=float(p[mask].mean()),
                fraction_positive=float(y[mask].mean()),
                count=int(mask.sum()),
            )
        )
    return out
