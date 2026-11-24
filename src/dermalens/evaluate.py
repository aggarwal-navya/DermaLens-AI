"""Measuring a trained model honestly.

- per-class precision / recall / F1 (accuracy alone hides weak rare classes)
- 95% confidence intervals by bootstrap, resampling whole patients
- results split by source dataset and by skin type (fairness)
- calibration: does "90% confident" really mean right 90% of the time?
"""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)


# ---------------------------------------------------------------------------
# Predictions
# ---------------------------------------------------------------------------
def predict_logits(model, loader, device: str, tta: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Run the model over a loader of (images, metadata, targets).

    tta=True averages logits over 4 views (original, horizontal flip,
    vertical flip, both) - test-time augmentation, usually +0.5-1.5 points.
    Returns (logits [N, C], true class indices [N]).
    """
    import torch

    model.eval()
    all_logits, all_targets = [], []
    with torch.no_grad():
        for images, metas, targets in loader:
            images, metas = images.to(device), metas.to(device)
            views = [images] if not tta else [images, images.flip(3), images.flip(2), images.flip(2).flip(3)]
            logits = torch.stack([model(view, metas).float() for view in views]).mean(0)
            all_logits.append(logits.cpu().numpy())
            all_targets.append(targets.numpy())
    return np.concatenate(all_logits), np.concatenate(all_targets)


def ensemble_probabilities(logit_list: list[np.ndarray], temperatures: list[float] | None = None,
                           weights: list[float] | None = None) -> np.ndarray:
    """Average calibrated probabilities of several models (soft voting)."""
    temperatures = temperatures or [1.0] * len(logit_list)
    weights = np.asarray(weights or [1.0] * len(logit_list), dtype=float)
    stacked = np.stack([softmax(z, t) for z, t in zip(logit_list, temperatures, strict=False)])
    return np.tensordot(weights / weights.sum(), stacked, axes=1)


def grouped_metrics(y_true: np.ndarray, y_pred: np.ndarray, classes: list[str],
                    mapping: dict[str, str]) -> dict[str, float]:
    """Headline metrics after merging classes (e.g. Contact Dermatitis + Eczema)."""
    names = np.array([mapping.get(c, c) for c in classes])
    return headline_metrics(names[y_true], names[y_pred])


def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Turn raw scores into probabilities that sum to 1 per image."""
    scaled = logits / temperature
    scaled = scaled - scaled.max(axis=1, keepdims=True)    # numerical safety
    exp = np.exp(scaled)
    return exp / exp.sum(axis=1, keepdims=True)


# ---------------------------------------------------------------------------
# Core metrics
# ---------------------------------------------------------------------------
def headline_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """The four numbers reported for every model."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted"),
    }


def per_class_table(y_true: np.ndarray, y_pred: np.ndarray, classes: list[str]) -> pd.DataFrame:
    """Precision, recall, F1 and number of test images for each class."""
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=range(len(classes)), zero_division=0)
    table = pd.DataFrame({"precision": precision, "recall": recall, "f1": f1, "images": support},
                         index=pd.Index(classes, name="class"))
    return table.round(3)


def bootstrap_ci(y_true: np.ndarray, y_pred: np.ndarray, groups: np.ndarray | None = None,
                 metric=lambda t, p: f1_score(t, p, average="macro"),
                 n_boot: int = 1000, seed: int = 42) -> tuple[float, float, float]:
    """Point estimate and 95% interval of a metric.

    The test set is redrawn `n_boot` times with replacement. When `groups` is
    given, whole patients are redrawn (photos of one patient are not independent),
    which gives a more honest, wider interval.
    """
    rng = np.random.default_rng(seed)
    point = float(metric(y_true, y_pred))
    if groups is None:
        groups = np.arange(len(y_true))
    unique_groups = np.unique(groups)
    rows_of = {g: np.flatnonzero(groups == g) for g in unique_groups}

    scores = []
    with warnings.catch_warnings():               # a resample may miss a rare class: expected, not an error
        warnings.simplefilter("ignore")
        for _ in range(n_boot):
            picked = rng.choice(unique_groups, size=len(unique_groups), replace=True)
            rows = np.concatenate([rows_of[g] for g in picked])
            scores.append(metric(y_true[rows], y_pred[rows]))
    # A metric may return NaN for a resample where it is undefined (e.g. no image of one class): skip those.
    low, high = np.nanpercentile(np.asarray(scores, dtype=float), [2.5, 97.5])
    return point, float(low), float(high)


def metrics_by_group(table: pd.DataFrame, group_col: str, classes: list[str]) -> pd.DataFrame:
    """Accuracy and macro-F1 inside each subgroup (e.g. per source or per skin type).

    `table` needs columns y_true, y_pred and `group_col`.
    """
    rows = []
    for value, part in table.groupby(group_col, dropna=False):
        rows.append({
            group_col: value, "images": len(part),
            "accuracy": accuracy_score(part["y_true"], part["y_pred"]),
            "macro_f1": f1_score(part["y_true"], part["y_pred"], average="macro",
                                 labels=sorted(part["y_true"].unique())),
        })
    return pd.DataFrame(rows).round(3)


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------
def fit_temperature(val_logits: np.ndarray, val_targets: np.ndarray) -> float:
    """Find T so that softmax(logits / T) gives honest confidences on validation data.

    T > 1 softens over-confident predictions. T never changes which class wins,
    only how confident the model claims to be.
    """
    from scipy.optimize import minimize_scalar

    def negative_log_likelihood(temperature: float) -> float:
        probs = softmax(val_logits, temperature)
        return -np.log(probs[np.arange(len(val_targets)), val_targets] + 1e-12).mean()

    result = minimize_scalar(negative_log_likelihood, bounds=(0.05, 10.0), method="bounded")
    return float(result.x)


def expected_calibration_error(probs: np.ndarray, y_true: np.ndarray, n_bins: int = 10) -> float:
    """Average gap between confidence and actual accuracy (0 = perfectly calibrated)."""
    confidence, predicted = probs.max(axis=1), probs.argmax(axis=1)
    edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for low, high in zip(edges[:-1], edges[1:], strict=False):
        in_bin = (confidence > low) & (confidence <= high)
        if in_bin.any():
            gap = abs((predicted[in_bin] == y_true[in_bin]).mean() - confidence[in_bin].mean())
            ece += in_bin.mean() * gap
    return float(ece)


# ---------------------------------------------------------------------------
# Plots (each saves a PNG for the report and returns the figure)
# ---------------------------------------------------------------------------
def plot_confusion_matrix(y_true, y_pred, classes, path: Path | None = None, title: str = ""):
    """Rows = true class, columns = predicted class; each row shows % of that class."""
    matrix = confusion_matrix(y_true, y_pred, labels=range(len(classes)))
    shares = matrix / matrix.sum(axis=1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.imshow(shares, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, f"{matrix[i, j]}\n{shares[i, j]:.0%}", ha="center", va="center", fontsize=8,
                    color="white" if shares[i, j] > 0.6 else "black")
    ax.set_xticks(range(len(classes)), classes, rotation=35, ha="right")
    ax.set_yticks(range(len(classes)), classes)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title or "Confusion matrix (count and row %)")
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150)
    return fig


def plot_training_curves(history: pd.DataFrame, path: Path | None = None, title: str = ""):
    """Loss and macro-F1 per epoch, train vs validation, with the stage switch marked."""
    fig, (ax_loss, ax_f1) = plt.subplots(1, 2, figsize=(12, 4))
    ax_loss.plot(history["epoch"], history["train_loss"], label="train")
    ax_loss.plot(history["epoch"], history["val_loss"], label="validation")
    ax_loss.set_title("Loss")
    ax_loss.set_xlabel("epoch")
    ax_loss.legend()
    ax_f1.plot(history["epoch"], history["val_macro_f1"], color="tab:green", label="val macro-F1")
    ax_f1.plot(history["epoch"], history["train_accuracy"], color="tab:gray", ls="--", label="train accuracy")
    ax_f1.set_title("Validation macro-F1")
    ax_f1.set_xlabel("epoch")
    ax_f1.set_ylim(0, 1)
    ax_f1.legend()
    warmup_end = history.loc[history["stage"].isin(["warmup", "frozen"]), "epoch"].max()
    if pd.notna(warmup_end):
        for ax in (ax_loss, ax_f1):
            ax.axvline(warmup_end + 0.5, color="gray", ls=":", lw=1)
    fig.suptitle(title)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150)
    return fig


def plot_reliability(probs: np.ndarray, y_true: np.ndarray, path: Path | None = None,
                     n_bins: int = 10, title: str = ""):
    """Confidence vs real accuracy; a calibrated model follows the diagonal."""
    confidence, correct = probs.max(axis=1), probs.argmax(axis=1) == y_true
    edges = np.linspace(0, 1, n_bins + 1)
    centres, accuracies = [], []
    for low, high in zip(edges[:-1], edges[1:], strict=False):
        in_bin = (confidence > low) & (confidence <= high)
        if in_bin.any():
            centres.append(confidence[in_bin].mean())
            accuracies.append(correct[in_bin].mean())
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], ls="--", color="gray", label="perfect")
    ax.plot(centres, accuracies, marker="o", label="model")
    ax.set_xlabel("confidence")
    ax.set_ylabel("accuracy")
    ax.legend()
    ax.set_title(title or f"Reliability (ECE = {expected_calibration_error(probs, y_true, n_bins):.3f})")
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150)
    return fig
