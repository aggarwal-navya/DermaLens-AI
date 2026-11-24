"""Open-set decisions: should the app name a condition, or say "not identified"?

Three layers, cheapest first:
    1. skin_pixel_ratio  - is this a photo of skin at all? (YCrCb colour rule)
    2. "Other" class     - the model was trained on other diseases (outlier exposure)
    3. thresholds        - low confidence or high energy score -> not identified

Thresholds are tuned on the validation set (notebook 10), never on test.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageFilter

from .labels import OTHER_CLASS


def energy_score(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Negative free energy (Liu et al., 2020). Higher = more out-of-distribution."""
    z = np.asarray(logits, dtype=np.float64) / temperature
    peak = z.max(axis=1, keepdims=True)
    return -(temperature * (peak[:, 0] + np.log(np.exp(z - peak).sum(axis=1))))


def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    z = np.asarray(logits, dtype=np.float64) / temperature
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def skin_pixel_ratio(image: Image.Image, size: int = 128) -> float:
    """Share of pixels inside a broad skin-colour range in YCrCb space.

    Works for light to very dark skin (Cr 133-180, Cb 77-135; Chai & Ngan 1999,
    widened). A cat, a car or a screenshot usually scores far below 0.25.
    """
    # A light blur first: real skin is smooth, so it barely changes, while noise/texture collapses to grey.
    small = image.convert("RGB").resize((size, size)).filter(ImageFilter.GaussianBlur(2)).convert("YCbCr")
    ycbcr = np.asarray(small, dtype=np.int16)
    y, cb, cr = ycbcr[..., 0], ycbcr[..., 1], ycbcr[..., 2]
    mask = (cr >= 133) & (cr <= 180) & (cb >= 77) & (cb <= 135) & (y >= 25)
    return float(mask.mean())


def tune_threshold(scores_in: np.ndarray, target_tpr: float = 0.95) -> float:
    """Confidence threshold that keeps `target_tpr` of in-distribution validation images."""
    return float(np.quantile(np.asarray(scores_in), 1 - target_tpr))


def open_set_auroc(scores_in: np.ndarray, scores_out: np.ndarray) -> float:
    """AUROC for telling known classes (higher score) from unknown ones."""
    from sklearn.metrics import roc_auc_score

    y = np.r_[np.ones(len(scores_in)), np.zeros(len(scores_out))]
    return float(roc_auc_score(y, np.r_[scores_in, scores_out]))


def decide(probabilities: np.ndarray, classes: list[str], confidence_threshold: float,
           skin_ratio: float | None = None, min_skin_ratio: float = 0.15,
           healthy_label: str = "Healthy") -> tuple[str, str | None]:
    """Turn one probability vector into (status, label).

    status is one of: not_skin, not_identified, healthy, identified.
    'Other' probability is folded into the decision, never shown as a diagnosis.
    """
    if skin_ratio is not None and skin_ratio < min_skin_ratio:
        return "not_skin", None
    probabilities = np.asarray(probabilities)
    best = int(probabilities.argmax())
    label = classes[best]
    if label == OTHER_CLASS or probabilities[best] < confidence_threshold:
        return "not_identified", None
    if label == healthy_label:
        return "healthy", label
    return "identified", label
