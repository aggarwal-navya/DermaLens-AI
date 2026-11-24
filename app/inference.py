"""Prediction logic for the web app, kept separate from the HTTP layer so it can be tested alone.

    predictor = Predictor("models/deploy")
    result = predictor.predict(pil_image, age_group="20-40", sex="male", body_regions=["hand"])

Steps for one photo:
    1. skin check (colour rule) -> "not_skin" if the photo is not skin
    2. every ensemble member: calibrated probabilities (with metadata if given)
    3. average -> decide identified / healthy / not_identified (Other class or low confidence)
    4. Grad-CAM heatmap from the first member
"""

from __future__ import annotations

import base64
import io
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from dermalens import metadata as meta  # noqa: E402
from dermalens import ood  # noqa: E402
from dermalens.data import build_transforms, center_square  # noqa: E402
from dermalens.explain import GradCAM, overlay_heatmap  # noqa: E402
from dermalens.export import DISCLAIMER, load_bundle  # noqa: E402
from dermalens.labels import OTHER_CLASS  # noqa: E402
from dermalens.model import gradcam_target_layer  # noqa: E402

try:
    from .conditions import CONDITIONS, DISPLAY_NAMES
except ImportError:                                   # run as a plain script
    from conditions import CONDITIONS, DISPLAY_NAMES  # type: ignore

APP_VERSION = "3.1.0"

MESSAGES = {
    "identified": "The photo most closely matches {label}.",
    "healthy": "No sign of the five covered skin conditions was found.",
    "not_identified": "Not identified - this may be a condition DermaLens does not cover, or the photo is "
                      "unclear. Please consult a dermatologist.",
    "not_skin": "This does not look like a photo of skin. Please upload a close, well-lit photo of the "
                "affected area.",
}


class Predictor:
    """Loads the model bundle once and answers predictions."""

    def __init__(self, bundle_dir: str | Path, device: str = "cpu"):
        import torch

        torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
        self.members, self.card = load_bundle(Path(bundle_dir), device)
        self.device = device
        self.classes: list[str] = self.card["classes"]
        self.threshold = float(self.card.get("confidence_threshold", 0.5))
        self.min_skin_ratio = float(self.card.get("min_skin_ratio", 0.15))
        color = bool(self.card.get("color_constancy", True))
        self.transforms = [build_transforms(m["image_size"], train=False, color_constancy=color)
                           for _, m in self.members]
        first_model = self.members[0][0]
        try:
            self.cam = GradCAM(first_model, gradcam_target_layer(first_model))
        except ValueError:
            self.cam = None

    # ------------------------------------------------------------------
    def info(self) -> dict:
        """Everything the site needs to build its pages and forms (see docs/API_CONTRACT.md)."""
        metrics = self.card.get("test_metrics", {})
        names = [m["model_name"] for _, m in self.members]
        shown = [c for c in self.classes if c != OTHER_CLASS]
        return {
            "app_version": APP_VERSION,
            "model": {"name": ("ensemble: " if len(names) > 1 else "") + " + ".join(names),
                      "exported_on": self.card.get("exported_on"),
                      "uses_metadata": bool(self.card.get("uses_metadata", False)),
                      "image_size": self.members[0][1]["image_size"]},
            "classes": shown,
            "metrics": metrics,
            "metadata_options": meta.options_for_app(),
            "conditions": {c: CONDITIONS[c] for c in shown if c in CONDITIONS},
            "disclaimer": DISCLAIMER,
        }

    # ------------------------------------------------------------------
    def predict(self, image: Image.Image, age_group: str | None = None, sex: str | None = None,
                body_regions: list[str] | None = None) -> dict:
        import torch

        started = time.perf_counter()
        image = ImageOps.exif_transpose(image).convert("RGB")
        vector = meta.encode(age_group, sex, body_regions or [])
        used_metadata = bool(age_group or sex or body_regions)

        skin_ratio = ood.skin_pixel_ratio(image)
        if skin_ratio < self.min_skin_ratio:
            return self._result("not_skin", None, None, None, used_metadata, None, started)

        probability_sets, weights, heatmap, logits_first = [], [], None, None
        meta_tensor = torch.as_tensor(vector, device=self.device)[None]
        for index, ((model, member), transform) in enumerate(zip(self.members, self.transforms, strict=False)):
            tensor = transform(image)
            if index == 0 and self.cam is not None:
                heatmap, _, _ = self.cam(tensor, meta_vector=vector)
            with torch.no_grad():
                batch = tensor[None].to(self.device)
                views = [batch, batch.flip(3)]                           # light test-time augmentation
                logits = torch.stack([model(v, meta_tensor) for v in views]).mean(0)[0].cpu().numpy()
            if index == 0:
                logits_first = logits
            probability_sets.append(ood.softmax(logits[None], member["temperature"])[0])
            weights.append(member.get("weight", 1.0))
        probabilities = np.average(np.stack(probability_sets), axis=0, weights=weights)

        status, label = ood.decide(probabilities, self.classes, self.threshold)
        energy = float(ood.energy_score(logits_first[None])[0])
        return self._result(status, label, probabilities, heatmap if status != "not_skin" else None,
                            used_metadata, energy, started, image)

    # ------------------------------------------------------------------
    def _result(self, status, label, probabilities, heatmap, used_metadata, energy, started, image=None):
        top3, confidence = [], None
        if probabilities is not None:
            shown = [(c, p) for c, p in zip(self.classes, probabilities, strict=False) if c != OTHER_CLASS]
            total = sum(p for _, p in shown) or 1.0
            shown.sort(key=lambda item: -item[1])
            top3 = [{"label": c, "probability": round(float(p / total), 4)} for c, p in shown[:3]]
            confidence = top3[0]["probability"]   # same number as the first bar on the page
        return {
            "status": status,
            "label": label,
            "display_label": DISPLAY_NAMES.get(label, label) if label else None,
            "confidence": confidence,
            "top3": top3,
            "message": MESSAGES[status].format(label=DISPLAY_NAMES.get(label, label)),
            "condition": CONDITIONS.get(label) if label else None,
            "heatmap_png_base64": self._heatmap_png(image, heatmap) if heatmap is not None else None,
            "used_metadata": used_metadata,
            "ood_score": round(energy, 3) if energy is not None else None,
            "processing_ms": round((time.perf_counter() - started) * 1000, 1),
            "disclaimer": DISCLAIMER,
        }

    @staticmethod
    def _heatmap_png(image: Image.Image, heatmap: np.ndarray) -> str:
        overlay = overlay_heatmap(center_square(image).resize((384, 384)), heatmap)   # same crop the model saw
        buffer = io.BytesIO()
        overlay.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("ascii")
