"""The model bundle the web app loads (v3: one or more models = an ensemble).

models/deploy/
    member_0.pt, member_1.pt ...  weights (state_dict) of each ensemble member
    model_card.json               members, classes, thresholds, metrics, date
"""

from __future__ import annotations

import time
from datetime import date
from pathlib import Path

from .utils import load_json, save_json

DISCLAIMER = ("DermaLens AI is a research prototype. It does not give a diagnosis. "
              "Please consult a qualified dermatologist about any skin concern.")


def save_bundle(members: list[dict], out_dir: Path, classes: list[str], metrics: dict,
                confidence_threshold: float, min_skin_ratio: float = 0.15,
                color_constancy: bool = True, notes: str = "") -> Path:
    """Write the ensemble bundle.

    members: [{"model": DermaNet, "name": "convnext_small", "image_size": 288,
               "temperature": 1.3, "weight": 1.0}, ...]
    """
    import torch

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    card_members = []
    for i, member in enumerate(members):
        file = f"member_{i}.pt"
        torch.save(member["model"].state_dict(), out_dir / file)
        card_members.append({"file": file, "model_name": member["name"], "image_size": member["image_size"],
                             "temperature": float(member.get("temperature", 1.0)),
                             "weight": float(member.get("weight", 1.0)),
                             "uses_metadata": bool(member["model"].use_metadata)})
    save_json({
        "format_version": 3,
        "members": card_members,
        "classes": classes,
        "confidence_threshold": float(confidence_threshold),
        "min_skin_ratio": float(min_skin_ratio),
        "color_constancy": color_constancy,
        "uses_metadata": any(m["uses_metadata"] for m in card_members),
        "test_metrics": metrics,
        "exported_on": date.today().isoformat(),
        "intended_use": "Decision support for five skin conditions + healthy skin on clinical photos. "
                        "Not a diagnosis.",
        "disclaimer": DISCLAIMER,
        "notes": notes,
    }, out_dir / "model_card.json")
    return out_dir


def load_bundle(bundle_dir: Path, device: str = "cpu"):
    """Rebuild every member. Returns (list of (model, member_card)), model_card)."""
    import torch

    from .model import build_model

    bundle_dir = Path(bundle_dir)
    card = load_json(bundle_dir / "model_card.json")
    loaded = []
    for member in card["members"]:
        model = build_model(member["model_name"], len(card["classes"]), pretrained=False,
                            use_metadata=member["uses_metadata"], image_size=member["image_size"])
        model.load_state_dict(torch.load(bundle_dir / member["file"], map_location=device, weights_only=True))
        loaded.append((model.to(device).eval(), member))
    return loaded, card


def measure_latency(model, image_size: int = 224, runs: int = 20, device: str = "cpu") -> dict:
    """Median and 95th-percentile time (ms) for one image, after 3 warm-up runs."""
    import numpy as np
    import torch

    dummy = torch.randn(1, 3, image_size, image_size, device=device)
    timings = []
    with torch.no_grad():
        for i in range(runs + 3):
            started = time.perf_counter()
            model(dummy)
            if i >= 3:
                timings.append((time.perf_counter() - started) * 1000)
    return {"median_ms": round(float(np.median(timings)), 1),
            "p95_ms": round(float(np.percentile(timings, 95)), 1)}
