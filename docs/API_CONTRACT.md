# DermaLens AI v3 — API contract (frontend ↔ backend)

All endpoints are served by FastAPI from the same origin as the static site.

## GET /api/info

```json
{
  "app_version": "3.0.0",
  "model": {"name": "ensemble: convnext_small + vit_base_dinov2", "exported_on": "2026-11-20",
            "uses_metadata": true, "image_size": 288},
  "classes": ["Contact Dermatitis", "Eczema", "Healthy", "Psoriasis", "Scabies", "Tinea Corporis"],
  "metrics": {"accuracy": 0.81, "macro_f1": 0.80, "balanced_accuracy": 0.80,
              "open_set_auroc": 0.88, "test_images": 343},
  "metadata_options": {
    "age_groups": [{"key": "0-10", "label": "0–10 years"}, {"key": "10-20", "label": "10–20 years"},
                   {"key": "20-40", "label": "20–40 years"}, {"key": "40-60", "label": "40–60 years"},
                   {"key": "60+", "label": "60 years and above"}],
    "sexes": [{"key": "female", "label": "Female"}, {"key": "male", "label": "Male"}],
    "body_regions": [{"key": "face", "label": "Face"}, {"key": "scalp", "label": "Scalp"},
                     {"key": "neck", "label": "Neck"}, {"key": "chest_abdomen", "label": "Chest / abdomen"},
                     {"key": "back", "label": "Back"}, {"key": "buttocks", "label": "Buttocks"},
                     {"key": "groin_genital", "label": "Groin / genital area"}, {"key": "arm", "label": "Arm / elbow"},
                     {"key": "hand", "label": "Hand / fingers / palm"}, {"key": "leg", "label": "Thigh / knee / leg"},
                     {"key": "foot", "label": "Foot / toes / sole"}]
  },
  "conditions": {
    "Tinea Corporis": {"summary": "...", "typical_sites": "...", "next_steps": "..."}
  },
  "disclaimer": "DermaLens AI is a research prototype..."
}
```

Numbers above are examples; the real values come from the model card.

## POST /api/predict  (multipart/form-data)

| field | required | values |
|---|---|---|
| `file` | yes | JPEG / PNG / WebP, ≤ 10 MB, ≥ 64 px |
| `age_group` | no | one of the `age_groups` keys |
| `sex` | no | `female` / `male` |
| `body_regions` | no | comma-separated `body_regions` keys, e.g. `hand,arm` |

Response 200:

```json
{
  "status": "identified",
  "label": "Tinea Corporis",
  "display_label": "Tinea Corporis (ringworm)",
  "confidence": 0.87,
  "top3": [{"label": "Tinea Corporis", "probability": 0.87},
           {"label": "Eczema", "probability": 0.07},
           {"label": "Contact Dermatitis", "probability": 0.03}],
  "message": "The photo most closely matches Tinea Corporis.",
  "condition": {"summary": "...", "typical_sites": "...", "next_steps": "..."},
  "heatmap_png_base64": "iVBORw0...",
  "used_metadata": true,
  "ood_score": 0.12,
  "processing_ms": 412.0,
  "disclaimer": "..."
}
```

`status` decides the result card:

| status | meaning | what the page shows |
|---|---|---|
| `identified` | one of the 5 conditions, confident | condition name, confidence, top-3 bars, heatmap, condition info |
| `healthy` | healthy skin | "No skin condition detected" (green), confidence, heatmap optional |
| `not_identified` | a skin photo, but not confidently one of the 6 classes (another disease, or too uncertain) | "Not identified — this may be a condition DermaLens does not cover. Please consult a dermatologist." top3 still sent but shown greyed as "closest matches" |
| `not_skin` | the photo does not look like skin | "This does not look like a photo of skin. Please upload a close, well-lit photo of the affected area." No top-3, no heatmap |

Errors: 400 / 413 / 415 with `{"detail": "human readable message"}`.

## GET /health → `{"status": "ok", "model_loaded": true}`
