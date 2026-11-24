# Version 3 changes (19 Nov 2026)

Goal of v3: higher accuracy, "not identified" instead of random guesses, metadata input, a professional
website, and enough research depth for a major project and a paper.

| # | Area | Change | Problem it fixes (v2) | Where |
|---|---|---|---|---|
| 1 | Bug | Training reads the original images when the resized cache is missing | `KeyError: 'cache_path'` in notebook 07 cell 4 when 06 was skipped | `data.resolve_images` |
| 2 | Accuracy | Stronger backbones: ConvNeXt-S (IN-22k), EfficientNetV2-S (IN-21k), Swin-T, ViT-B DINOv2; 288 px | small IN-1k CNNs at 224 px | `model.py` |
| 3 | Accuracy | Layer-wise LR decay, warmup + cosine, weight decay 0.05, drop-path | backbone barely adapted (one low LR) | `train.py` |
| 4 | Accuracy | Mixup, stronger augmentation, EMA weights | over-fitting, noisy final weights | `train.py`, `data.py` |
| 5 | Accuracy | Test-time augmentation + calibrated soft-voting ensemble of the top-k models | single model | `evaluate.py`, nb 09 |
| 6 | Accuracy | Metadata fusion: age group, sex, 11 body regions (21-number vector, field dropout) | photo only | `metadata.py`, `model.py` |
| 7 | Shortcut | Shades-of-Gray colour constancy on every image; shortcut test with vs without it | camera colour reveals the dataset | `data.py`, nb 12 |
| 8 | Open set | "Other" class trained on non-target diseases (outlier exposure) | other diseases got a confident wrong label | `manifest.py`, `labels.py` |
| 9 | Open set | Confidence threshold tuned on validation; skin-pixel check; statuses identified / healthy / not_identified / not_skin | "healthy" and random answers for non-skin or unknown photos | `ood.py`, nb 10 |
| 10 | Research | Metadata ablation, open-set AUROC, accuracy-coverage curve, grouped taxonomy retrain, Grad-CAM for ViT/Swin | too little analysis for a major project | nb 08, 10, 11, 13 |
| 11 | Product | 3-page website: home (intro, features, conditions, metrics), analyse (photo + metadata form, rich result card, heatmap, print report, history), model & research | one plain upload page | `app/static/` |
| 12 | Product | API v3: metadata form fields, statuses, condition info, `/api/info` drives the site; ensemble bundle | single model, no metadata | `app/`, `export.py`, `docs/API_CONTRACT.md` |

Verified: 22 automated tests; all 15 notebooks executed end to end on the synthetic project (with notebook 06
skipped first, to prove fix 1); the website was driven in a real browser against the real API.

# Version 2 changes (29 Oct 2026)

Twelve changes, from basic housekeeping to the high-level additions. Each row names the problem in v1 and where the fix lives.

| # | Level | Change | Problem it fixes (v1) | Where |
|---|---|---|---|---|
| 1 | Basic | Shared code library `src/dermalens/` | Logic copied between notebooks, one-letter names | `src/dermalens/*` |
| 2 | Basic | One config file + one paths module | Paths and numbers typed by hand in many cells | `configs/baseline.yaml`, `paths.py` |
| 3 | Basic | Notebooks renumbered 00-12 with a standard header (purpose, reads, writes, run time) and the same setup cell | 03 used a different class set than 04; stale outputs from out-of-order runs | `notebooks/`, `scripts/nbsrc/` |
| 4 | Basic | v1 split files archived, one splits folder | Two sets of `train/val/test.csv` on Drive | notebook 00 §3 |
| 5 | Medium | Manifests built from raw metadata in code; SkinDisNet joined on `Image_id` | Two input CSVs no notebook created; images matched by row order | `manifest.py`, notebook 03 |
| 6 | Medium | Poor-quality DermaCon-IN rows removed; phone rotation (EXIF) applied when loading | 2 "quality insufficient" photos kept; portrait photos could load sideways | `manifest.py`, `data.py` |
| 7 | Medium | Real SHA-256 de-duplication of the final images (cached), stops on conflicting labels | Hashing was switched off; "hash-based" claim untrue; 1 duplicate found unresolved | `manifest.py`, notebook 03 |
| 8 | High | Every image decoded once + perceptual-hash near-duplicates forced into the same patient group | Resized copies of one photo could land in train and test | `manifest.py`, notebook 03 |
| 9 | Medium | Full EDA incl. image size/colour per source and a quick "guess the source" test | No measure of the source shortcut risk | notebook 04 |
| 10 | High | Patient-grouped split with seed search for best class balance, leakage asserts, frozen `class_info.json` | Psoriasis 36 val vs 22 test; Scabies 19% train vs 28% test | `splits.py`, notebook 05 |
| 11 | High | Training system: resized-image cache, two-stage fine-tuning, weighted loss + label smoothing, mixed precision, early stopping on macro-F1, resume after Colab disconnect, 4 models × 3 seeds | No training code yet | `data.py`, `train.py`, notebooks 06-08 |
| 12 | High | Honest evaluation and delivery: grouped bootstrap 95% CIs, per-source and per-skin-type results, temperature calibration, Grad-CAM review sheet, CNN source-shortcut test, model bundle, FastAPI app, 18 tests, CI | Synopsis targets had no way to be measured; no app | `evaluate.py`, `explain.py`, `export.py`, `app/`, `tests/`, notebooks 09-12 |

Also documented, not code: SCIN case IDs on Drive were damaged by a spreadsheet tool (re-download), and the
synopsis corrections listed in the review doc.
