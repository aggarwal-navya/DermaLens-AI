# Data Card: DermaLens AI training data

Numbers marked † are filled in from `reports/dataset_audit/cleaning_log.csv` and `data/splits/class_info.json`
after notebooks 03 and 05 run on the real data. Values below are from the v1 audit (October 2026).

## Sources

| Source | Origin | Used for | Licence / terms | Photos used |
|---|---|---|---|---|
| DermaCon-IN | South India outpatient clinics, dermatologist-labelled (NeurIPS 2025) | Tinea Corporis, Scabies, Eczema, Contact Dermatitis, Psoriasis | authors' data-use terms | ~962 † |
| SkinDisNet | Two Bangladeshi medical colleges, smartphone photos | Tinea Corporis, Scabies, Eczema, Contact Dermatitis | Mendeley Data (doi:10.17632/yj3md44hxg.2) | 1,561 of 1,710 originals † |
| MCSI | web-collected skin images | Healthy (`diagnostic == "normal"`) | CC BY 4.0 | 100 |
| SCIN | US crowdsourced (Google) | not used for training; reference only | SCIN terms | 0 |

## Cleaning steps (notebook 03)

1. Only originals: SkinDisNet `skindisnet2/Preprocessed` (the 23,523 augmented files and the `skindisnet1` mirror are excluded).
2. DermaCon-IN rows marked "quality insufficient" by dermatologists are removed.
3. Exact duplicates (same SHA-256) removed; identical files with different labels stop the pipeline.
4. Every image decoded once; unreadable files removed.
5. Near duplicates (perceptual hash ≤ 6 bits apart) are put in the same patient group.

## Known limitations and biases

* **Single-source classes:** Healthy = MCSI only; Psoriasis = DermaCon-IN only. A model can partly recognise the
  source instead of the disease. Measured in notebooks 04 and 11; results reported per source in notebook 09.
* **Skin-tone labels:** only DermaCon-IN has Fitzpatrick/Monk labels (FST 3-6; FST 6 is rare). Fairness results
  cover that subset only and carry wide confidence intervals.
* **Geography:** DermaCon-IN is South Indian, SkinDisNet Bangladeshi; North Indian populations are not directly sampled.
* **Healthy class** is small (100) and web-collected, so its photos differ in style from clinical photos.
* **SCIN metadata** on Drive was damaged by a spreadsheet tool (case IDs in scientific notation); re-download before any use.

## Splits (notebook 05)

Stratified by class, grouped by patient (no patient in two splits), seed chosen from 50 candidates to minimise class-mix
drift; ~72% train, ~14% validation, ~14% test. Frozen after creation.
