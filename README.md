# DermaLens AI

Explainable skin-condition decision support for Indian and South Asian skin (v3):
**Contact Dermatitis, Eczema, Healthy, Psoriasis, Scabies, Tinea Corporis** - plus open-set rejection
("not identified" for other diseases, "not skin" for other photos), patient-metadata fusion
(age group, sex, body site), a model ensemble with Grad-CAM, and a multi-page web app.
B.Tech CSE (Semester VII) project, Meerut Institute of Technology. Supervisor: Mrs. Srishti Agarwal.

> Research prototype for decision support. It does not give a diagnosis.

## Run it locally

See **[docs/LOCAL_SETUP.md](docs/LOCAL_SETUP.md)** — clone, `pip install`, download the model bundle from Drive, `uvicorn app.api:app`. No datasets needed.

## Repository layout

```
DermaLens_AI/                  (this folder lives on Google Drive and is also the Git repo)
├── configs/
│   ├── baseline.yaml          every setting used for real runs
│   └── smoke.yaml             tiny settings for a 5-minute pipeline check
├── src/dermalens/             shared code (see the table below)
├── notebooks/                 00 → 14, run in order
├── app/                       FastAPI API + 3-page website (home, analyse, model & research) + Dockerfile
├── tests/                     pytest suite (runs on synthetic data, no real images needed)
├── scripts/
│   ├── build_notebooks.py     regenerates notebooks/ from scripts/nbsrc/
│   └── make_synthetic_project.py   fake data/raw for tests
├── docs/                      WEB_APP_GUIDE, DATA_CARD, MODEL_CARD, CHANGES
├── data/                      raw / processed / splits / cache   (not in Git)
├── models/                    checkpoints + deploy bundle         (not in Git)
└── reports/                   audit CSVs, figures, results
```

| Module | What it does |
|---|---|
| `paths.py` | Every folder location in one place |
| `config.py` | Reads `configs/*.yaml` |
| `labels.py` | The six classes, the "Other" class, grouped taxonomy, source-label mapping |
| `metadata.py` | Age / sex / body-region mapping and the 21-number metadata vector |
| `ood.py` | Open-set decisions: skin check, "Other", confidence threshold, energy score |
| `audit.py` | Read-only inventory of `data/raw` |
| `manifest.py` | Builds the image table from raw metadata; SHA-256 and perceptual-hash de-duplication |
| `splits.py` | Patient-grouped, class-balanced train/val/test; class weights |
| `data.py` | Resized-image cache, transforms, PyTorch Dataset/DataLoader |
| `model.py` | DermaNet: timm backbone (CNN, ViT, Swin) + metadata branch |
| `train.py` | Layer-wise LR decay, warmup-cosine, mixup, EMA, early stopping, resume |
| `evaluate.py` | Metrics, TTA, ensembles, bootstrap CIs, calibration, grouped taxonomy |
| `explain.py` | Grad-CAM for CNNs, ViT and Swin |
| `export.py` | Ensemble bundle for the web app |

## Notebooks

| # | Notebook | Output |
|---|---|---|
| 00 | Setup & environment check | confirms GPU, libraries, raw data |
| 01 | Data audit | inventory, metadata overview, SkinDisNet mirror check |
| 02 | Label taxonomy | why these six classes; mapping table |
| 03 | Build clean manifest | `data/processed/manifest_clean.csv` + cleaning log |
| 04 | EDA & data quality | figures + `eda_summary.json` |
| 05 | Splits & class weights | `data/splits/*` (frozen) |
| 06 | Dataset & transforms | `data/cache/`, augmentation preview |
| 07 | Training recipe | one strong model (ConvNeXt-S + metadata), learning curves |
| 08 | Model comparison & metadata ablation | 5 architectures × 2 seeds; image-only vs image+metadata; `best_models.json` |
| 09 | Ensemble & test evaluation | TTA ensemble, calibration, 95% CIs, per class / source / skin type, grouped view |
| 10 | Open-set evaluation | confidence threshold, open-set AUROC, accuracy-coverage curve, skin check |
| 11 | Grad-CAM | heatmaps + human review sheet |
| 12 | Source-shortcut test | dataset-style shortcut, with vs without colour constancy |
| 13 | Grouped taxonomy | retrain with Contact Dermatitis + Eczema merged |
| 14 | Export for web app | `models/deploy/` (ensemble bundle) |

## Quick start (Colab)

1. Put this repository in `MyDrive/DermaLens_AI/` (next to the existing `data/` folder).
2. Open `notebooks/00_setup_and_environment_check.ipynb` in Colab, choose **Runtime → T4 GPU**, run all.
3. Run notebooks 01 → 14 in order. Each notebook's first cell says what it reads and writes.

## Check the pipeline without real data

```bash
pip install -r requirements.txt -r app/requirements.txt pytest httpx
pytest -q                                            # 18 tests, ~20 s on a laptop CPU
python scripts/make_synthetic_project.py /tmp/fake   # then run the notebooks with:
export DERMALENS_PROJECT_DIR=/tmp/fake DERMALENS_CODE_DIR=$PWD DERMALENS_CONFIG=$PWD/configs/smoke.yaml
```

## Web app

See [docs/WEB_APP_GUIDE.md](docs/WEB_APP_GUIDE.md). Short version:

```bash
uvicorn app.api:app --reload --port 8000      # needs models/deploy/ from notebook 14
```

## Team workflow

* One branch per task; open a pull request into `main`; CI runs `ruff` and `pytest`.
* Edit notebooks through `scripts/nbsrc/*.py`, then `python scripts/build_notebooks.py`, so all notebooks keep the same structure.
* Never edit `data/splits/` after notebook 05 without a team decision.

## Data sources and licences

DermaCon-IN (NeurIPS 2025, data-use terms of the authors) · SkinDisNet (Mendeley Data, doi:10.17632/yj3md44hxg.2) ·
MCSI normal-skin images (CC BY 4.0) · SCIN (reference only). See [docs/DATA_CARD.md](docs/DATA_CARD.md).
