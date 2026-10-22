# Skin Disease Classification for Indian Skin — Project Research & Plan

**Target:** 6-class classification (5 diseases + 1 healthy) · Progress report due Oct 24, 2026

---

## 0. Quick Links

**Datasets**
- DermaCon-IN paper (arXiv): https://arxiv.org/abs/2506.06099
- DermaCon-IN paper (PDF): https://arxiv.org/pdf/2506.06099
- DermaCon-IN (NeurIPS proceedings page): https://proceedings.nips.cc/paper_files/paper/2025/hash/a2c548b2ed3141400d86a235eaa6fad0-Abstract-Datasets_and_Benchmarks_Track.html
- DermaCon-IN (OpenReview, full PDF w/ reviews): https://openreview.net/pdf/7d53d989204cfbb8445f4fcf8c1d3060e5954643.pdf
- DermaCon-IN (NeurIPS poster/slides page): https://neurips.cc/virtual/2025/poster/121561
- SCIN dataset (GitHub, download + code): https://github.com/google-research-datasets/scin
- SCIN methodology paper (arXiv): https://arxiv.org/html/2402.18545v1
- SkinDisNet (Mendeley Data, download): https://data.mendeley.com/datasets/yj3md44hxg/2
- Combined DermaConIN + SkinDisNet + Arsenic Skin Images (Hugging Face): https://huggingface.co/datasets/actorcritic/DermaConIN_SkinDisNet_ArsenicSkinImageBD
- Alternate 5-category Skin Disease Classification Dataset (Mendeley): https://data.mendeley.com/datasets/3hckgznc67/1

**Tools / libraries**
- `timm` (pretrained models incl. ResNet/EfficientNet/ViT/Swin/MaxViT): https://github.com/huggingface/pytorch-image-models
- `albumentations` (augmentation): https://albumentations.ai/
- `pytorch-grad-cam`: https://github.com/jacobgil/pytorch-grad-cam
- `captum` (explainability): https://captum.ai/
- Weights & Biases: https://wandb.ai/

---

## 1. Dataset Evaluation — Are DermaCon-IN, SCIN, SkinDisNet suitable?

### Verdict up front
| Dataset | Suitable as PRIMARY dataset? | Why |
|---|---|---|
| **DermaCon-IN** | ✅ **Yes — best fit** | Only one actually collected from Indian patients. Real clinical relevance. |
| **SCIN** | ⚠️ Partial / supplementary only | US-crowdsourced, not Indian. Useful for skin-tone diversity augmentation, not as your core "Indian skin" dataset. |
| **SkinDisNet** | ⚠️ Usable but not Indian | Collected in Bangladesh (South Asian skin, same Fitzpatrick range as North India broadly, but not "Indian" per se) — closest substitute if DermaCon-IN access/classes don't fit. |

### Detailed findings

**DermaCon-IN** (NeurIPS 2025 Datasets & Benchmarks Track, IIT Hyderabad + partner hospitals)
- 5,450 clinical images from ~3,000 patients, collected prospectively at outpatient clinics in **South India**.
- Dermatologist-annotated with **245 distinct diagnoses**, structured in a hierarchical taxonomy (main class → sub class → disease) adapted from Rook's classification.
- Includes Fitzpatrick Skin Type (FST) and Monk Skin Tone (MST) labels — skews toward **FST III–VI** (medium-to-dark skin), which is exactly the Indian skin representation gap you're trying to address.
- Benchmarked by the authors themselves on ResNet, DenseNet, EfficientNet, ViT, MaxViT, Swin, and Concept Bottleneck Models.
- Links: [arXiv abstract](https://arxiv.org/abs/2506.06099) · [PDF](https://arxiv.org/pdf/2506.06099) · [NeurIPS page](https://proceedings.nips.cc/paper_files/paper/2025/hash/a2c548b2ed3141400d86a235eaa6fad0-Abstract-Datasets_and_Benchmarks_Track.html) · [OpenReview (full paper + reviews)](https://openreview.net/pdf/7d53d989204cfbb8445f4fcf8c1d3060e5954643.pdf)
- **Caveat for you:** it has 245 fine-grained diagnoses, heavily long-tailed (a handful of diseases dominate, most have very few images). You will need to **pick/merge 5 high-frequency disease classes + define a "healthy/normal skin" class** (healthy skin may not be natively present — check carefully, since it's a clinical outpatient dataset, meaning almost all images are of a condition). This is your most important action item (see Section 4).
- Access: hosted via the paper's companion site/GitHub (check the NeurIPS/arXiv page for current hosting — may require filling a request form since it's a clinical dataset with ethics approvals). **Action item: confirm exact download link and any data-use agreement before your report.**

**SCIN (Skin Condition Image Network)** — Google Research + Stanford Medicine, 2024
- 10,000+ images, 5,000+ volunteer contributions, crowdsourced via **Google Search ads from US internet users** — NOT Indian.
- Self-reported Fitzpatrick Skin Type (sFST) and Monk Skin Tone (eMST/eFST) labels — does include darker skin tones, which is its main value to you.
- Captures common short-duration conditions (rashes, allergies, infections) rather than dermoscopic/clinical-grade images.
- Open access, hosted on GitHub: https://github.com/google-research-datasets/scin
- Methodology paper: https://arxiv.org/html/2402.18545v1
- **Use case for your project:** supplementary data to balance skin-tone representation or to pretrain/augment, but should not be your "Indian skin" justification — a reviewer will ask why a US crowdsourced dataset represents Indian skin, and you won't have a strong answer.

**SkinDisNet** — Mendeley Data, 2025, Begum Rokeya University (Bangladesh)
- 1,710 clinical images (416 patients) across **exactly 6 classes**: Atopic Dermatitis, Contact Dermatitis, Eczema, Scabies, Seborrheic Dermatitis, Tinea Corporis — plus an augmented folder of 11,970 images.
- Collected via smartphone cameras at two Bangladeshi medical colleges (Rangpur Medical College, Shahid Syed Nazrul Islam Medical College).
- **Important:** this gives you 6 disease classes but **no "healthy skin" class** — you'd need to source healthy images separately (see Section 4).
- Geographically it's Bangladeshi, not Indian, but South Asian skin tone/phototype distribution is broadly comparable to North/East Indian populations — defensible as a secondary/comparison dataset, weaker as your sole justification for "Indian skin."
- Freely downloadable from Mendeley Data: https://data.mendeley.com/datasets/yj3md44hxg/2 — no access request needed, easiest to start working with immediately.
- There's already a combined Hugging Face dataset merging **DermaConIN + SkinDisNet + an Arsenic Skin Image dataset from Bangladesh**, including a healthy-skin class: https://huggingface.co/datasets/actorcritic/DermaConIN_SkinDisNet_ArsenicSkinImageBD — worth inspecting directly since someone has already done exactly this kind of class harmonization; you can review their class-merging logic even if you don't use their exact pipeline for your own curated version.

### Recommendation for your progress report (due 24th)
1. State DermaCon-IN as primary dataset (Indian, clinically sourced, largest and most rigorous).
2. State SkinDisNet as the easiest-to-access secondary dataset to begin coding/pipeline work on immediately (no access request delay) while DermaCon-IN access/class selection is finalized.
3. State SCIN as a tone-diversity augmentation source, not a core dataset.
4. Flag the "healthy skin" class as an open data-sourcing task (see Section 4) — this is a real gap across all three and showing you've identified it looks good in a progress review.

---

## 2. Related Work / Research Papers to Study

### Core papers (read these first, in order)
1. **DermaCon-IN: A Multi-concept Annotated Dermatological Image Dataset of Indian Skin Disorders for Clinical AI Research** — Madarkar et al., NeurIPS 2025 Datasets & Benchmarks Track. (arXiv: 2506.06099) — your primary dataset paper; also gives you baseline accuracy numbers on ResNet/DenseNet/EfficientNet/ViT/MaxViT/Swin that you can cite as the benchmark to beat or match.
2. **SCIN: Crowdsourcing Dermatology Images with Google Search Ads** (arXiv: 2402.18545) — methodology for building a diverse, skin-tone-labeled dataset; useful if you supplement with SCIN or build your own data-collection pipeline later.
3. **SkinDisNet dataset description** (Mendeley, DOI: 10.17632/yj3md44hxg.2) — dataset card; read for exact class definitions and metadata schema (7 attributes per image).
4. **Fitzpatrick17k dataset paper** — the most-cited prior dataset DermaCon-IN positions itself against; important for your literature review to explain *why* Indian-specific data matters (reported 30–40% accuracy drop on darker skin tones when models trained on Fitzpatrick17k are evaluated on darker-skin test sets like DDI).
5. **DDI (Diverse Dermatology Images) dataset paper** — the paper that first quantified the accuracy drop on darker skin tones; cited by DermaCon-IN as core motivation.
6. **HAM10000** and **ISIC 2019** papers — not Indian-specific, but the standard benchmark dermoscopic datasets; read to understand the *difference* between dermoscopic (zoomed, dermatoscope-captured) vs. clinical (regular camera, in-context) image datasets, since DermaCon-IN and SkinDisNet are both clinical-image datasets, not dermoscopic — this affects your choice of architecture and augmentation strategy.

### Model/methodology papers
7. A **ResNet** paper (He et al., "Deep Residual Learning for Image Recognition") — baseline CNN architecture.
8. An **EfficientNet** paper (Tan & Le) — efficient scaling, strong baseline for medical imaging with limited data.
9. A **Vision Transformer (ViT)** paper (Dosovitskiy et al.) — since DermaCon-IN's authors benchmark ViT/MaxViT/Swin, study at least one transformer-based image classifier.
10. A paper on **Concept Bottleneck Models (CBMs)** — DermaCon-IN's authors specifically built annotations for this; if your team wants an "explainable AI" angle (very attractive for a progress evaluation and for a final report), this is a differentiator over generic CNN classification.
11. A survey paper on **class imbalance handling in medical image classification** (focal loss, class-weighted loss, oversampling/SMOTE for images, data augmentation) — you will have an imbalanced dataset regardless of which source you pick, so this is essential early reading, not optional.
12. A paper on **transfer learning for skin disease classification on small clinical datasets** — directly relevant since your usable Indian-labeled data will likely be in the low thousands of images, not hundreds of thousands.

---

## 3. Tech Stack

| Layer | Recommendation | Notes |
|---|---|---|
| Language | Python 3.10+ | Standard for ML |
| DL framework | **PyTorch** (+ `torchvision`) | More control for custom architectures, CBMs, and easier to follow the DermaCon-IN benchmark code if released |
| Pretrained models | `timm` library | Gives you ResNet, DenseNet, EfficientNet, ViT, Swin, MaxViT with ImageNet weights in one line |
| Data handling | `pandas`, `pillow`/`opencv-python`, `albumentations` | `albumentations` for medical-image-safe augmentation (rotation, color jitter, CLAHE for contrast) |
| Experiment tracking | **Weights & Biases (wandb)** or MLflow | Essential for a multi-week team project — lets you show loss/accuracy curves in your progress report |
| Explainability | `pytorch-grad-cam` or `captum` | Grad-CAM visualizations are extremely strong evidence to show in a progress/evaluation report — shows the model is "looking at" the lesion, not the background |
| Environment | Google Colab Pro / Kaggle Notebooks (free GPU) initially → move to a shared GPU (college lab / AWS/GCP credits) once dataset is finalized | |
| Versioning | Git + GitHub, DVC (optional) for dataset versioning | |
| Class imbalance | `imbalanced-learn`-style weighted sampling, or manually compute `class_weight` for `CrossEntropyLoss`, or **Focal Loss** | Non-negotiable given all three datasets are imbalanced |

---

## 4. Immediate Action Items Before the 24th

To "show some progress" in two days, focus on things that are visible and don't require the final dataset decision to be locked:

1. **Download SkinDisNet now** (open access, no request form) — unzip, inspect folder structure, write a basic `Dataset`/`DataLoader` class in PyTorch. This alone is a demonstrable artifact for your report ("data pipeline built and tested on one of the three candidate datasets").
2. **Request/locate DermaCon-IN access** today — email the authors or check the arXiv/NeurIPS page for a hosting link; access requests for clinical datasets often take days, so starting this now is the single highest-leverage action for your timeline.
3. **Run a baseline**: fine-tune a pretrained ResNet-18 or EfficientNet-B0 (via `timm`) on SkinDisNet's 6 classes (even without a healthy class yet) for a few epochs. A training curve + confusion matrix, however rough, is a strong visual for a progress report.
4. **Solve the "healthy skin" gap**: none of the three datasets cleanly gives you a labeled "healthy/normal skin" class (DermaCon-IN and SkinDisNet are both condition-only clinical collections). Options to raise in your report as "identified and being addressed": (a) use normal-skin patches from the **Fitzpatrick17k** or **DDI** datasets, (b) use a public "healthy skin" Kaggle dataset, (c) extract clean-skin crops from the clinical images' surrounding (non-lesion) regions as a fourth-image-like negative class. Flagging this gap explicitly in your evaluation report demonstrates rigor.
5. **Draft your literature review section** citing at minimum DermaCon-IN, SCIN, Fitzpatrick17k, and the DDI paper — this directly shows "related work has been studied" without needing GPU time.
6. **One-slide class-mapping table**: pick the 5 target diseases now (e.g. candidates appearing across datasets: Eczema/Atopic Dermatitis, Tinea/Fungal infection, Scabies, Acne, Psoriasis, Contact Dermatitis — cross-reference against what's actually well-represented in DermaCon-IN's top classes) + healthy. Showing this decision table in the progress report demonstrates you've evaluated dataset suitability, which is literally the task you were set.

---

## 4a. Getting Started Today — Colab Setup, Project Structure, Notebook Order

You're starting from zero, running on Google Colab. Here's the concrete first-two-days sequence.

### Recommended project structure

Keep this in a shared GitHub repo from day one (even an empty-ish one) — Colab notebooks mount Drive/clone this repo, so the structure outlives any one person's Colab session.

```
skin-disease-classification/
├── README.md
├── data/                      # NOT committed to git — add to .gitignore
│   ├── raw/                   # untouched downloaded datasets
│   │   └── skindisnet/
│   ├── processed/             # resized/cleaned images, train/val/test splits
│   └── class_mapping.csv      # your 5-disease + healthy label mapping (Section 4, item 6)
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_preprocessing_and_split.ipynb
│   ├── 03_baseline_training.ipynb
│   ├── 04_gradcam_explainability.ipynb
│   └── 05_evaluation_report.ipynb
├── src/                       # reusable code, imported into notebooks
│   ├── dataset.py             # PyTorch Dataset/DataLoader classes
│   ├── transforms.py          # albumentations pipelines
│   ├── model.py                # timm model wrapper, class-weighted loss setup
│   ├── train.py                # training loop function
│   └── utils.py
├── outputs/
│   ├── checkpoints/            # saved model weights — NOT committed (large files)
│   ├── figures/                 # confusion matrices, training curves, Grad-CAM images
│   └── logs/
├── requirements.txt
└── .gitignore                  # data/, outputs/checkpoints/, *.pth, .ipynb_checkpoints/
```

Why this shape: separating `notebooks/` (exploration + narrative, what you show in the report) from `src/` (reusable functions) means your training code isn't duplicated and pasted across five notebooks — each notebook just does `from src.dataset import SkinDataset` etc. For a two-day sprint you can skip `src/` and keep everything inline in the notebooks, then refactor into `src/` once the pipeline stabilizes — don't over-engineer before the first baseline runs.

### Day 1 — Notebook 01: Data Exploration

Goal: download SkinDisNet, understand it, produce your first visual artifact.

1. In Colab: `Runtime → Change runtime type → T4 GPU` (free tier is enough for this).
2. Mount Google Drive so downloaded data and checkpoints persist across sessions (Colab's local disk is wiped when the runtime recycles):
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```
3. Download SkinDisNet directly from Mendeley into Drive:
   ```python
   !wget -O /content/drive/MyDrive/skin-disease-classification/data/raw/skindisnet.zip \
       "https://data.mendeley.com/public-files/datasets/yj3md44hxg/files"
   # If the direct wget link doesn't resolve (Mendeley sometimes requires a browser click),
   # download manually from https://data.mendeley.com/datasets/yj3md44hxg/2 and upload the zip to Drive.
   !unzip -q /content/drive/MyDrive/skin-disease-classification/data/raw/skindisnet.zip -d /content/drive/MyDrive/skin-disease-classification/data/raw/skindisnet/
   ```
4. Inspect: count images per class, check image sizes, plot a grid of sample images per class (`matplotlib`), read the metadata CSV (7 attributes per image, per the dataset card).
5. **Deliverable for today:** a per-class image count bar chart + a sample image grid. This alone is screenshot-able for your progress report.

### Day 1–2 — Notebook 02: Preprocessing & Split

1. Resize all images to a consistent size (224×224 or 256×256 — matches most pretrained `timm` model inputs).
2. Stratified train/val/test split (70/15/15 or 80/10/10) — use `sklearn.model_selection.train_test_split` with `stratify=labels` so each class is proportionally represented in every split.
3. Save split manifests as CSVs (`train.csv`, `val.csv`, `test.csv` with image path + label columns) rather than physically copying files into folders — faster and keeps `data/raw/` untouched.
4. Write the `SkinDataset(torch.utils.data.Dataset)` class now — this is the piece every later notebook reuses.

### Day 2 — Notebook 03: Baseline Training

1. `pip install timm albumentations` in the first cell.
2. Load a pretrained model:
   ```python
   import timm
   model = timm.create_model('resnet18', pretrained=True, num_classes=6)  # or efficientnet_b0
   ```
3. Compute class weights from your training split and pass to `nn.CrossEntropyLoss(weight=...)` — non-negotiable given the imbalance noted in Section 3.
4. Train for a small number of epochs (5–10 is enough to get a believable, honest curve for a progress report — you're not trying to hit final accuracy yet).
5. Plot train/val loss and accuracy curves, and a confusion matrix on the val set.
6. **Deliverable for the 24th:** training curve + confusion matrix + a one-paragraph "what we observed" note (e.g. which classes are confused, whether imbalance is visibly hurting minority classes). This, combined with Notebook 01's exploration plots and the dataset comparison table from Section 1, is a credible two-day progress report.

### Practical Colab notes
- Free-tier Colab disconnects after inactivity and has session time limits — save checkpoints to Drive every epoch or two (`torch.save(model.state_dict(), '/content/drive/MyDrive/.../checkpoint.pth')`) so a disconnect doesn't lose progress.
- Keep `requirements.txt` updated as you `pip install` things in Colab, so a teammate can reproduce your environment: `!pip freeze > /content/drive/MyDrive/skin-disease-classification/requirements.txt`.
- Once DermaCon-IN access comes through, Notebook 01/02 patterns repeat for it — you're not rebuilding the pipeline, just pointing `SkinDataset` at a new root folder and a new label-mapping CSV.

---

## 5. Suggested High-Level Pipeline (for the full project, post-24th)

1. Finalize dataset(s) and the 5-disease + healthy class mapping.
2. Build unified preprocessing (resize, normalize, FST/MST-aware stratified train/val/test split so skin-tone distribution isn't biased across splits).
3. Baseline: transfer-learned EfficientNet-B0/ResNet-50 with class-weighted loss.
4. Compare against a ViT/Swin baseline (matching DermaCon-IN's own benchmark setup, so your numbers are directly comparable to a published paper — good for your final report).
5. Add Grad-CAM explainability layer.
6. (Stretch goal, aligns with DermaCon-IN's own contribution) Explore a Concept Bottleneck Model for interpretability — strong differentiator for a final-year major project.
7. Evaluate with per-class precision/recall/F1 (not just accuracy, given imbalance) + skin-tone-stratified evaluation (report accuracy separately by FST group — this is the "Indian skin" fairness story your whole project is about).

---

*Sources consulted: DermaCon-IN paper (arXiv 2506.06099, NeurIPS 2025), SCIN dataset paper (arXiv 2402.18545) and Google/Stanford announcements, SkinDisNet dataset card (Mendeley DOI 10.17632/yj3md44hxg.2), and a Hugging Face dataset combining DermaCon-IN + SkinDisNet + Arsenic Skin Images.*
