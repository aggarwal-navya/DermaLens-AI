# DermaLens AI: Web App Guide

From a trained model to a public demo link, step by step.

## 1. How the pieces fit

```
Browser (index.html)
   │  POST /api/predict  (photo, multipart form)
   ▼
FastAPI (app/api.py)          checks type (JPEG/PNG/WebP), size (≤10 MB), min 64 px
   │
   ▼
Predictor (app/inference.py)  loaded once at start-up
   │  1. fix phone rotation (EXIF), convert to RGB
   │  2. resize 256 → centre-crop 224 → normalise   (same as validation in training)
   │  3. model forward + Grad-CAM backward          (one pass)
   │  4. temperature-scaled probabilities           (T from notebook 09)
   ▼
JSON: label, confidence, uncertain flag, top-3, heatmap (base64 PNG), disclaimer, processing time
```

The model bundle `models/deploy/` (written by notebook 14) holds the ensemble:
`member_0.pt`, `member_1.pt`, ... (weights of each model) and `model_card.json` (members, class order, temperatures, confidence threshold, skin-check limit, test metrics).

Privacy: the photo is decoded in memory and never written to disk or logged.

## 2. Run it on your laptop

```bash
git clone <your repo> && cd DermaLens_AI
python -m venv .venv && source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r app/requirements.txt
# copy everything in models/deploy/ from Drive into models/deploy/
uvicorn app.api:app --reload --port 8000
```

Open http://localhost:8000. Interactive API docs: http://localhost:8000/docs.

## 3. API reference

| Method | Path | Body | Returns |
|---|---|---|---|
| GET | `/health` | none | `{"status": "ok", "model_loaded": true}` |
| GET | `/api/info` | none | model name, classes, export date, disclaimer |
| POST | `/api/predict` | form field `file` (image) | prediction JSON (below) |

```json
{
  "label": "Tinea Corporis",
  "confidence": 0.87,
  "uncertain": false,
  "top3": [{"label": "Tinea Corporis", "probability": 0.87},
           {"label": "Eczema", "probability": 0.08},
           {"label": "Contact Dermatitis", "probability": 0.03}],
  "heatmap_png_base64": "iVBORw0KGgo...",
  "disclaimer": "DermaLens AI is a research prototype...",
  "processing_ms": 210.4
}
```

Errors: `415` wrong file type, `413` larger than 10 MB, `400` unreadable or smaller than 64 px.
`uncertain` is `true` below 50% confidence; the page then asks for a sharper photo instead of trusting one class.

Test from the command line:

```bash
curl -F "file=@photo.jpg;type=image/jpeg" http://localhost:8000/api/predict
```

## 4. Build the Docker image

```bash
docker build -f app/Dockerfile -t dermalens-app .
docker run -p 7860:7860 dermalens-app        # open http://localhost:7860
```

The image uses CPU-only PyTorch (~1 GB). It contains the code, `src/`, and `models/deploy/`.

## 5. Deploy (free options)

### Option A: Hugging Face Spaces (recommended for the demo)

1. Create a Space at huggingface.co → **New Space** → SDK **Docker** → hardware **CPU basic** (free).
2. Add a `README.md` at the Space root with this header:
   ```yaml
   ---
   title: DermaLens AI
   sdk: docker
   app_port: 7860
   ---
   ```
3. Copy into the Space repository: `app/`, `src/`, `models/deploy/` and `app/Dockerfile` (move it to the root as `Dockerfile`).
   Each `member_*.pt` is 30-350 MB: commit it with Git LFS (`git lfs track "*.pt"`) or upload it through the Space's web UI.
4. The Space builds and gives a public `https://<user>-dermalens-ai.hf.space` link.

### Option B: Render

1. Push the repository (with `models/deploy/`) to GitHub.
2. Render → **New Web Service** → connect the repo → **Docker**, Dockerfile path `app/Dockerfile`.
3. Free instance; Render sets `$PORT`, which the Dockerfile already uses.

## 6. Checklist before the viva demo

- [ ] `models/deploy/model_card.json` shows the final model and its test metrics
- [ ] `/health` returns `model_loaded: true` on the public URL
- [ ] Prediction takes under 2 s (synopsis target); the response's `processing_ms` shows it
- [ ] The disclaimer is visible on the page
- [ ] Tried a blurry photo: the page shows the "not confident" message
- [ ] A screen recording of the demo exists, in case the network fails

## 7. Possible next steps

* **Faster on phones:** export to ONNX and quantise to INT8 (≈4x smaller).
* **Confidence threshold per class:** tune on the validation set so uncertain Healthy predictions are flagged.
* **Out-of-scope detection:** a photo that is not skin should be rejected (e.g. a separate "is this skin?" check).
* **Feedback button** (with consent) for dermatologists to flag wrong answers, stored without the image.
