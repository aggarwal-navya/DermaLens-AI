"""DermaLens AI web API.

Endpoints
    GET  /              the website (app/static/: home, analyse, model & research pages)
    GET  /health        "is the server up and the model loaded?"
    GET  /api/info      model name, classes, disclaimer
    POST /api/predict   upload one image -> prediction JSON

Privacy: uploaded images are processed in memory and never written to disk.

Run locally:
    uvicorn app.api:app --reload --port 8000
"""

from __future__ import annotations

import io
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError

from .inference import APP_VERSION, Predictor  # isort: skip  (puts src/ on the import path first)

from dermalens import metadata as meta  # noqa: E402  isort: skip

APP_DIR = Path(__file__).resolve().parent
BUNDLE_DIR = Path(os.environ.get("DERMALENS_BUNDLE", APP_DIR.parent / "models" / "deploy"))
MAX_UPLOAD_BYTES = 10 * 1024 * 1024                      # 10 MB
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp"}

predictor: Predictor | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model once when the server starts, not on every request."""
    global predictor
    predictor = Predictor(Path(os.environ.get("DERMALENS_BUNDLE", BUNDLE_DIR)))
    yield


app = FastAPI(title="DermaLens AI", version=APP_VERSION, lifespan=lifespan,
              description="Skin condition decision support for five conditions plus healthy skin. Not a diagnosis.")
app.add_middleware(CORSMiddleware, allow_origins=os.environ.get("ALLOWED_ORIGINS", "*").split(","),
                   allow_methods=["GET", "POST"], allow_headers=["*"])
app.mount("/static", StaticFiles(directory=APP_DIR / "static"), name="static")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_loaded": predictor is not None}


@app.get("/api/info")
def info() -> dict:
    return predictor.info()


def _check_metadata(age_group: str | None, sex: str | None, body_regions: str | None):
    """Validate the optional form fields against the allowed keys; empty -> None."""
    age_group, sex = (age_group or None), (sex or None)
    regions = [r.strip() for r in (body_regions or "").split(",") if r.strip()]
    if age_group and age_group not in meta.AGE_GROUPS:
        raise HTTPException(400, f"Unknown age group '{age_group}'.")
    if sex and sex not in meta.SEXES:
        raise HTTPException(400, f"Unknown sex '{sex}'.")
    unknown = [r for r in regions if r not in meta.BODY_REGIONS]
    if unknown:
        raise HTTPException(400, f"Unknown body region(s): {', '.join(unknown)}.")
    return age_group, sex, regions


@app.post("/api/predict")
async def predict(file: UploadFile = File(...), age_group: str | None = Form(None),
                  sex: str | None = Form(None), body_regions: str | None = Form(None)) -> dict:
    """Validate the upload and metadata, decode the image in memory, return the prediction."""
    age_group, sex, regions = _check_metadata(age_group, sex, body_regions)
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(415, f"Please upload a JPEG, PNG or WebP image (got {file.content_type}).")
    data = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "Image is larger than 10 MB.")
    try:
        image = Image.open(io.BytesIO(data))
        image.load()
    except (UnidentifiedImageError, OSError) as error:
        raise HTTPException(400, "The file could not be read as an image.") from error
    if min(image.size) < 64:
        raise HTTPException(400, "Image is too small (minimum 64 x 64 pixels).")
    return predictor.predict(image, age_group=age_group, sex=sex, body_regions=regions)


# Last: the website itself (index.html, analyze.html, about.html, styles.css, ...) at the root URL.
# Mounted after the API routes so /api/* and /health keep priority.
app.mount("/", StaticFiles(directory=APP_DIR / "static", html=True), name="site")
