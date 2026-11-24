# DermaLens AI web app - CPU-only image.
# Build from the repository root:   docker build -f app/Dockerfile -t dermalens-app .
# Run:                               docker run -p 7860:7860 dermalens-app
FROM python:3.11-slim

WORKDIR /srv
# CPU build of PyTorch keeps the image ~1 GB instead of ~5 GB.
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu
COPY app/requirements.txt app/requirements.txt
RUN pip install --no-cache-dir -r app/requirements.txt

COPY src/ src/
COPY app/ app/
COPY models/deploy/ models/deploy/

ENV DERMALENS_BUNDLE=/srv/models/deploy
EXPOSE 7860
# 7860 is the port Hugging Face Spaces expects; Render/Railway read $PORT.
CMD ["sh", "-c", "uvicorn app.api:app --host 0.0.0.0 --port ${PORT:-7860}"]
