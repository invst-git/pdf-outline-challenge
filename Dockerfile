# ──────────────────────────────────────────────────────────────
# Dockerfile  (root of repo)
# Build:  docker build --platform=linux/amd64 -t pdfoutline.challenge .
# docker run --rm \
# -v "$(pwd)/sample_dataset/pdfs:/app/input:ro" \
# -v "$(pwd)/sample_dataset/outputs:/app/output" \
# --network none \
# pdfoutline.challenge

# ──────────────────────────────────────────────────────────────
FROM --platform=linux/amd64 python:3.10-slim

WORKDIR /app

# Make every Python print instantly visible & switch HF/Transformers to offline
ENV PYTHONUNBUFFERED=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_HUB_OFFLINE=1 \
    PIP_NO_INDEX=1               \
    PIP_FIND_LINKS=/app/wheelhouse

# 1. copy the whole repo (Docker ignores files in .dockerignore)
COPY . .

# 2. install all wheels from wheelhouse ONLY  (no network)
RUN pip install --no-cache-dir -r requirements.txt \
 && pip install --no-cache-dir .

# 3. run the batch processor
CMD ["python", "process_pdfs.py"]
