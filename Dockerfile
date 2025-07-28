# ──────────────────────────────────────────────────────────────
# Dockerfile  (root of pdf_outline_project/)
#
# Build:
#   docker build --platform linux/amd64 -t pdfoutline.challenge .
#
# Run:
#   docker run --rm \
#     -v "$(pwd)/sample_dataset/pdfs:/app/input:ro" \
#     -v "$(pwd)/sample_dataset/outputs:/app/output" \
#     --network none \
#     pdfoutline.challenge
# ──────────────────────────────────────────────────────────────

FROM --platform=linux/amd64 python:3.10-slim

WORKDIR /app

# 1) Unbuffered Python, force HF/Transformers offline, pip only from wheelhouse
ENV PYTHONUNBUFFERED=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_HUB_OFFLINE=1 \
    PIP_NO_INDEX=1 \
    PIP_FIND_LINKS=/app/wheelhouse

# 2) Copy pre-downloaded wheels
COPY wheelhouse/ wheelhouse/

# 3) Copy requirements
COPY requirements.txt .

# 4) Copy your package metadata
COPY setup.py pyproject.toml ./

# 5) Copy your library and entry-point
COPY pdf_outline/ pdf_outline/
COPY process_pdfs.py .

# 6) Copy the ONNX models & head
COPY models/ models/

# 7) Install dependencies (offline) and then your package
RUN pip install --no-cache-dir -r requirements.txt \
 && pip install --no-cache-dir .

# 8) Default command: process everything under /app/input → /app/output
CMD ["python", "process_pdfs.py"]
