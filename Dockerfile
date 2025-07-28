FROM --platform=linux/amd64 python:3.10-slim AS build

WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_HUB_OFFLINE=1

# copy full repo (honours .dockerignore)
COPY . .

# install slim runtime deps
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir .

CMD ["python", "process_pdfs.py"]
