# PDF‑Outline‑Challenge

Extract a clean **heading outline (Title → H1/H2/H3)** from any PDF, fully offline, on CPU‑only hardware.

- **Encoder** – Donut‑base INT8 (ONNX)  
- **Classifier head** – small MLP (`models/donut_head.pkl`)  
- **Clustering** – rule‑based, layout‑agnostic (`pdf_outline/cluster.py`)  
- **Renderer** – PyMuPDF (`fitz`) + Pillow, so **no** external OCR engine required  
- **Fully offline at runtime** – all models & wheels shipped in repo

---

## 1  Quick Start (Offline Docker Build)

1. **Pre‑download all Python wheels** into `wheelhouse/` (one‑time, already done):

    ```bash
    mkdir -p wheelhouse
    pip download \
      --only-binary=:all: \
      --no-deps \
      -d wheelhouse \
      -r requirements.txt \
      --extra-index-url https://download.pytorch.org/whl/cpu
    ```

2. **Build the Docker image** (uses `wheelhouse/` for offline install):

    ```bash
    docker build --platform linux/amd64 -t pdfoutline.challenge .
    ```

3. **Run** against your PDFs:

    ```bash
    docker run --rm \
      -v "$(pwd)/sample_dataset/pdfs:/app/input:ro" \
      -v "$(pwd)/sample_dataset/outputs:/app/output" \
      --network none \
      pdfoutline.challenge
    ```

   Your JSON outlines will appear in `sample_dataset/outputs/` in ≈2 min (<1 GB image).

---

## 2  Repository Layout

pdf_outline_project/
├── Dockerfile              # offline, CPU‑only container spec (uses wheelhouse/)
├── process_pdfs.py         # entry‑point: loops /app/input → /app/output
├── requirements.txt        # runtime deps
├── README.md               # ← you are here
│
├── wheelhouse/             # pre‑downloaded wheels for pip install --no-index
│
├── models/                 # bundled INT8 Donut & head
│   ├── donut_base_int8/int8/  # encoder+decoder .onnx
│   └── donut_head.pkl
│
├── pdf_outline/            # installable package
│   ├── __init__.py         # exposes CLI entrypoint
│   ├── cli.py              # thin wrapper around run()
│   ├── render.py           # PDF → RGB images
│   ├── donut_infer.py      # ONNXRuntime inference
│   ├── extract_lines.py    # text‑line detection
│   ├── classify.py         # small MLP head
│   └── cluster.py          # robust heading‑level assignment
│
└── sample_dataset/         # demo + schema
    ├── pdfs/               # input PDFs
    ├── outputs/            # output JSONs
    └── schema/
        └── output_schema.json



---

## 3  How It Works

| Stage                | Component                         | Description                                                     |
| -------------------- | --------------------------------- | --------------------------------------------------------------- |
| **1. Render** pages  | `pdf_outline.render` (PyMuPDF)    | Renders each page at *DPI* 120 (configurable)                   |
| **2. Encode** pages  | Donut‑base INT8 ONNX              | Produces one 1024‑D CLS token embedding per page                |
| **3. Classify** lines| MLP head (`donut_head.pkl`)       | Yields a heading‑probability for each detected text line        |
| **4. Cluster** heads | `pdf_outline.cluster.assign_levels`| Font-size ranking + numbering rules + repeat filtering ⇒ Title/H1… |
| **5. Dump JSON**     | `process_pdfs.py`                 | Writes `<pdf_name>.json` matching `output_schema.json`          |

All steps run **CPU**, **offline**, on **amd64** with ≤16 GB RAM.

---

## 4  Key Commands

| Purpose              | Command                                                                                                                         |
| -------------------- | ------------------------------------------------------------------------------------------------------------------------------- |
| **Build** image      | `docker build --platform linux/amd64 -t pdfoutline.challenge .`                                                                  |
| **Run** pipeline     | `docker run --rm -v $(pwd)/sample_dataset/pdfs:/app/input:ro -v $(pwd)/sample_dataset/outputs:/app/output --network none pdfoutline.challenge` |
| **Local dev**        | `python -m pdf_outline.cli <file.pdf> --dpi 120`                                                                                |

---

## 5  Troubleshooting

| Symptom                               | Fix                                                                                 |
| ------------------------------------- | ----------------------------------------------------------------------------------- |
| `ConvInteger` node not implemented    | Ensure you're using the **INT8** ONNX files under `models/donut_base_int8/int8/`     |
| Out-of-memory (>16 GB)                | Lower DPI (`--dpi 80`) or reduce batch size (in `donut_infer.py`)                  |
| No JSON outputs                       | Verify you mounted `/app/output` correctly; inspect container logs for per‑PDF errors |

---

