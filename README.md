# PDF‑Outline‑Challenge 🏗️

Extract a clean **heading outline (Title → H1/H2/H3)** from any PDF, fully offline, on CPU‑only hardware.

* **Encoder** – [Donut‑base INT8 ONNX] embedded in the image  
* **Classifier head** – small MLP (`donut_head.pkl`)  
* **Clustering** – rule–based, layout‑agnostic `pdf_outline/cluster.py`  
* **Renderer** – PyMuPDF (fitz) + Pillow, so no external OCR engine is required  
* **No Internet at runtime** – all models are shipped in `models/`

---

## 1  Quick start (Docker)

```bash
# clone
git clone https://github.com/<your‑org>/pdf_outline_project.git
cd pdf_outline_project

# build (≈2 min, <1 GB final image)
docker build --platform linux/amd64 -t pdfoutline.challenge .

# run – process every *.pdf inside sample_dataset/pdfs
docker run --rm \
  -v "$(pwd)/sample_dataset/pdfs:/app/input:ro" \
  -v "$(pwd)/output:/app/output" \
  --network none \
  pdfoutline.challenge




pdf_outline_project/
├── Dockerfile                # final container spec (CPU‑only, offline)
├── process_pdfs.py           # entry‑point: loops through /app/input
├── requirements.txt          # runtime deps (CPU wheels only)
├── README.md                 # ← you are here
│
├── models/                   # bundled open‑source models
│   └── donut_base_int8/int8/ … encoder & decoder .onnx
│   └── donut_head.pkl
│
├── pdf_outline/              # installable package
│   ├── __init__.py           # exposes extract_outline()
│   ├── cli.py                # thin wrapper around run()
│   ├── donut_infer.py        # ONNXRuntime inference
│   ├── cluster.py            # robust heading‑level assignment
│   ├── classify.py           # small MLP head
│   ├── render.py             # PDF → images
│   └── extract_lines.py      # line detection
│
└── sample_dataset/           # demo PDFs + schema
    ├── pdfs/
    └── schema/output_schema.json



| Stage                   | Tool                                        | Details                                                           |
| ----------------------- | ------------------------------------------- | ----------------------------------------------------------------- |
| **1. Raster** PDF pages | PyMuPDF (`fitz`) @ *DPI 120* (configurable) | renders each page to RGB                                          |
| **2. Encode**           | Donut‑base INT8 (ONNX)                      | outputs a 1024‑D cls‑token per page                               |
| **3. Classify** lines   | 3‑layer MLP on top of Donut embeddings      | yields a *heading‑probability* per text‑line                      |
| **4. Cluster**          | `pdf_outline.cluster.assign_levels`         | font‑analysis + numbering + repetition filtering ⇒ Title/H1/H2/H3 |
| **5. Dump JSON**        | `process_pdfs.py`                           | conforms to `sample_dataset/schema/output_schema.json`            |

The entire pipeline runs on CPU (amd64) and never tries to contact the Internet (TRANSFORMERS_OFFLINE=1, HF_HUB_OFFLINE=1)




KEY COMMANDS
| Purpose                   | Command                                                                                                          |
| ------------------------- | ---------------------------------------------------------------------------------------------------------------- |
| **Build image**           | `docker build --platform linux/amd64 -t pdfoutline.challenge .`                                                  |
| **Run pipeline**          | `docker run --rm -v $(pwd)/input:/app/input:ro -v $(pwd)/output:/app/output --network none pdfoutline.challenge` |
| **Local dev (no Docker)** | `python -m pdf_outline.cli <some.pdf> --dpi 120`                                                                 |


TROUBLESHOOTING
| Symptom                       | Fix                                                                           |
| ----------------------------- | ----------------------------------------------------------------------------- |
| `ConvInteger` not implemented | make sure you are using the INT8 models inside `models/donut_base_int8/int8/` |
| Out of memory (>16 GB)        | lower rendering DPI (`process_pdfs.DPI`) or batch size (`donut_infer.py`)     |
| No JSON produced              | check container log; a per‑PDF traceback is printed if extraction fails       |
