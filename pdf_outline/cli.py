#!/usr/bin/env python3
"""extract_outline  –  offline PDF outline extractor CLI"""

from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np

from pdf_outline import render_pdf, extract_lines, assign_levels
from pdf_outline.donut_infer import DonutEncoder
from pdf_outline.classify import load_head, predict


# --------------------------------------------------------------------- CLI
def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract Title + H1‑H3 outline from PDF.")
    p.add_argument("pdf", type=Path, help="input PDF file")
    p.add_argument("-o", "--out", type=Path, help="output JSON (default <PDF>_outline.json)")
    p.add_argument("--dpi", type=int, default=120, help="render DPI (120 recommended)")
    p.add_argument("--model", type=Path, default=Path("models/donut_base_int8/int8"),
                   help="folder with encoder_model.onnx")
    p.add_argument("--head", type=Path, default=Path("models/donut_head.pkl"),
                   help="logistic‑head weights file")
    return p.parse_args()


# ------------------------------------------------------------------ driver
def run(pdf_path: Path, out_path: Path,
        dpi: int, model_dir: Path, head_path: Path) -> None:

    t0 = time.time()

    # 1. raster + text boxes
    pages  = render_pdf(pdf_path, dpi=dpi, max_workers=2)
    lines  = extract_lines(pdf_path)

    # 2. CLS embeddings (batched)
    encoder = DonutEncoder(model_dir)
    cls_vecs = encoder.encode_pages(pages, batch_size=8)          # (N,1024)

    # 3. heading probability per page then broadcast to lines
    W = load_head(head_path)
    page_probs = predict(cls_vecs, W)                             # (N,)
    probs = np.concatenate([np.full(sum(l["page"] == p for l in lines), page_probs[p])
                            for p in range(len(pages))])

    lines = assign_levels(lines, probs, p_thresh=0.60)

    title_line = next((L for L in lines if L["level"] == "Title"), None)
    outline = {
        "title": title_line["text"] if title_line else "",
        "outline": [
            {"level": L["level"], "text": L["text"], "page": L["page"] + 1}
            for L in lines if L["level"] in ("H1", "H2", "H3")
        ],
    }

    out_path.write_text(json.dumps(outline, indent=2, ensure_ascii=False))
    print(f"✓ Saved outline to {out_path}  ({time.time() - t0:.1f}s)")


def main() -> None:
    args = _parse()
    pdf  = args.pdf.resolve()
    if not pdf.exists():
        raise SystemExit(f"❌ PDF not found: {pdf}")
    out = args.out or pdf.with_name(pdf.stem + "_outline.json")
    run(pdf, out, args.dpi, args.model, args.head)


if __name__ == "__main__":
    main()
