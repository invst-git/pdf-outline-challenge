#!/usr/bin/env python3
"""
outline_demo.py  – low‑RAM streaming outline extractor
"""

from pathlib import Path
import sys, json, gc, numpy as np

from pdf_outline import render, extract_lines, donut_infer, classify, cluster


def main(pdf_path):
    pdf_path = Path(pdf_path).expanduser().resolve()
    if not pdf_path.exists():
        sys.exit(f"File not found: {pdf_path}")

    # render all pages first (PyMuPDF is light on RAM)
    pages = render.render_pdf(pdf_path, dpi=150, max_workers=2)
    lines = extract_lines.extract_lines(pdf_path)

    enc      = donut_infer.DonutEncoder("models/donut_base_int8/int8")
    weights  = classify.load_head(Path("models/donut_head.pkl"))
    page_probs = []

    # --- stream page by page to keep memory low ---------------------------
    for pno, img in enumerate(pages):
        cls_vec = enc.encode([img])[0][0]           # (1024,)
        prob    = classify.predict(cls_vec[None, :], weights)[0]
        page_probs.append(prob)
        # free immediately
        del img, cls_vec; gc.collect()

    # expand probs per line
    probs = []
    for p in range(len(pages)):
        n = sum(1 for L in lines if L["page"] == p)
        probs.extend([page_probs[p]] * n)
    probs = np.array(probs)

    lines = cluster.assign_levels(lines, probs, p_thresh=0.60)

    title_line = next((L for L in lines if L["level"] == "Title"), None)
    outline = {
        "title": title_line["text"] if title_line else "",
        "outline": [
            {"level": L["level"], "text": L["text"], "page": L["page"] + 1}
            for L in lines if L["level"] in ("H1", "H2", "H3")
        ],
    }
    print(json.dumps(outline, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("Usage: python outline_demo.py path/to/file.pdf")
    main(sys.argv[1])
