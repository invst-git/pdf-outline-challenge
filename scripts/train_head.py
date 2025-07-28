# scripts/train_head.py
from pathlib import Path
import glob
import random
import gc
import numpy as np

from pdf_outline import render, extract_lines, donut_infer, classify, cluster

# --- choose a handful of bootstrap PDFs -----------------------------------
PDFS = glob.glob("pdfs/*.pdf")[:6]          # adjust folder / count as needed
random.shuffle(PDFS)

enc = donut_infer.DonutEncoder("models/donut_base_int8/int8")

X, y = [], []
for pdf in PDFS:
    imgs  = render.render_pdf(pdf, dpi=150, max_workers=2)
    lines = extract_lines.extract_lines(pdf)

    for page_idx, img in enumerate(imgs):
        cls_vec = enc.encode([img])[0][0]         # CLS token (1024,)
        page_lines = [L for L in lines if L["page"] == page_idx]

        # weak label = 1 if page has a Title/H1 per font heuristic
        cluster.assign_levels(page_lines, np.ones(len(page_lines)))
        label = 1 if any(L["level"] in ("Title", "H1") for L in page_lines) else 0

        X.append(cls_vec)
        y.append(label)

    del imgs, lines; gc.collect()

X = np.vstack(X)
y = np.array(y)
print("Training samples:", X.shape[0])

weights = classify.train_head(X, y)
Path("models").mkdir(exist_ok=True)
classify.save_head(weights, Path("models/donut_head.pkl"))
print("✅  Saved head weights to models/donut_head.pkl")
