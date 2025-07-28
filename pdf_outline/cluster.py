# pdf_outline/cluster.py  –  heading‑level assignment
"""
Algorithm (General-Purpose Robust Version)
---------
1.  Attach prob / is_head / level=None to every line.
    *STRICT*: Aggressively filter lines that look like table/form cells.
2.  Demote short running headers repeated on ≥3 pages. (Noise reduction first)
3.  Numbering override (1., 1.1, 1.1.1). (High-confidence patterns first with corrected regex)
4.  Merge consecutive heads of same font + x‑bucket on title page.
5.  Choose Title (largest font, multi‑word, not logo‑tagline).
6.  Global font ranking → default H1/H2/H3. (Used as a fallback only)
"""


import re
import numpy as np
from itertools import groupby
from collections import Counter


# ---------- regex helpers (CORRECTED for general use) ---------------------
_num_re   = re.compile(r"^\s*\d+(\.\d+)*\s")       # numbering
_allcaps  = re.compile(r"^[A-Z0-9\s\-]+$")          # logo taglines
_dots_re  = re.compile(r"^[.\u2022·•_—–-]{3,}$")   # ..... or bullet/separator rows
_num_only = re.compile(r"^[0-9./-]+$")             # pure numeric cell


# CORRECTED: These patterns no longer require a trailing dot after the last digit.
pat_h1 = re.compile(r"^\d+\.\s")
pat_h2 = re.compile(r"^\d+\.\d+\s")
pat_h3 = re.compile(r"^\d+\.\d+\.\d+\s")


# ---------- table / form detector (STRICTER HEURISTIC) --------------------
def _looks_like_table(txt: str) -> bool:
    """
    Pragmatic filter for table / form fragments.
    STRICTER VERSION: More aggressive filtering to prioritize removing
    table/form content, at the risk of removing some valid short headings.
    """
    t = txt.strip()
    if not t: return True
    if _dots_re.fullmatch(t): return True
    if _num_only.fullmatch(t): return True
    if len(t) <= 2: return True
    words = t.split()
    if len(words) <= 2 and all(len(w) <= 4 for w in words): return True
    return False


# ---------- utility --------------------------------------------------------
def _merge_same_font_block(lines):
    """Merge consecutive heads with identical font & x‑position."""
    merged = []
    for _, group in groupby(
        lines,
        key=lambda l: (l["font_size"], round(l["bbox"][0] / 2) * 2),
    ):
        block = list(group)
        if len(block) == 1:
            merged.append(block[0])
        else:
            txt = " ".join(b["text"].strip() for b in block)
            head = block[0].copy()
            head["text"] = txt
            merged.append(head)
    return merged


# ---------- main (GENERAL-PURPOSE ROBUST LOGIC) ----------------------------
def assign_levels(lines, probs, p_thresh=0.90):
    # 1. basic mark + strict table/filter pass
    for L, p in zip(lines, probs):
        L["prob"]    = float(p)
        L["level"]   = None
        if _looks_like_table(L["text"]):
            L["is_head"] = False
        else:
            L["is_head"] = p >= p_thresh

    heads = [L for L in lines if L["is_head"]]
    if not heads:
        return lines

    # 2. Demote short running headers repeated on ≥3 pages
    REPEAT_THRESHOLD = 3
    counts = Counter(h["text"].strip() for h in heads)
    for h in heads:
        key = h["text"].strip()
        if counts[key] >= REPEAT_THRESHOLD and len(key.split()) <= 5:
            h["level"]   = None
            h["is_head"] = False

    heads = [h for h in heads if h["is_head"]]

    # 3. Numbering override: High-confidence patterns first
    for h in heads:
        if h["level"] is not None: continue
        t = h["text"].strip() + " " # Add space to handle no-text headings like "4.1"
        # The order is important: check for most specific (H3) first.
        if pat_h3.match(t):
            h["level"] = "H3"
        elif pat_h2.match(t):
            h["level"] = "H2"
        elif pat_h1.match(t):
            h["level"] = "H1"

    # 4. Title page processing
    title_page = min((h["page"] for h in heads), default=0)
    page_heads = [h for h in heads if h["page"] == title_page and h["level"] is None]
    if page_heads:
        page_heads = _merge_same_font_block(page_heads)
        
        top_size = max((h["font_size"] for h in page_heads), default=0)
        title_cands = [h for h in page_heads if h["font_size"] == top_size]

        title_cands = [
            h for h in title_cands
            if not (_allcaps.match(h["text"]) and len(h["text"].split()) > 4)
        ]

        if title_cands:
            title_text = " ".join(h["text"].strip() for h in title_cands)
            for h in title_cands:
                h["level"] = "Title"
                h["text"]  = title_text

    # 5. Global font ranking as a fallback for remaining pages
    remain = [h for h in heads if h["level"] is None]
    if remain:
        uniq   = sorted({h["font_size"] for h in remain}, reverse=True)[:3]
        size_map = {s: lvl for s, lvl in zip(uniq, ["H1", "H2", "H3"])}

        for h in remain:
            if h["level"] is None:
                h["level"] = size_map.get(h["font_size"], None)

    return lines
