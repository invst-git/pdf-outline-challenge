#!/usr/bin/env python3
"""
quick_test.py  – sanity‑check Task 2 utilities

Usage:
    # from project root (~/pdf_outline_project)
    python quick_test.py pdfs/E0CCG5S312.pdf
"""

from pathlib import Path
import sys

from pdf_outline import render, extract_lines


def main(pdf_path: str | Path):
    pdf_path = Path(pdf_path).expanduser().resolve()
    if not pdf_path.exists():
        print(f"❌  File not found: {pdf_path}")
        sys.exit(1)

    # --- Render pages ------------------------------------------------------
    pages = render.render_pdf(pdf_path, dpi=150, max_workers=4)
    print(f"Rendered {len(pages)} pages – first page size: {pages[0].size}")

    # --- Extract text lines ------------------------------------------------
    lines = extract_lines.extract_lines(pdf_path)
    print(f"Extracted {len(lines)} text lines.")

    # show first 5 lines for inspection
    for i, line in enumerate(lines[:5], 1):
        print(f"{i:02d} | page {line['page']} | size {line['font_size']:.1f} | "
              f"{'BOLD' if line['is_bold'] else 'norm'} | {line['text'][:80]!r}")

    print("Quick‑test completed ✔")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python quick_test.py path/to/file.pdf")
        sys.exit(2)
    main(sys.argv[1])
