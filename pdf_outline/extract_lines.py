# pdf_outline/extract_lines.py
from pathlib import Path
from typing import List, Dict, Any
import fitz

def extract_lines(pdf_path: str | Path) -> List[Dict[str, Any]]:
    """
    Output: list of dicts with keys
       page, text, bbox(x0,y0,x1,y1), font_size, font_name, is_bold
    Empty/whitespace lines are skipped.
    """
    out: List[Dict[str, Any]] = []
    pdf_path = Path(pdf_path)

    with fitz.open(pdf_path) as doc:
        for pno in range(len(doc)):
            page = doc.load_page(pno)
            blocks = page.get_text("dict", flags=fitz.TEXTFLAGS_TEXT)["blocks"]

            for b in blocks:
                if b["type"] != 0:        # 0 = text, 1 = image, etc.
                    continue
                for line in b["lines"]:
                    if not line["spans"]:
                        continue
                    text = "".join(s["text"] for s in line["spans"]).strip()
                    if not text:
                        continue
                    # assume uniform style inside a line – take first span
                    span = line["spans"][0]
                    out.append(
                        {
                            "page": pno,
                            "text": text,
                            "bbox": tuple(line["bbox"]),        # (x0,y0,x1,y1)
                            "font_size": span["size"],
                            "font_name": span["font"],
                            "is_bold": "Bold" in span["font"],
                        }
                    )
    return out
