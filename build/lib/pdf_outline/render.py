# pdf_outline/render.py
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List
import fitz                      # PyMuPDF
from PIL import Image

def _render_one(pdf_path: Path, pno: int, dpi: int) -> Image.Image:
    with fitz.open(pdf_path) as doc:
        page = doc.load_page(pno)
        mat  = fitz.Matrix(dpi / 72, dpi / 72)          # 72 dpi is PDF default
        pix  = page.get_pixmap(matrix=mat, alpha=False)
        return Image.frombytes("RGB", (pix.width, pix.height), pix.samples)

def render_pdf(pdf_path: str | Path,
               dpi: int = 150,
               max_workers: int = 8) -> List[Image.Image]:
    """
    Return list[ PIL.Image ] – one per page, in original order.
    """
    pdf_path = Path(pdf_path)
    with fitz.open(pdf_path) as doc:
        npages = len(doc)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        pages = list(
            pool.map(lambda idx: _render_one(pdf_path, idx, dpi), range(npages))
        )
    return pages
    