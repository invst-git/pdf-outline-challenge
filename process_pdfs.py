# process_pdfs.py
from pathlib import Path
import json, sys, traceback

# import the pipeline driver
from pdf_outline.cli import run as extract_outline

# ---------- config ---------------------------------------------------------
INPUT_DIR  = Path("/app/input")
OUTPUT_DIR = Path("/app/output")
MODEL_DIR  = Path("/app/models/donut_base_int8/int8")        # encoder/decoder .onnx
HEAD_PATH  = Path("/app/models/donut_head.pkl")              # your trained head
DPI        = 120                                              # default resolution
# ---------------------------------------------------------------------------


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for pdf in INPUT_DIR.glob("*.pdf"):
        out_json = OUTPUT_DIR / f"{pdf.stem}.json"
        try:
            data = extract_outline(
                pdf_path=str(pdf),
                out_path=out_json,
                model_dir=str(MODEL_DIR),
                head_path=str(HEAD_PATH),
                dpi=DPI,
            )
            # `run` already writes the file; data returned for completeness
            print(f"✓ {pdf.name}  →  {out_json.relative_to(OUTPUT_DIR.parent)}")
        except Exception as e:
            print(f"✗ {pdf.name}: {e}", file=sys.stderr)
            traceback.print_exc()

if __name__ == "__main__":
    main()
