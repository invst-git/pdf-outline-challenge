#!/usr/bin/env python3
"""
Quantise Donut ONNX models to dynamic INT8 (CPU) – no internet required.
Run once; produces ~70 MB encoder and ~15 MB decoder.
"""

from pathlib import Path
from onnxruntime.quantization import quantize_dynamic, QuantType


SRC_DIR = Path("models/donut_base_int8/onnx")   # where your FP32 .onnx live
DST_DIR = Path("models/donut_base_int8/int8")   # output folder
DST_DIR.mkdir(parents=True, exist_ok=True)

pairs = [
    ("encoder_model.onnx",  "encoder_model_quantized.onnx"),
    ("decoder_model.onnx",  "decoder_model_quantized.onnx"),
]

for src_name, dst_name in pairs:
    src = SRC_DIR / src_name
    dst = DST_DIR / dst_name
    if not src.exists():
        raise SystemExit(f"❌  {src} not found – export FP32 first.")
    print(f"➜  Quantising {src.name}  ➜  {dst.name}")
    quantize_dynamic(
        model_input  = str(src),
        model_output = str(dst),
        weight_type  = QuantType.QInt8,
        op_types_to_quantize = ["MatMul", "Add"],   # int8 weights
    )
print("\n✅  All done.")
print("Files:")
for f in DST_DIR.glob("*quantized.onnx"):
    print("  ", f.name, f.stat().st_size // (1024*1024), "MB")
