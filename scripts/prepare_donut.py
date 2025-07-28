from pathlib import Path
from huggingface_hub import snapshot_download
from optimum.exporters.onnx import main_export
from optimum.onnxruntime import ORTQuantizer, AutoQuantizationConfig

MODEL = "naver-clova-ix/donut-base"
OUT   = Path("models/donut_base_int8")
OUT.mkdir(parents=True, exist_ok=True)

# 1) Download fp32
snapshot_download(MODEL, local_dir=OUT/"fp32", local_dir_use_symlinks=False)

# 2) Export to ONNX (creates encoder_model.onnx & decoder_model.onnx)
main_export(
    model_name_or_path=MODEL,
    output=OUT/"onnx",
    task="document-question-answering",
    opset=17,
    device="cpu",
)

# 3) INT8 quantisation per file
save_dir = OUT/"int8"; save_dir.mkdir(parents=True, exist_ok=True)
qcfg = AutoQuantizationConfig.avx512_vnni(is_static=False)

for fp in (OUT/"onnx").glob("*.onnx"):
    print("Quantising", fp.name)
    quant = ORTQuantizer.from_pretrained(OUT/"onnx", file_name=fp.name)
    quant.quantize(save_dir=save_dir, quantization_config=qcfg)
    # rename output to original filename
    (save_dir/"model_quantized.onnx").rename(save_dir/fp.name)

print("✅  INT8 Donut saved to", save_dir)
