"""
DonutEncoder  –  lightweight ONNX front‑end that returns the CLS embedding
for each page image.

* loads **one** encoder_model.onnx into ONNXRuntime (CPU) with multi‑threading
* provides `.encode_pages(images, batch_size=8)` that returns   (N, 1024)
  CLS vectors as float32 numpy
"""

from __future__ import annotations
import os, warnings
from pathlib import Path
from typing import List

import numpy as np
from PIL import Image
import onnxruntime as ort


class DonutEncoder:
    def __init__(self, model_dir: Path | str):
        model_dir = Path(model_dir)
        fp = model_dir / "encoder_model.onnx"
        if not fp.exists():
            raise FileNotFoundError(f"Encoder model not found: {fp}")

        so = ort.SessionOptions()
        so.intra_op_num_threads = max(1, (os.cpu_count() or 1) // 2)  # e.g. 4 on 8‑core
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.session = ort.InferenceSession(
            str(fp), sess_options=so, providers=["CPUExecutionProvider"]
        )

        # input name & target shape
        self.input_name = self.session.get_inputs()[0].name
        self.size = 1280  # Donut base expects 1280×960 after padding

    # ------------------------------------------------------------------ utils
        # ------------------------------------------------------------------ utils
    def _preprocess(self, images: List[Image.Image]) -> np.ndarray:
        """
        Resize each page so that its *long edge* = 1280 px, then paste it onto
        a white 1280 × 960 canvas.  Output tensor shape is therefore
        (N, 3, 1280, 960) for every batch – exactly what the ONNX encoder
        expects.
        """
        H_CANVAS, W_CANVAS = 1280, 960
        out = np.zeros((len(images), 3, H_CANVAS, W_CANVAS), dtype=np.float32)

        for i, im in enumerate(images):
            im = im.convert("RGB")

            # proportional resize so max(H, W) = 1280
            scale = H_CANVAS / max(im.width, im.height)
            w, h  = int(im.width * scale), int(im.height * scale)
            im    = im.resize((w, h), Image.BILINEAR)

            # paste top‑left on white canvas 1280×960
            canvas = Image.new("RGB", (W_CANVAS, H_CANVAS), (255, 255, 255))
            canvas.paste(im, (0, 0))
            x = np.asarray(canvas).astype("float32") / 255.0       # HWC
            out[i] = x.transpose(2, 0, 1)                          # CHW

        return out


    # ------------------------------------------------------------------ public
        # ------------------------------------------------------------------ public
    def encode_pages(
        self,
        images: List[Image.Image],
        batch_size: int = 8,
    ) -> np.ndarray:
        """
        Return CLS vectors for a list of images.  Works even if the resized
        pages end up with *different* padded sizes by grouping identical shapes
        into the same ONNX run.
        """
        if not images:
            return np.zeros((0, 1024), dtype=np.float32)

        arr = self._preprocess(images)          # (N, 3, H?, W?) – H/W may differ
        cls_vecs = []

        # group indices that share the same (H,W) after padding
        hw2idx: dict[tuple[int, int], list[int]] = {}
        for i, x in enumerate(arr):
            _, h, w = x.shape
            hw2idx.setdefault((h, w), []).append(i)

        for (h, w), idx_list in hw2idx.items():
            # split into mini‑batches of identical shape
            for j in range(0, len(idx_list), batch_size):
                sel = idx_list[j : j + batch_size]
                out = self.session.run(
                    None, {self.input_name: arr[sel]}
                )[0][:, 0, :]                 # (B, 1024)
                cls_vecs.append((sel, out))

        # restore original order
        cls_final = np.zeros((len(images), 1024), dtype=np.float32)
        for sel, vec in cls_vecs:
            cls_final[sel, :] = vec
        return cls_final

