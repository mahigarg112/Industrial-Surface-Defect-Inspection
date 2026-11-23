"""Shared, offline inference and export for the CLI, Streamlit and release tests."""
import argparse
import hashlib
import io
import json
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError
import torch

from phase2.common import ROOT, load_model, seed_all, sha
from phase2.patches import predict

PACKAGE = ROOT / "phase2/artifacts/final"
MODEL_SHA = "0ce6c174d27fff24551faa7ce94f7a2ec7f9a98294a0eac7746a8c61b89eed59"
LOCKED_THRESHOLDS = (0.949999988079071, 0.9750000238418579, 0.5, 0.949999988079071)
MAX_BYTES = 20 * 1024 * 1024
MAX_PIXELS = 4_000_000
COLORS = np.array([[255, 60, 60], [50, 255, 100], [60, 150, 255], [255, 220, 50]], np.float32)
OVERLAY_ALPHA = 0.4  # Display only; probabilities and binary masks are unchanged.
DISCLAIMER = "This model is intended for academic research and demonstration only. Predictions may contain false alarms or missed defects. Do not use its outputs to automatically accept, reject or certify manufactured products."


def decode_upload(data, filename):
    """Validate the extension, actual format, byte size and dimensions before decoding."""
    if not data:
        raise ValueError("The file is empty. Upload a JPG, JPEG or PNG image.")
    if len(data) > MAX_BYTES:
        raise ValueError("Image exceeds the 20 MiB file limit.")
    name = str(filename).replace("\\", "/").split("/")[-1]
    extension = Path(name).suffix.lower()
    formats = {".jpg": "JPEG", ".jpeg": "JPEG", ".png": "PNG"}
    if extension not in formats:
        raise ValueError("Supported file types are JPG, JPEG and PNG.")
    try:
        with Image.open(io.BytesIO(data)) as opened:
            if opened.format != formats[extension]:
                raise ValueError("The decoded image format does not match the filename extension.")
            width, height = opened.size
            if min(width, height) < 16 or max(width, height) > 8192 or width * height > MAX_PIXELS:
                raise ValueError("Use an image with sides 16–8192 pixels and at most 4 million pixels.")
            opened.load()
            orientation = opened.getexif().get(274, 1)
            oriented = ImageOps.exif_transpose(opened)
            width, height = oriented.size
            image = np.asarray(oriented.convert("RGB")).copy()
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise ValueError("The image cannot be decoded. It may be damaged or unsupported.") from exc
    return image, {"filename": name[:200], "sha256": hashlib.sha256(data).hexdigest(),
                   "bytes": len(data), "width": width, "height": height,
                   "exif_orientation_applied": orientation in (2, 3, 4, 5, 6, 7, 8)}


class Inspector:
    """One pinned model. Only device placement can change; weights never change."""
    def __init__(self, device="auto", package=PACKAGE):
        if device not in ("auto", "cpu", "cuda"):
            raise ValueError("Device must be auto, cpu or cuda.")
        self.package = Path(package)
        self.lock = json.loads((self.package / "lock.json").read_text(encoding="utf-8"))
        if self.lock["checkpoint_sha256"] != MODEL_SHA or sha(self.package / "model.pt") != MODEL_SHA:
            raise ValueError("The checkpoint does not match the preserved K release model.")
        self.locked_thresholds = tuple(self.lock["thresholds"])
        if self.locked_thresholds != LOCKED_THRESHOLDS:
            raise ValueError("The original K thresholds have changed.")
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable. Select Auto or CPU.")
        self.preference = device
        self.device = "cuda" if device != "cpu" and torch.cuda.is_available() else "cpu"
        self.fallback_reason = "CUDA unavailable; using CPU." if device == "auto" and self.device == "cpu" else None
        seed_all(42)
        self.model, checkpoint = load_model(self.package / "model.pt", device=self.device)
        self.mode = checkpoint["config"]["mode"]
        self.parameters = sum(p.numel() for p in self.model.parameters())
        if self.parameters != 482788 or self.mode != "patch":
            raise ValueError("Unexpected architecture or input pipeline for the K release.")
        self.forward_runs = 0
        self._mutex = threading.Lock()

    def probabilities(self, image):
        with self._mutex:
            started = time.perf_counter()
            if self.device == "cuda":
                torch.cuda.synchronize()
            try:
                probability = predict(self.model, image, self.mode, device=self.device,
                                      batch_size=8 if self.device == "cuda" else 2,
                                      stride=self.lock["stride"], kind=self.lock["blend"])
            except torch.OutOfMemoryError:
                if self.preference != "auto" or self.device != "cuda":
                    raise
                self.model = self.model.cpu()
                self.device = "cpu"
                self.fallback_reason = "GPU memory was insufficient; inference continued on CPU."
                torch.cuda.empty_cache()
                probability = predict(self.model, image, self.mode, device="cpu", batch_size=2,
                                      stride=self.lock["stride"], kind=self.lock["blend"])
            if self.device == "cuda":
                torch.cuda.synchronize()
            if probability.shape != (4, *image.shape[:2]) or not np.isfinite(probability).all():
                raise RuntimeError("Invalid native-coordinate probability output.")
            if probability.min() < 0 or probability.max() > 1:
                raise RuntimeError("Probability output is outside [0,1].")
            self.forward_runs += 1
            return probability, {"inference_seconds": time.perf_counter() - started,
                                 "device": self.device,
                                 "device_name": torch.cuda.get_device_name(0) if self.device == "cuda" else "CPU",
                                 "fallback_reason": self.fallback_reason}


def png_bytes(array):
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="PNG")
    return buffer.getvalue()


def render(image, probability, thresholds):
    thresholds = np.asarray(thresholds, dtype=np.float32)
    if thresholds.shape != (4,) or not np.isfinite(thresholds).all() or np.any((thresholds < 0) | (thresholds > 1)):
        raise ValueError("Provide four finite thresholds between 0 and 1.")
    if probability.shape != (4, *image.shape[:2]) or not np.isfinite(probability).all():
        raise ValueError("Probabilities and original image dimensions do not match.")
    masks = probability >= thresholds[:, None, None]
    color = np.einsum("chw,ck->hwk", masks.astype(np.float32), COLORS)
    active = masks.sum(0)
    color /= np.maximum(active[:, :, None], 1)
    overlay = np.where(active[:, :, None] > 0,
                       (1 - OVERLAY_ALPHA) * image + OVERLAY_ALPHA * color, image).astype(np.uint8)
    count = masks.sum((1, 2), dtype=np.int64)
    return {"masks": masks, "overlay": overlay, "overlay_png": png_bytes(overlay),
            "mask_pngs": [png_bytes(mask.astype(np.uint8) * 255) for mask in masks],
            "positive_pixels": count.tolist(),
            "area_percent": (count * 100.0 / (image.shape[0] * image.shape[1])).tolist()}


def inspection_report(source, timing, view, thresholds, locked_thresholds=LOCKED_THRESHOLDS):
    exact = np.array_equal(np.asarray(thresholds, np.float32), np.asarray(locked_thresholds, np.float32))
    return {"schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
            "input": source, "model": {"candidate": "K_extended_patch", "parameters": 482788,
                                       "checkpoint_sha256": MODEL_SHA, "pipeline": "native overlapping patches / Hann probability blending"},
            "inference": timing, "thresholds": [float(x) for x in thresholds],
            "locked_validation_thresholds": list(locked_thresholds),
            "operating_point": "locked_validation_settings" if exact else "exploratory_user_settings_not_evaluated",
            "classes": [{"class_id": c + 1, "predicted_pixels": view["positive_pixels"][c],
                         "affected_area_percent": view["area_percent"][c]} for c in range(4)],
            "ground_truth_available": False, "accuracy_on_this_image": None,
            "note": "Independent class masks may overlap; area percentages need not sum to 100. Sigmoid outputs are not calibrated confidence.",
            "disclaimer": DISCLAIMER}


def report_bytes(report):
    return json.dumps(report, indent=2, allow_nan=False).encode("utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Choose a fresh output directory.")
    image, source = decode_upload(args.image.read_bytes(), args.image.name)
    inspector = Inspector(args.device)
    probability, timing = inspector.probabilities(image)
    view = render(image, probability, inspector.locked_thresholds)
    report = inspection_report(source, timing, view, inspector.locked_thresholds)
    args.output.mkdir(parents=True)
    (args.output / "overlay.png").write_bytes(view["overlay_png"])
    (args.output / "inspection.json").write_bytes(report_bytes(report))
    for c, content in enumerate(view["mask_pngs"], 1):
        (args.output / f"class_{c}.png").write_bytes(content)
    np.savez_compressed(args.output / "prediction.npz", probabilities=probability, masks=view["masks"])
    print(f"Saved {args.output}; {timing['device']}; native shape {probability.shape}")


if __name__ == "__main__":
    main()
