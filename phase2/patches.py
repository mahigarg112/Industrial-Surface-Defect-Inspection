"""Native-coordinate tiling; blend probabilities before applying thresholds."""
import numpy as np
import torch
from PIL import Image


def starts(length, patch, stride):
    if min(length, patch, stride) <= 0 or stride > patch:
        raise ValueError("Require positive dimensions and stride <= patch")
    values = list(range(0, max(length - patch, 0) + 1, stride))
    last = max(0, length - patch)
    if values[-1] != last:
        values.append(last)
    return values


def boxes(height, width, patch=256, stride=128):
    return [(y, x, min(y + patch, height), min(x + patch, width))
            for y in starts(height, patch, stride) for x in starts(width, patch, stride)]


def weight_window(patch, kind="hann"):
    if kind == "uniform":
        return np.ones((patch, patch), dtype=np.float32)
    if kind != "hann":
        raise ValueError(kind)
    # Strictly positive borders prevent zero denominators at image boundaries.
    axis = np.maximum(np.hanning(patch).astype(np.float32), 0.01)
    return np.outer(axis, axis)


def blend(probabilities, coordinates, height, width, patch=256, kind="hann"):
    total = np.zeros((4, height, width), dtype=np.float32)
    count = np.zeros((height, width), dtype=np.float32)
    window = weight_window(patch, kind)
    if len(probabilities) != len(coordinates):
        raise ValueError("One prediction per box required")
    for prediction, (y, x, bottom, right) in zip(probabilities, coordinates):
        h, w = bottom - y, right - x
        weight = window[:h, :w]
        total[:, y:bottom, x:right] += prediction[:, :h, :w] * weight
        count[y:bottom, x:right] += weight
    if np.any(count <= 0):
        raise ValueError("Incomplete image coverage")
    return total / count[None]


@torch.inference_mode()
def predict(model, image, mode, device="cuda", batch_size=8, patch=256, stride=128,
            kind="hann", amp=True):
    """RGB uint8 image -> four full native-resolution probability channels."""
    height, width = image.shape[:2]
    if mode == "whole":
        resized = np.asarray(Image.fromarray(image).resize((512, 128), Image.Resampling.BILINEAR))
        tensor = torch.from_numpy(resized.transpose(2, 0, 1).copy()).float()[None].to(device) / 255
        with torch.autocast(device_type=device, enabled=amp and device == "cuda"):
            logits = model(tensor)
        prob = logits.float().sigmoid()
        return torch.nn.functional.interpolate(prob, (height, width), mode="bilinear",
                                               align_corners=False)[0].cpu().numpy()
    if mode != "patch":
        raise ValueError(mode)
    coordinates = boxes(height, width, patch, stride)
    # Accumulate on CPU as each GPU batch completes; no full-image GPU tensor.
    total = np.zeros((4, height, width), dtype=np.float32)
    count = np.zeros((height, width), dtype=np.float32)
    window = weight_window(patch, kind)
    for offset in range(0, len(coordinates), batch_size):
        batch_boxes = coordinates[offset:offset + batch_size]
        crops = []
        for y, x, bottom, right in batch_boxes:
            crop = image[y:bottom, x:right]
            crop = np.pad(crop, ((0, patch - crop.shape[0]), (0, patch - crop.shape[1]), (0, 0)), mode="edge")
            crops.append(crop.transpose(2, 0, 1))
        tensor = torch.from_numpy(np.stack(crops)).float().to(device) / 255
        with torch.autocast(device_type=device, enabled=amp and device == "cuda"):
            logits = model(tensor)
        probabilities = logits.float().sigmoid().cpu().numpy()
        for probability, (y, x, bottom, right) in zip(probabilities, batch_boxes):
            h, w = bottom - y, right - x
            weight = window[:h, :w]
            total[:, y:bottom, x:right] += probability[:, :h, :w] * weight
            count[y:bottom, x:right] += weight
    if np.any(count <= 0):
        raise ValueError("Incomplete image coverage")
    return total / count[None]
