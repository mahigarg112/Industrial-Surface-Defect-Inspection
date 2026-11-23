"""Severstal CSV loading and pixel-wise mask utilities."""

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


CLASS_NAMES = ["class_1", "class_2", "class_3", "class_4"]
ORIGINAL_WIDTH = 1600
ORIGINAL_HEIGHT = 256


def decode_rle(encoded_pixels: str, height: int = ORIGINAL_HEIGHT,
               width: int = ORIGINAL_WIDTH) -> np.ndarray:
    """Decode Kaggle's 1-based, column-major RLE to a (height, width) uint8 mask."""
    mask = np.zeros(height * width, dtype=np.uint8)
    encoded_pixels = (encoded_pixels or "").strip()
    if not encoded_pixels:
        return mask.reshape((height, width), order="F")
    values = np.asarray(encoded_pixels.split(), dtype=np.int64)
    if values.size % 2:
        raise ValueError("RLE must contain start/length pairs")
    starts = values[0::2] - 1
    lengths = values[1::2]
    ends = starts + lengths
    if np.any(starts < 0) or np.any(ends > mask.size) or np.any(lengths < 0):
        raise ValueError("RLE run falls outside the declared image dimensions")
    for start, end in zip(starts, ends):
        mask[start:end] = 1
    return mask.reshape((height, width), order="F")


def read_annotations(csv_path: Path) -> dict[str, list[str]]:
    annotations = defaultdict(lambda: [""] * len(CLASS_NAMES))
    with Path(csv_path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        expected = {"ImageId", "ClassId", "EncodedPixels"}
        if not expected.issubset(reader.fieldnames or []):
            raise ValueError(f"Expected CSV columns {sorted(expected)}, got {reader.fieldnames}")
        for row in reader:
            class_index = int(row["ClassId"]) - 1
            if not 0 <= class_index < len(CLASS_NAMES):
                raise ValueError(f"Invalid Severstal class ID: {row['ClassId']}")
            rle = (row["EncodedPixels"] or "").strip()
            if rle:
                if annotations[row["ImageId"]][class_index]:
                    raise ValueError(f"Duplicate annotation for {row['ImageId']} class {class_index + 1}")
                annotations[row["ImageId"]][class_index] = rle
    return dict(annotations)


class SeverstalSegmentationDataset(Dataset):
    """Loads RGB inputs and four aligned binary class masks."""

    def __init__(self, image_dir: Path, manifest_csv: Path, split: str,
                 image_width: int = 512, image_height: int = 128):
        self.image_dir = Path(image_dir)
        with Path(manifest_csv).open(newline="", encoding="utf-8") as handle:
            self.records = [row for row in csv.DictReader(handle) if row["split"] == split]
        if not self.records:
            raise ValueError(f"No images found for split {split!r} in {manifest_csv}")
        self.width, self.height = image_width, image_height

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        with Image.open(self.image_dir / record["image_id"]) as source:
            image = source.convert("RGB").resize((self.width, self.height), Image.Resampling.BILINEAR)
            image_array = np.asarray(image, dtype=np.float32) / 255.0
        # Normalize RGB to [0,1]. Mask resizing uses nearest-neighbor to retain binary labels.
        image_tensor = torch.from_numpy(image_array.transpose(2, 0, 1).copy())
        masks = np.zeros((len(CLASS_NAMES), self.height, self.width), dtype=np.float32)
        for index, rle in enumerate(record["rle"].split("|")):
            if rle:
                original = decode_rle(rle)
                resized = Image.fromarray(original).resize((self.width, self.height), Image.Resampling.NEAREST)
                masks[index] = np.asarray(resized, dtype=np.float32)
        return image_tensor, torch.from_numpy(masks), record["image_id"]
