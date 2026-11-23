import csv

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .common import IMAGES, OUT, decode_rle
from .patches import boxes


def records(split, manifest=None):
    with (manifest or OUT / "partition.csv").open(newline="", encoding="utf-8") as f:
        return [r for r in csv.DictReader(f) if r["split"] == split]


def native(record):
    with Image.open(IMAGES / record["image_id"]) as f:
        image = np.asarray(f.convert("RGB")).copy()
    if image.shape != (256, 1600, 3):
        raise ValueError(f"Invalid native dimensions: {record['image_id']}: {image.shape}")
    rles = record["rle"].split("|")
    if len(rles) != 4:
        raise ValueError("Require four annotation channels")
    masks = np.stack([decode_rle(r) for r in rles])
    return image, masks


class TrainDataset(Dataset):
    def __init__(self, rows, mode, seed=42, epoch=0, foreground_crops=False):
        self.rows, self.mode, self.seed, self.epoch = rows, mode, seed, epoch
        self.coordinates = boxes(256, 1600)
        self.foreground_crops = foreground_crops
        if foreground_crops:
            if mode != "patch":
                raise ValueError("Foreground cropping is a training-patch option only")
            self.class_counts = [sum(bool(row["rle"].split("|")[c]) for row in rows) for c in range(4)]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        image, masks = native(self.rows[index])
        if self.mode == "whole":
            image = np.asarray(Image.fromarray(image).resize((512, 128), Image.Resampling.BILINEAR))
            masks = np.stack([np.asarray(Image.fromarray(m).resize((512, 128), Image.Resampling.NEAREST)) for m in masks])
        elif self.mode == "patch":
            # One uniformly drawn grid patch per source image per epoch. No label-aware sampling.
            rng = np.random.default_rng([self.seed, self.epoch, index])
            coordinates = self.coordinates
            # getattr preserves default behavior for already-running Windows DataLoader workers.
            if getattr(self,"foreground_crops",False):
                active = [c for c in range(4) if masks[c].any()]
                if active:
                    focus = min(active,key=lambda c:(self.class_counts[c],c))
                    coordinates = [box for box in coordinates if masks[focus,box[0]:box[2],box[1]:box[3]].any()]
            y, x, bottom, right = coordinates[int(rng.integers(len(coordinates)))]
            image, masks = image[y:bottom, x:right], masks[:, y:bottom, x:right]
        else:
            raise ValueError(self.mode)
        return torch.from_numpy(image.transpose(2, 0, 1).copy()).float() / 255, torch.from_numpy(masks.copy()).float()
