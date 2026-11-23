"""One global photometric policy; images only, original crop/masks untouched."""
import numpy as np
import torch
from phase2.data import TrainDataset


def parameters(seed, epoch, index):
    rng = np.random.default_rng([seed, epoch, index, 1701])
    apply = bool(rng.random() < .5)
    return apply, float(rng.uniform(.9, 1.1)), float(rng.uniform(.9, 1.1))


def photometric(image, brightness=1., contrast=1.):
    if brightness == 1. and contrast == 1.: return image
    mean = image.mean()
    return (((image-mean)*contrast+mean)*brightness).clamp(0, 1)


class AugmentedDataset(TrainDataset):
    def __init__(self, rows, epoch, seed=42):
        super().__init__(rows, "patch", seed=seed, epoch=epoch, foreground_crops=True)
        self.augmentation_parameters = [parameters(seed, epoch, index) for index in range(len(rows))]

    def __getitem__(self, index):
        image, mask = super().__getitem__(index)
        apply, brightness, contrast = self.augmentation_parameters[index]
        if apply: image = photometric(image, brightness, contrast)
        return image, mask
