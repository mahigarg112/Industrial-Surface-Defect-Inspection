"""One bounded training-only Class2 sampler; keeps every normal/C1/C4 source.

Same9048 draws and1131 attempted updates as L's first additional epoch.
Replace354 pure-Class3 source slots with2 extra draws of each177 C2 source.
Extra crops jitter around actual C2 pixels; original draws retain K's policy.
"""
from collections import Counter
import numpy as np
import torch

from phase2.data import TrainDataset, native


class Class2QuotaDataset(TrainDataset):
    def __init__(self, rows, epoch, seed=42):
        super().__init__(rows, "patch", seed=seed, epoch=epoch, foreground_crops=True)
        present = np.asarray([[bool(rle) for rle in r["rle"].split("|")] for r in rows])
        positives = np.flatnonzero(present[:, 1])
        majority_only = np.flatnonzero(present[:, 2] & ~present[:, [0, 1, 3]].any(1))
        rng = np.random.default_rng([seed, epoch, 20261120])
        slots = rng.choice(majority_only, size=2*len(positives), replace=False)
        extras = [(int(index), copy) for index in positives for copy in [1, 2]]
        rng.shuffle(extras)
        self.extra = {int(slot): item for slot, item in zip(slots, extras)}
        draws = [self.extra[i][0] if i in self.extra else i for i in range(len(rows))]
        frequencies = Counter(draws)
        assert max(frequencies.values()) <= 3
        assert all(frequencies[int(index)] == 3 for index in positives)
        assert all(frequencies[int(index)] == 1 for index in np.flatnonzero(~present.any(1)))
        assert all(frequencies[int(index)] == 1 for index in np.flatnonzero(present[:, [0, 3]].any(1) & ~present[:, 1]))
        self.summary = {"draws": len(rows), "class2_sources": len(positives),
                        "class2_positive_draws": len(positives)*3, "source_repetition_cap": 3,
                        "normal_source_draws": int((~present.any(1)).sum()),
                        "normal_sources_preserved": True, "class1_class4_sources_preserved": True,
                        "pure_class3_slots_replaced": len(slots),
                        "extra_draws": [{"slot": slot, "source_index": item[0], "image_id": rows[item[0]]["image_id"],
                                         "copy": item[1]} for slot, item in sorted(self.extra.items())]}

    def _coordinate(self, masks, source_index, copy):
        positive = np.flatnonzero(masks[1].ravel())
        if not len(positive):
            raise ValueError("Quota extra requires a real training Class2 mask")
        rng = np.random.default_rng([self.seed, self.epoch, source_index, copy, 2])
        column = int(positive[int(rng.integers(len(positive)))] % 1600)
        low, high = max(0, column-255), min(column, 1344)
        start = int(rng.integers(low, high+1))
        if copy == 2 and high > low and start == self._coordinate(masks, source_index, 1):
            start = low + (start-low+1) % (high-low+1)
        return start

    def __getitem__(self, index):
        if index not in self.extra:
            return super().__getitem__(index)
        source_index, copy = self.extra[index]
        image, masks = native(self.rows[source_index])
        start = self._coordinate(masks, source_index, copy)
        image, masks = image[:, start:start+256], masks[:, :, start:start+256]
        assert image.shape == (256, 256, 3) and masks.shape == (4, 256, 256) and masks[1].any()
        return torch.from_numpy(image.transpose(2, 0, 1).copy()).float()/255, torch.from_numpy(masks.copy()).float()
