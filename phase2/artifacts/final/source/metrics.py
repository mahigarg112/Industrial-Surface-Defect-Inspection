"""Exact int64 pixel totals and explicit empty-class conventions."""
import numpy as np


def counts(prediction, truth):
    prediction, truth = np.asarray(prediction, bool), np.asarray(truth, bool)
    return np.stack([(prediction & truth).sum((1, 2), dtype=np.int64),
                     (prediction & ~truth).sum((1, 2), dtype=np.int64),
                     (~prediction & truth).sum((1, 2), dtype=np.int64)])


def summarize(totals):
    tp, fp, fn = np.asarray(totals, dtype=np.int64)
    result = {}
    for c in range(4):
        t, f, n = int(tp[c]), int(fp[c]), int(fn[c])
        result[str(c + 1)] = {
            "dice": 2*t / (2*t+f+n) if 2*t+f+n else None,
            "iou": t / (t+f+n) if t+f+n else None,
            "precision": t / (t+f) if t+f else None,
            "recall": t / (t+n) if t+n else None,
            "true_positive_pixels": t, "false_positive_pixels": f, "false_negative_pixels": n,
        }
    return {"macro_dice": float(np.mean([r["dice"] for r in result.values()]))
            if all(r["dice"] is not None for r in result.values()) else None,
            "macro_iou": float(np.mean([r["iou"] for r in result.values()]))
            if all(r["iou"] is not None for r in result.values()) else None,
            "per_class": result,
            "aggregation": "int64 pixel counts over entire split, then unweighted four-class macro; undefined ratios=null"}
