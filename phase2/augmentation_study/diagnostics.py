"""Multilabel-set confusion and declared error strata at native resolution."""
from pathlib import Path
import json
import sys
import time

import numpy as np
from scipy import ndimage
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from phase2.common import load_model, seed_all
from phase2.data import native, records
from phase2.patches import predict
from phase2.improvement_sprint.sprint import Accumulator, small_components

HERE = Path(__file__).resolve().parent
PROTOCOL = json.loads((HERE / "protocol.json").read_text())


def label_bits(masks):
    result = np.zeros(masks.shape[1:], np.uint8)
    for c in range(4): result |= masks[c].astype(np.uint8) << c
    return result


class Context:
    def __init__(self, image, truth):
        self.truth = np.asarray(truth, bool)
        self.bits = label_bits(self.truth)
        self.background = self.bits == 0
        self.small = small_components(self.truth)
        self.edge = np.zeros(image.shape[:2], bool)
        self.edge[:16] = self.edge[-16:] = True
        self.edge[:, :16] = self.edge[:, -16:] = True
        gray = image.astype(np.float32).mean(2)
        mean = ndimage.uniform_filter(gray, size=7, mode="reflect")
        variance = ndimage.uniform_filter(gray*gray, size=7, mode="reflect")-mean*mean
        self.high_texture = np.maximum(variance, 0) >= 8**2
        self.outside_distance, self.inside_distance = [], []
        for mask in self.truth:
            if mask.any():
                self.outside_distance.append(ndimage.distance_transform_edt(~mask))
                self.inside_distance.append(ndimage.distance_transform_edt(np.pad(mask, 1))[1:-1, 1:-1])
            else:
                self.outside_distance.append(None); self.inside_distance.append(None)


class Diagnosis:
    def __init__(self, name):
        self.name = name
        self.confusion = np.zeros((16, 16), np.int64)
        self.cross = np.zeros((4, 4), np.int64)
        self.background_fp = np.zeros(4, np.int64)
        self.other_fp = np.zeros(4, np.int64)
        self.fp_distance = np.zeros((4, 4), np.int64)  # <=3,3..16,>16,GT class absent
        self.fn_distance = np.zeros((4, 2), np.int64)  # boundary<=3,interior>3
        self.edge_bg_fp = np.zeros((4, 2), np.int64)
        self.texture_bg_fp = np.zeros((4, 2), np.int64)
        self.edge_exposure = np.zeros(2, np.int64)
        self.texture_exposure = np.zeros(2, np.int64)
        self.gt_overlap = self.pred_overlap = self.pred_overlap_background = 0
        self.pixels = 0

    def add(self, prediction, ctx):
        bits = label_bits(prediction)
        self.pixels += int(bits.size)
        encoded = ctx.bits.astype(np.int16)*16+bits
        self.confusion += np.bincount(encoded.ravel(), minlength=256).reshape(16, 16)
        self.gt_overlap += int((ctx.truth.sum(0) > 1).sum())
        overlaps = prediction.sum(0) > 1
        self.pred_overlap += int(overlaps.sum())
        self.pred_overlap_background += int((overlaps & ctx.background).sum())
        bg_edge = ctx.background & ctx.edge
        bg_inner = ctx.background & ~ctx.edge
        bg_high = ctx.background & ctx.high_texture
        bg_low = ctx.background & ~ctx.high_texture
        self.edge_exposure += [bg_edge.sum(), bg_inner.sum()]
        self.texture_exposure += [bg_high.sum(), bg_low.sum()]
        for c in range(4):
            fp = prediction[c] & ~ctx.truth[c]
            fn = ~prediction[c] & ctx.truth[c]
            self.background_fp[c] += int((prediction[c] & ctx.background).sum())
            self.other_fp[c] += int((fp & ~ctx.background).sum())
            for i in range(4):
                if i != c: self.cross[i, c] += int((fp & ctx.truth[i]).sum())
            self.edge_bg_fp[c] += [(prediction[c] & bg_edge).sum(), (prediction[c] & bg_inner).sum()]
            self.texture_bg_fp[c] += [(prediction[c] & bg_high).sum(), (prediction[c] & bg_low).sum()]
            if ctx.outside_distance[c] is None:
                self.fp_distance[c, 3] += int(fp.sum())
            else:
                distances = ctx.outside_distance[c][fp]
                self.fp_distance[c] += [(distances <= 3).sum(), ((distances > 3) & (distances <= 16)).sum(),
                                        (distances > 16).sum(), 0]
                inside = ctx.inside_distance[c][fn]
                self.fn_distance[c] += [(inside <= 3).sum(), (inside > 3).sum()]

    def result(self, metric):
        for c in range(4):
            row = metric["per_class"][str(c+1)]
            assert self.background_fp[c]+self.other_fp[c] == row["false_positive_pixels"]
            assert self.fp_distance[c].sum() == row["false_positive_pixels"]
            assert self.fn_distance[c].sum() == row["false_negative_pixels"]
        assert self.confusion.sum() == self.pixels
        def rates(values, exposures):
            return [[float(v/n) if n else None for v, n in zip(row, exposures)] for row in values]
        return {"label_set_confusion_16x16": self.confusion.tolist(), "pixel_locations": self.pixels,
                "label_set_axis": {str(i): [c+1 for c in range(4) if i & (1 << c)] for i in range(16)},
                "axis_definition": "rows GT label set; columns predicted label set;0=background. Exclusive set-pair bins, no argmax.",
                "background_false_positive_pixels_by_class": self.background_fp.tolist(),
                "other_defect_false_positive_pixels_by_class": self.other_fp.tolist(),
                "wrong_cross_class_pairs_GT_rows_predicted_columns": self.cross.tolist(),
                "cross_pair_caution": "Requires target predicted class absent. Multiple GT labels may contribute to multiple rows; pair totals are not additive unique FP counts.",
                "fp_distance_to_same_class_GT_le3_3to16_gt16_class_absent": self.fp_distance.tolist(),
                "fn_boundary_le3_and_interior_gt3": self.fn_distance.tolist(),
                "edge16_background_exposure_edge_interior": self.edge_exposure.tolist(),
                "edge16_background_FP_by_class_edge_interior": self.edge_bg_fp.tolist(),
                "edge16_background_FP_rates_by_class_edge_interior": rates(self.edge_bg_fp, self.edge_exposure),
                "texture_background_exposure_high_low": self.texture_exposure.tolist(),
                "texture_background_FP_by_class_high_low": self.texture_bg_fp.tolist(),
                "texture_background_FP_rates_by_class_high_low": rates(self.texture_bg_fp, self.texture_exposure),
                "texture_definition": "7x7 grayscale local std>=8/255; prespecified association proxy, not a causal/semantic texture label",
                "GT_multi_label_pixels": self.gt_overlap, "predicted_multi_label_pixels": self.pred_overlap,
                "predicted_multi_label_pixels_on_GT_background": self.pred_overlap_background,
                "annotation_caution": "Errors are relative to supplied masks. Disagreement does not establish incorrect annotation."}


def evaluate(networks, label, reset_seed=True):
    if reset_seed: seed_all(42)
    models = {name: value.eval() if isinstance(value, torch.nn.Module) else load_model(value, "cuda")[0]
              for name, value in networks.items()}
    policy = {"thresholds": PROTOCOL["thresholds"], "min_area": [0]*4, "min_mean_probability": [0]*4}
    accum = {name: Accumulator({**policy, "name": name}) for name in models}
    diagnostic = {name: Diagnosis(name) for name in models}
    times = {name: [] for name in models}
    fixed = {name: {} for name in models}
    rows = records("val")
    warm, _ = native(rows[0])
    for model in models.values():
        for _ in range(3): predict(model, warm, "patch")
    start = time.perf_counter()
    names = list(models)
    for index, row in enumerate(rows):
        image, truth = native(row)
        ctx = Context(image, truth)
        # Rotate paired execution order; metrics are not affected by order.
        order = names[index % len(names):]+names[:index % len(names)]
        for name in order:
            torch.cuda.synchronize(); tick = time.perf_counter()
            probability = predict(models[name], image, "patch", device="cuda", batch_size=8)
            torch.cuda.synchronize(); times[name].append(time.perf_counter()-tick)
            if probability.shape != (4, 256, 1600) or not np.isfinite(probability).all():
                raise RuntimeError("Invalid native probabilities")
            prediction = probability >= np.asarray(PROTOCOL["thresholds"], np.float32)[:, None, None]
            accum[name].add(prediction, truth, ctx.small)
            diagnostic[name].add(prediction, ctx)
            if row["image_id"] in PROTOCOL["fixed_panel_ids"]:
                tp = (prediction & ctx.truth).sum((1,2), dtype=np.int64)
                fp = (prediction & ~ctx.truth).sum((1,2), dtype=np.int64)
                fn = (~prediction & ctx.truth).sum((1,2), dtype=np.int64)
                fixed[name][row["image_id"]] = {"tp": tp.tolist(), "fp": fp.tolist(), "fn": fn.tolist(),
                                               "background_fp": (prediction & ctx.background[None]).sum((1,2),dtype=np.int64).tolist(),
                                               "ground_truth_empty": not bool(ctx.truth.any())}
        if (index+1) % 100 == 0:
            print(f"{label} validation {index+1}/2514 elapsed={time.perf_counter()-start:.1f}s", flush=True)
    output = {"images": len(rows), "seconds": time.perf_counter()-start, "models": {}}
    for name in models:
        metric = accum[name].result()
        output["models"][name] = {"metrics": metric, "diagnosis": diagnostic[name].result(metric),
                                   "fixed_examples": fixed[name],
                                   "runtime_ms": {"mean": float(np.mean(times[name])*1000), "median": float(np.median(times[name])*1000),
                                                  "p95": float(np.percentile(times[name],95)*1000),
                                                  "scope": "resident model native RGB to probabilities; CPU prep/GPU/transfer/stitch,3 warmups; excludes image/GT loading and diagnostics"}}
    return output


def diagnosis_text(baseline):
    lines = ["DEVELOPMENT-VALIDATION DIAGNOSIS (2514 images; no test tuning)",
             "Multilabel analysis uses a16x16 truth-set/prediction-set table, not a four-way argmax confusion matrix."]
    for name, entry in baseline["models"].items():
        m, d = entry["metrics"], entry["diagnosis"]
        lines.append(f"\n{name}: Dice/IoU {m['macro_dice']:.6f}/{m['macro_iou']:.6f}; normal alarms {m['false_alarm_images']}/{m['normal_images']}")
        for c in range(4):
            r = m["per_class"][str(c+1)]
            lines.append(f"C{c+1}: FP={r['false_positive_pixels']},FN={r['false_negative_pixels']}; FP on annotated background={d['background_false_positive_pixels_by_class'][c]}, wrong-other-class={d['other_defect_false_positive_pixels_by_class'][c]}; FP distances <=3/3..16/>16/no-own-GT={d['fp_distance_to_same_class_GT_le3_3to16_gt16_class_absent'][c]}; FN boundary/interior={d['fn_boundary_le3_and_interior_gt3'][c]}")
            lines.append(f"  Background FP rates edge/interior={d['edge16_background_FP_rates_by_class_edge_interior'][c]}; high/low local texture proxy={d['texture_background_FP_rates_by_class_high_low'][c]}")
        pair = np.asarray(d["wrong_cross_class_pairs_GT_rows_predicted_columns"])
        i,j = np.unravel_index(np.argmax(pair), pair.shape)
        lines.append(f"Largest wrong-class overlap pair: GT C{i+1} -> predicted C{j+1}: {int(pair[i,j])} pixels. Pair counts can overlap across GT labels.")
        lines.append(f"GT multi-label pixels={d['GT_multi_label_pixels']}; predicted multi-label pixels={d['predicted_multi_label_pixels']}, of which background={d['predicted_multi_label_pixels_on_GT_background']}.")
    lines += ["\nBoundary bins use3/16-pixel native Euclidean distance. Image-edge band is16px; exposure-normalized background rates avoid confusing area with risk.",
              "Local std strata measure association with intensity variation only. They do not prove texture causes errors. Overlapping strata are not additive.",
              "All errors are relative to existing annotations. No annotation was changed or assumed wrong."]
    return "\n".join(lines)+"\n"
