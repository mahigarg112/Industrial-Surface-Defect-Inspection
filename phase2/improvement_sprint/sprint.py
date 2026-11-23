"""Validation-only postprocessing and K fine-tuning; outputs stay in this sprint.

Reuses the frozen split, SmallUNet, native tiler, K crop policy and K objective.
Never reads test images/results or writes historical artifacts/clean release.
"""
import argparse
import csv
import ctypes
import json
import os
from pathlib import Path
import shutil
import sys
import time

import numpy as np
from scipy import ndimage
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.pop("SDI_PHASE2_OUTPUT", None)
from phase2.common import CLASSES, OUT, load_model, seed_all, sha, write_json
from phase2.data import TrainDataset, native, records
from phase2.metrics import counts, summarize
from phase2.objectives import balanced_present_dice_loss
from phase2.patches import boxes, predict

HERE = Path(__file__).resolve().parent
LOCK = json.loads((OUT / "final/lock.json").read_text())
K_CONFIG = json.loads((OUT / "K_extended_patch/config.json").read_text())
K_PATH = OUT / "final/model.pt"
CONNECTED = np.ones((3, 3), bool)


def save(path, value):
    path = Path(path).resolve()
    if not path.is_relative_to(HERE):
        raise ValueError("All new outputs must stay in improvement_sprint")
    write_json(path, value)


def policies():
    locked = LOCK["thresholds"]
    base = {"thresholds": locked, "min_area": [0]*4, "min_mean_probability": [0]*4}
    result = [{**base, "name": "K_threshold_05", "thresholds": [0.5]*4},
              {**base, "name": "K_locked"}]
    for area in [16, 64, 256]:
        result.append({**base, "name": f"K_area_{area}", "min_area": [area]*4})
    result.append({**base, "name": "K_area_classwise", "min_area": [16, 16, 64, 64]})
    result.append({**base, "name": "K_area16_component_probability",
                   "min_area": [16]*4,
                   "min_mean_probability": [float(t + .25*(1-t)) for t in locked]})
    return result


def freeze_protocol():
    protocol = {
        "split": "val", "images": 2514, "partition_sha256": LOCK["partition_sha256"],
        "postprocessing": policies(), "connectivity": 8,
        "small_component_definition": "GT area <=1024 native pixels; recovered >=10% means detected",
        "ranking": "highest validation macro Dice among safeguards-passing candidates; first tie wins",
        "safeguards_vs_original_K_locked": {
            "minimum_macro_dice_gain": .002, "maximum_class_pixel_recall_drop": .02,
            "maximum_class2_recall_drop": .01, "minimum_class2_precision_ratio": 1.,
            "maximum_small_object_recall_drop_each_class": .02,
            "maximum_small_pixel_recall_drop_each_class": .02,
            "maximum_normal_image_false_alarm_rate_increase": .005,
            "maximum_localized_positive_image_recall_drop": .01},
        "finetune": {"start": "K epoch9 weights + matching Adam/scaler states",
                     "lr": .0001, "additional_epochs": 3, "batch_size": 8,
                     "num_workers": 2, "seed": 42, "augmentation": None,
                     "objective": K_CONFIG["loss"], "class_weights": K_CONFIG["dice_class_weights"],
                     "crop_policy": K_CONFIG["train_sampling"],
                     "epoch_selection": "maximum full native validation macro Dice using fixed K locked thresholds",
                     "also_report": "threshold0.5", "patience": 2, "useful_gain": .002,
                     "threshold_retuning": False,
                     "validation_loss": "same K objective over ALL 12 native grid patches per validation image, patch-count weighted mean"},
        "limitations": "Development validation reused extensively; no independent generalization claim; old test never evaluated here."
    }
    path = HERE / "protocol.json"
    if path.exists():
        if json.loads(path.read_text()) != protocol:
            raise ValueError("Frozen sprint protocol differs")
    else:
        save(path, protocol)
    return protocol


def filter_mask(probability, thresholds, min_area, min_mean_probability):
    masks = probability >= np.asarray(thresholds, np.float32)[:, None, None]
    for c in range(4):
        if not min_area[c] and not min_mean_probability[c]:
            continue
        labels, number = ndimage.label(masks[c], structure=CONNECTED)
        area = np.bincount(labels.ravel(), minlength=number+1)
        keep = area >= min_area[c]
        if min_mean_probability[c]:
            total = np.bincount(labels.ravel(), weights=probability[c].ravel(), minlength=number+1)
            keep &= total / np.maximum(area, 1) >= min_mean_probability[c]
        keep[0] = False
        masks[c] = keep[labels]
    return masks


class Accumulator:
    def __init__(self, policy):
        self.policy = policy
        self.totals = np.zeros((3, 4), np.int64)
        self.small = np.zeros((4, 4), np.int64)
        self.images = self.normal = self.alarms = self.positive = self.image_alarm = self.localized = 0
        self.class_positive = np.zeros(4, np.int64)
        self.class_localized = np.zeros(4, np.int64)

    def add(self, prediction, truth, small_info):
        # Reuse GT-component foreground indices rather than scanning three full
        # Boolean volumes for every policy. Exact same int64 TP/FP/FN definition.
        positive_pixels = prediction.sum((1, 2), dtype=np.int64)
        tp, gt_pixels = np.zeros(4, np.int64), np.zeros(4, np.int64)
        for c, (positions, component_ids, areas) in enumerate(small_info):
            if not len(areas):
                continue
            recovered = np.bincount(component_ids, weights=prediction[c].ravel()[positions], minlength=len(areas)+1)[1:]
            tp[c] = int(recovered.sum())
            gt_pixels[c] = int(areas.sum())
            select = areas <= 1024
            self.small[c] += np.asarray([select.sum(), ((recovered >= .1*areas) & select).sum(),
                                         areas[select].sum(), recovered[select].sum()], dtype=np.int64)
        current = np.stack([tp, positive_pixels-tp, gt_pixels-tp])
        self.totals += current
        self.images += 1
        active = gt_pixels > 0
        self.class_positive += active
        self.class_localized += active & (tp > 0)
        if active.any():
            self.positive += 1
            self.image_alarm += int(positive_pixels.sum() > 0)
            self.localized += int(tp.sum() > 0)
        else:
            self.normal += 1
            self.alarms += int(positive_pixels.sum() > 0)

    def result(self):
        result = summarize(self.totals)
        result.update({"policy": self.policy, "images": self.images,
                       "normal_images": self.normal, "false_alarm_images": self.alarms,
                       "normal_false_alarm_rate": self.alarms/self.normal if self.normal else None,
                       "positive_images": self.positive,
                       "image_defect_recall_any_alarm": self.image_alarm/self.positive if self.positive else None,
                       "image_defect_recall_localized": self.localized/self.positive if self.positive else None,
                       "class_positive_images": self.class_positive.tolist(),
                       "class_localized_positive_images": self.class_localized.tolist(),
                       "small_defects": {str(c+1): {"objects": int(s[0]), "detected": int(s[1]),
                                                   "object_recall": float(s[1]/s[0]) if s[0] else None,
                                                   "gt_pixels": int(s[2]), "recovered_pixels": int(s[3]),
                                                   "pixel_recall": float(s[3]/s[2]) if s[2] else None}
                                         for c, s in enumerate(self.small)},
                       "partition_sha256": LOCK["partition_sha256"],
                       "metric_scope": "validation native pixel totals per class, then equal-class macro; image recall localized requires at least one correct-class TP pixel"})
        return result


def small_components(truth):
    result = []
    for mask in truth:
        labels, number = ndimage.label(mask, structure=CONNECTED)
        positions = np.flatnonzero(mask.ravel())
        component_ids = labels.ravel()[positions]
        areas = np.bincount(labels.ravel(), minlength=number+1)[1:]
        result.append((positions, component_ids, areas))
    return result


def predict_loss(model, image, truth, weights):
    """Observe the reused predictor's logits; keep BN eval and compute true patch loss."""
    coordinates = boxes(*image.shape[:2])
    offset, total_loss = 0, 0.
    def loss_hook(module, inputs, logits):
        nonlocal offset, total_loss
        selected = coordinates[offset:offset+len(logits)]
        target = np.stack([truth[:, y:b, x:r] for y, x, b, r in selected])
        target = torch.from_numpy(target).to(device=logits.device, dtype=torch.float32)
        total_loss += float(balanced_present_dice_loss(logits, target, weights))*len(logits)
        offset += len(logits)
    handle = model.register_forward_hook(loss_hook)
    try:
        probability = predict(model, image, "patch", device="cuda", batch_size=8,
                              stride=LOCK["stride"], kind=LOCK["blend"])
    finally:
        handle.remove()
    assert offset == len(coordinates)
    return probability, total_loss, offset


def evaluate_validation(model, candidate_policies, label):
    model.eval()
    accumulators = [Accumulator(policy) for policy in candidate_policies]
    weights = torch.tensor(K_CONFIG["dice_class_weights"], device="cuda", dtype=torch.float32)
    loss_sum, patches = 0., 0
    started = time.perf_counter()
    for index, row in enumerate(records("val")):
        image, truth = native(row)
        probability, loss, patch_count = predict_loss(model, image, truth, weights)
        if probability.shape != (4, 256, 1600) or not np.isfinite(probability).all():
            raise RuntimeError("Invalid reconstructed validation probabilities")
        loss_sum += loss
        patches += patch_count
        small = small_components(truth)
        # Reuse connected labels within each unique threshold set for all filters.
        shared = {}
        for accumulator in accumulators:
            policy = accumulator.policy
            key = tuple(policy["thresholds"])
            if key not in shared:
                base = probability >= np.asarray(key, np.float32)[:, None, None]
                shared[key] = (base, [])
            base, components = shared[key]
            if not any(policy["min_area"]) and not any(policy["min_mean_probability"]):
                masks = base
            else:
                if not components:
                    for c in range(4):
                        labels, number = ndimage.label(base[c], structure=CONNECTED)
                        area = np.bincount(labels.ravel(), minlength=number+1)
                        probsum = np.bincount(labels.ravel(), weights=probability[c].ravel(), minlength=number+1)
                        components.append((labels, area, probsum))
                masks = np.empty_like(base)
                for c, (labels, area, probsum) in enumerate(components):
                    keep = (area >= policy["min_area"][c]) & (probsum/np.maximum(area, 1) >= policy["min_mean_probability"][c])
                    keep[0] = False
                    masks[c] = keep[labels]
            accumulator.add(masks, truth, small)
        if (index+1) % 100 == 0:
            print(f"{label} validation {index+1}/2514 elapsed={time.perf_counter()-started:.1f}s", flush=True)
    return {"results": {a.policy["name"]: a.result() for a in accumulators},
            "validation_loss": loss_sum/patches, "validation_patch_count": patches,
            "seconds": time.perf_counter()-started}


def safeguards(candidate, reference, require_gain=True):
    failures = []
    if require_gain and candidate["macro_dice"] < reference["macro_dice"]+.002:
        failures.append("macro Dice gain below0.002")
    for c in range(1, 5):
        cur, ref = candidate["per_class"][str(c)], reference["per_class"][str(c)]
        tolerance = .01 if c == 2 else .02
        if cur["recall"] < ref["recall"]-tolerance:
            failures.append(f"class{c} pixel recall loss")
        a, b = candidate["small_defects"][str(c)], reference["small_defects"][str(c)]
        for metric in ["object_recall", "pixel_recall"]:
            if b[metric] is not None and a[metric] < b[metric]-.02:
                failures.append(f"class{c} small {metric} loss")
    if (candidate["per_class"]["2"]["precision"] is None or
            candidate["per_class"]["2"]["precision"] < reference["per_class"]["2"]["precision"]):
        failures.append("class2 precision loss")
    if candidate["normal_false_alarm_rate"] > reference["normal_false_alarm_rate"]+.005:
        failures.append("normal-image false-alarm rate increase")
    if candidate["image_defect_recall_localized"] < reference["image_defect_recall_localized"]-.01:
        failures.append("localized positive-image recall loss")
    return {"passed": not failures, "reasons": failures}


def postprocess():
    freeze_protocol()
    if (HERE / "postprocessing/results.json").exists():
        raise FileExistsError("Preserve completed postprocessing evidence")
    seed_all(42)
    model, _ = load_model(K_PATH, "cuda")
    torch.cuda.reset_peak_memory_stats()
    result = evaluate_validation(model, policies(), "original_K")
    baseline = result["results"]["K_locked"]
    acceptable = []
    for name, metric in result["results"].items():
        metric["safeguards"] = safeguards(metric, baseline)
        if name not in ["K_threshold_05", "K_locked"] and metric["safeguards"]["passed"]:
            acceptable.append(name)
    result["selected_postprocessing"] = max(acceptable, key=lambda n: result["results"][n]["macro_dice"]) if acceptable else None
    result["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
    result["checkpoint_sha256"] = sha(K_PATH)
    save(HERE / "postprocessing/results.json", result)
    print(json.dumps({"postprocessing_complete": True, "selected": result["selected_postprocessing"],
                      "scores": {n: m["macro_dice"] for n, m in result["results"].items()}}, indent=2), flush=True)


def train(name="L_low_lr", epochs=3):
    protocol = freeze_protocol()
    if name not in ["L_low_lr", "M_class2_quota"]:
        raise ValueError("Unknown controlled trial")
    if name == "M_class2_quota":
        if json.loads((HERE / "L_low_lr/run_info.json").read_text())["status"] != "completed":
            raise ValueError("Complete Stages1-3 before the optional sampling pilot")
        if epochs != 1:
            raise ValueError("Class2 pilot is prespecified as one matched epoch only")
    directory = HERE / name
    if directory.exists():
        raise FileExistsError("Use a new run; historical experiments are never overwritten")
    directory.mkdir()
    baseline = json.loads((HERE / "postprocessing/results.json").read_text())["results"]["K_locked"]
    seed_all(42)
    state = torch.load(OUT / "K_extended_patch/last_training_state.pt", map_location="cpu", weights_only=True)
    model, checkpoint = load_model(K_PATH, "cuda")
    assert state["epoch"] == checkpoint["epoch"] == 9
    assert all(torch.equal(v.cpu(), state["model_state_dict"][k]) for k, v in model.state_dict().items())
    optimizer = torch.optim.Adam(model.parameters(), lr=.0001)
    optimizer.load_state_dict(state["optimizer"])
    for group in optimizer.param_groups:
        group["lr"] = .0001
    scaler = torch.amp.GradScaler("cuda")
    scaler.load_state_dict(state["scaler"])
    torch.set_rng_state(state["torch_rng"])
    torch.cuda.set_rng_state_all(state["cuda_rng"])
    del state
    weights = torch.tensor(K_CONFIG["dice_class_weights"], device="cuda", dtype=torch.float32)
    config = {**K_CONFIG, "learning_rate": .0001, "epochs": 9+epochs,
              "sprint_additional_epoch_cap": epochs,
              "initial_checkpoint_sha256": sha(K_PATH), "continuation_parent": "K_extended_patch",
              "initial_training_state_sha256": sha(OUT / "K_extended_patch/last_training_state.pt"),
              "continuation_state_sha256": sha(OUT / "K_extended_patch/last_training_state.pt"),
              "continuation_start_epoch": 9, "optimizer_moments_restored": True,
              "scaler_restored": True, "description": "fine-tuning continuation with deliberately reduced LR; not an unchanged exact continuation",
              "selection": protocol["finetune"]["epoch_selection"],
              "fixed_thresholds": LOCK["thresholds"], "augmentation": None,
              "training_source_sha256": sha(Path(__file__)), "protocol_sha256": sha(HERE / "protocol.json")}
    if name == "M_class2_quota":
        from phase2.improvement_sprint.rare_sampling import Class2QuotaDataset
        config["train_sampling"] = "Replace354 pure-Class3 slots with2 extra C2 draws/source; preserve all normal/C1/C4 sources; cap3; jitter extra native crops around training C2 pixels"
        config["matched_reference"] = "L_low_lr epoch10 validation; identical K start, optimizer/scaler/RNG, LR, loss, batch and1131 attempts"
        config["sampling_source_sha256"] = sha(Path(__file__).parent / "rare_sampling.py")
        config["class2_protocol_sha256"] = sha(HERE / "class2_protocol.json")
    save(directory / "config.json", config)
    rows = records("train")
    best, useful_best, bad = -1., baseline["macro_dice"], 0
    history, attempted, skipped = [], 0, 0
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(9, 9+epochs):
        model.train()  # Continue BN running mean/variance; no resetting/freezing.
        dataset = (Class2QuotaDataset(rows, epoch=epoch) if name == "M_class2_quota" else
                   TrainDataset(rows, "patch", epoch=epoch, foreground_crops=True))
        if name == "M_class2_quota":
            save(directory / f"epoch_{epoch+1}_sampling_manifest.json", dataset.summary)
        loader = DataLoader(dataset,
                            batch_size=8, shuffle=True, generator=torch.Generator().manual_seed(42+epoch),
                            num_workers=2, pin_memory=True)
        loss_sum, samples = 0., 0
        exposure_pixels, exposure_pairs = np.zeros(4, np.int64), np.zeros(4, np.int64)
        train_start = time.perf_counter()
        for index, (images, masks) in enumerate(loader):
            exposure_pixels += masks.sum((0, 2, 3)).numpy().astype(np.int64)
            exposure_pairs += masks.flatten(2).any(2).sum(0).numpy().astype(np.int64)
            images, masks = images.cuda(non_blocking=True), masks.cuda(non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16):
                logits = model(images)
                loss = balanced_present_dice_loss(logits, masks, weights)
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite training loss")
            scaler.scale(loss).backward()
            old_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            skipped += int(scaler.get_scale() < old_scale)
            attempted += 1
            loss_sum += float(loss.detach())*len(images)
            samples += len(images)
            if (index+1) % 100 == 0:
                print(f"{name} epoch{epoch+1} train {index+1}/{len(loader)} loss={loss_sum/samples:.5f}", flush=True)
        training_seconds = time.perf_counter()-train_start
        current_policies = [{**p, "name": f"{name}_{'05' if p['name']=='K_threshold_05' else 'locked'}"}
                            for p in policies()[:2]]
        evaluation = evaluate_validation(model, current_policies, f"{name}_epoch{epoch+1}")
        score = evaluation["results"][name+"_locked"]
        score05 = evaluation["results"][name+"_05"]
        score["safeguards"] = safeguards(score, baseline)
        save(directory / f"epoch_{epoch+1}_validation.json", evaluation)
        save(directory / f"epoch_{epoch+1}_target_exposure.json", {"samples": samples,
             "positive_samples_by_class": exposure_pairs.tolist(), "positive_pixels_by_class": exposure_pixels.tolist()})
        entry = {"epoch": epoch+1, "train_loss": loss_sum/samples,
                 "validation_loss_all_native_patches": evaluation["validation_loss"],
                 "val_macro_dice": score["macro_dice"], "val_macro_iou": score["macro_iou"],
                 "val_dice_05": score05["macro_dice"], "learning_rate": optimizer.param_groups[0]["lr"],
                 "normal_false_alarm_rate": score["normal_false_alarm_rate"],
                 "class2_precision": score["per_class"]["2"]["precision"],
                 "class2_recall": score["per_class"]["2"]["recall"],
                 "train_seconds": training_seconds, "validation_seconds": evaluation["seconds"],
                 "attempted_steps": attempted, "skipped_steps": skipped}
        entry.update({f"dice_class_{c}": score["per_class"][str(c)]["dice"] for c in range(1, 5)})
        history.append(entry)
        with (directory / "history.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(entry)); writer.writeheader(); writer.writerows(history)
        if score["macro_dice"] > best:
            best = score["macro_dice"]
            torch.save({"model_state_dict": model.state_dict(), "config": config,
                        "class_names": CLASSES, "epoch": epoch+1, "val_macro_dice": best,
                        "validation": score, "smoke_test_only": False}, directory / "best_model.pt")
            save(directory / "best_validation.json", evaluation)
        if score["macro_dice"] >= useful_best+.002:
            useful_best, bad = score["macro_dice"], 0
        else:
            bad += 1
        torch.save({"model_state_dict": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict(), "epoch": epoch+1,
                    "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all(),
                    "config": config}, directory / "last_training_state.pt")
        info = {"status": "running", "epochs_completed": len(history), "best_validation_macro_dice": best,
                "attempted_steps": attempted, "successful_steps": attempted-skipped, "skipped_steps": skipped,
                "duration_seconds": time.perf_counter()-started,
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                "checkpoint_sha256": sha(directory / "best_model.pt")}
        save(directory / "run_info.json", info)
        print(json.dumps(entry, indent=2), flush=True)
        if bad >= 2:
            print("Stopping: two epochs without useful >=0.002 validation gain", flush=True)
            break
    info["status"] = "completed"
    save(directory / "run_info.json", info)


def main():
    global HERE
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["protocol", "postprocess", "train", "rare"])
    parser.add_argument("--output-dir", type=Path, default=HERE,
                        help="Fresh directory inside improvement_sprint for an independent reproduction")
    args = parser.parse_args()
    base = Path(__file__).resolve().parent
    HERE = args.output_dir.resolve()
    if not HERE.is_relative_to(base):
        raise ValueError("Experiment outputs must stay inside improvement_sprint")
    HERE.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; do not run expensive CPU training")
    if sha(K_PATH) != LOCK["checkpoint_sha256"] or sha(OUT / "partition.csv") != LOCK["partition_sha256"]:
        raise ValueError("Locked checkpoint/partition changed")
    if shutil.disk_usage(ROOT).free < 512*1024**2:
        raise RuntimeError("Insufficient disk headroom")
    # Keep only this bounded experiment process awake; do not change power plans.
    awake = bool(ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)) if os.name == "nt" else False
    try:
        if args.stage == "protocol": freeze_protocol()
        elif args.stage == "postprocess": postprocess()
        elif args.stage == "train": train()
        elif args.stage == "rare": train("M_class2_quota", epochs=1)
    finally:
        if awake:
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


if __name__ == "__main__":
    main()
