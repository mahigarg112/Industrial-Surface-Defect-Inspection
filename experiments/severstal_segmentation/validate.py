"""Evaluate a saved checkpoint and tune per-class thresholds on validation only."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import CLASS_NAMES, SeverstalSegmentationDataset
from model import SmallUNet


def scores(tp, fp, fn):
    dice_den = 2 * tp + fp + fn
    iou_den = tp + fp + fn
    dice = torch.where(dice_den > 0, 2 * tp / dice_den, torch.ones_like(tp))
    iou = torch.where(iou_den > 0, tp / iou_den, torch.ones_like(tp))
    return dice, iou


def package_metrics(tp, fp, fn, thresholds, val_loss, image_count, label):
    dice, iou = scores(tp, fp, fn)
    per_class = {}
    for index, class_name in enumerate(CLASS_NAMES):
        pred_count = int(tp[index] + fp[index])
        actual_count = int(tp[index] + fn[index])
        per_class[str(index + 1)] = {
            "name": class_name,
            "threshold": float(thresholds[index]),
            "dice": float(dice[index]),
            "iou": float(iou[index]),
            "precision": float(tp[index] / (tp[index] + fp[index])) if pred_count else None,
            "recall": float(tp[index] / (tp[index] + fn[index])) if actual_count else None,
            "true_positive_pixels": int(tp[index]),
            "false_positive_pixels": int(fp[index]),
            "missed_defect_pixels": int(fn[index]),
        }
    return {
        "split": "val",
        "images": image_count,
        "evaluation": label,
        "validation_loss_bce_plus_soft_dice": val_loss,
        "macro_dice": float(dice.mean()),
        "macro_iou": float(iou.mean()),
        "per_class": per_class,
        "note": "All results use the held-out validation split. Threshold-tuned metrics were optimized on this same split and may be optimistic.",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path,
                        default=Path("experiments/severstal_segmentation/results/best_model.pt"))
    parser.add_argument("--output-dir", type=Path,
                        help="where to save default and validation-tuned metrics")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent.parent
    checkpoint_path = args.checkpoint if args.checkpoint.is_absolute() else root / args.checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("smoke_test_only"):
        raise ValueError("This is a smoke-test checkpoint, not a trained model; validation is blocked.")
    if checkpoint.get("class_names") != CLASS_NAMES:
        raise ValueError(f"Checkpoint class mapping mismatch: {checkpoint.get('class_names')}")
    if "model_state_dict" not in checkpoint or "epoch" not in checkpoint:
        raise ValueError("Checkpoint must contain model weights and a completed training epoch")

    config = checkpoint["config"]
    device = torch.device("cuda" if config["device"] == "auto" and torch.cuda.is_available() else "cpu")
    dataset_root = root / config["dataset_root"]
    manifest = root / config["prepared_dir"] / "manifest.csv"
    dataset = SeverstalSegmentationDataset(dataset_root / "train_images", manifest, "val",
                                           config["image_width"], config["image_height"])
    loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=False,
                        num_workers=config["num_workers"])
    model = SmallUNet(out_channels=len(CLASS_NAMES), base_channels=config["base_channels"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()

    # Include probabilities below 0.01: the rare-class diagnostic showed that
    # the previous model's channels can remain very weak across validation.
    candidates = [0.00001, 0.000025, 0.00005, 0.0001, 0.00025, 0.0005,
                  0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.075, 0.1,
                  0.15, 0.2, 0.25, 0.3,
                  0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    threshold_counts = {threshold: [torch.zeros(4), torch.zeros(4), torch.zeros(4)]
                        for threshold in candidates}
    truth_total = torch.zeros(4)
    bce_sum = dice_loss_sum = seen = 0.0
    with torch.no_grad():
        for images, masks, _ in loader:
            images, masks = images.to(device), masks.to(device)
            logits = model(images)
            probabilities = torch.sigmoid(logits)
            bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, masks)
            intersection = (probabilities * masks).sum((2, 3))
            denominator = probabilities.sum((2, 3)) + masks.sum((2, 3))
            dice_loss = 1.0 - ((2 * intersection + 1.0) / (denominator + 1.0)).mean()
            count = images.size(0)
            bce_sum += bce.item() * count
            dice_loss_sum += dice_loss.item() * count
            seen += count
            truth = masks >= 0.5
            truth_total += truth.sum((0, 2, 3)).cpu()
            for threshold in candidates:
                predicted = probabilities >= threshold
                tp, fp, fn = threshold_counts[threshold]
                tp += (predicted & truth).sum((0, 2, 3)).cpu()
                fp += (predicted & ~truth).sum((0, 2, 3)).cpu()
                fn += (~predicted & truth).sum((0, 2, 3)).cpu()

    val_loss = 0.5 * (bce_sum / seen) + 0.5 * (dice_loss_sum / seen)
    default_threshold = float(config.get("threshold", 0.5))
    if default_threshold not in threshold_counts:
        raise ValueError(f"Default threshold {default_threshold} is not in the validation sweep")
    default_tp, default_fp, default_fn = threshold_counts[default_threshold]
    default = package_metrics(default_tp, default_fp, default_fn, [default_threshold] * 4,
                              val_loss, int(seen), "default threshold 0.5")

    # Choose one threshold per class by maximum Dice on validation. On ties,
    # prefer the higher threshold to reduce false alarms.
    tuned_thresholds = []
    for class_index in range(4):
        options = []
        for threshold in candidates:
            tp, fp, fn = threshold_counts[threshold]
            denominator = 2 * tp[class_index] + fp[class_index] + fn[class_index]
            value = float(2 * tp[class_index] / denominator) if denominator else 1.0
            options.append((value, threshold))
        best_dice = max(value for value, _ in options)
        tuned_thresholds.append(max(threshold for value, threshold in options if value == best_dice))
    tuned_tp, tuned_fp, tuned_fn = (torch.zeros(4), torch.zeros(4), torch.zeros(4))
    for class_index, threshold in enumerate(tuned_thresholds):
        tp, fp, fn = threshold_counts[threshold]
        tuned_tp[class_index], tuned_fp[class_index], tuned_fn[class_index] = (
            tp[class_index], fp[class_index], fn[class_index])
    tuned = package_metrics(tuned_tp, tuned_fp, tuned_fn, tuned_thresholds, val_loss,
                            int(seen), "per-class thresholds tuned to maximize validation Dice")

    output_dir = args.output_dir if args.output_dir and args.output_dir.is_absolute() else (
        root / args.output_dir if args.output_dir else checkpoint_path.parent)
    output_dir.mkdir(parents=True, exist_ok=True)
    default_path = output_dir / "validation_metrics_default.json"
    tuned_path = output_dir / "validation_metrics_tuned.json"
    default_path.write_text(json.dumps(default, indent=2), encoding="utf-8")
    tuned_path.write_text(json.dumps(tuned, indent=2), encoding="utf-8")
    print(json.dumps({"default": default, "validation_tuned": tuned}, indent=2))
    print(f"Saved default metrics: {default_path.resolve()}")
    print(f"Saved tuned metrics: {tuned_path.resolve()}")


if __name__ == "__main__":
    main()
