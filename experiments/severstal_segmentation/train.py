"""Train the Severstal U-Net; supports separate configs and output folders."""

import argparse
import csv
import json
import random
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from dataset import CLASS_NAMES, SeverstalSegmentationDataset
from model import SmallUNet


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def bce_plus_soft_dice(logits, targets):
    """Equal BCE and soft Dice loss; Dice loss is 1 minus soft overlap."""
    bce = nn.functional.binary_cross_entropy_with_logits(logits, targets)
    probabilities = torch.sigmoid(logits)
    dims = (2, 3)
    intersection = (probabilities * targets).sum(dims)
    denominator = probabilities.sum(dims) + targets.sum(dims)
    # Soft Dice = (2 * intersection + smooth) / (sum(probabilities) + sum(targets) + smooth).
    # Smoothing keeps the value defined for masks with no positive pixels.
    soft_dice = (2 * intersection + 1.0) / (denominator + 1.0)
    dice_loss = 1.0 - soft_dice.mean()
    return 0.5 * bce + 0.5 * dice_loss, bce, dice_loss


def pixel_metrics(tp, fp, fn):
    dice_den = 2 * tp + fp + fn
    iou_den = tp + fp + fn
    dice = torch.where(dice_den > 0, 2 * tp / dice_den, torch.ones_like(tp))
    iou = torch.where(iou_den > 0, tp / iou_den, torch.ones_like(tp))
    return dice, iou


def active_classes(record):
    """Return the 0-based classes with a non-empty RLE in one manifest row."""
    return [i for i, rle in enumerate(record["rle"].split("|")) if rle.strip()]


def make_class_aware_plan(records, seed, positive_fraction=0.65,
                          max_appearances=2, rarity_power=0.5):
    """Make one fixed-size epoch, repeating weighted positives at most once."""
    positives, negatives = [], []
    class_counts = np.zeros(len(CLASS_NAMES), dtype=np.int64)
    classes_by_index = {}
    for index, record in enumerate(records):
        labels = active_classes(record)
        classes_by_index[index] = labels
        if labels:
            positives.append(index)
            class_counts[labels] += 1
        else:
            negatives.append(index)

    draw_count = len(records)
    positive_draw_count = round(draw_count * positive_fraction)
    negative_draw_count = draw_count - positive_draw_count
    extra_positive_draws = positive_draw_count - len(positives)
    repeat_capacity = len(positives) * (max_appearances - 1)
    if not 0 <= extra_positive_draws <= repeat_capacity:
        raise ValueError("Positive draw target cannot be met under the configured repetition cap")
    if negative_draw_count > len(negatives):
        raise ValueError("Negative draw target exceeds the number of unique defect-free images")

    rng = np.random.default_rng(seed)
    negative_draws = rng.choice(negatives, size=negative_draw_count, replace=False).tolist()
    class_weights = np.zeros(len(CLASS_NAMES), dtype=np.float64)
    present = class_counts > 0
    class_weights[present] = class_counts[present].astype(np.float64) ** (-rarity_power)
    image_weights = np.asarray([
        np.mean(class_weights[classes_by_index[index]]) for index in positives
    ], dtype=np.float64)
    image_weights /= image_weights.sum()
    repeated_positive_draws = (rng.choice(positives, size=extra_positive_draws, replace=False,
                                           p=image_weights).tolist()
                               if extra_positive_draws else [])
    plan = positives + repeated_positive_draws + negative_draws
    rng.shuffle(plan)

    appearances = Counter(plan)
    sampled_class_counts = np.zeros(len(CLASS_NAMES), dtype=np.int64)
    for index in plan:
        sampled_class_counts[classes_by_index[index]] += 1
    stats = {
        "seed": int(seed),
        "training_draws": len(plan),
        "defect_positive_draws": positive_draw_count,
        "defect_free_draws": negative_draw_count,
        "unique_positive_images": len(positives),
        "unique_defect_free_images_drawn": len(set(negative_draws)),
        "unique_source_images_used": len(appearances),
        "repeat_draws_beyond_first_use": len(plan) - len(appearances),
        "maximum_appearances_of_any_source_image": max(appearances.values()),
        "maximum_appearances_allowed": int(max_appearances),
        "source_positive_image_counts_by_class": {
            str(i + 1): int(class_counts[i]) for i in range(len(CLASS_NAMES))
        },
        "sampled_positive_image_draws_by_class": {
            str(i + 1): int(sampled_class_counts[i]) for i in range(len(CLASS_NAMES))
        },
        "per_class_rarity_weight_n_to_negative_power": {
            str(i + 1): float(class_weights[i]) for i in range(len(CLASS_NAMES))
        },
    }
    if len(plan) != draw_count or stats["maximum_appearances_of_any_source_image"] > max_appearances:
        raise AssertionError("Sampling plan violates the draw count or repetition cap")
    return plan, stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, help="JSON training configuration")
    parser.add_argument("--output-dir", type=Path, help="folder for this run's artifacts")
    parser.add_argument("--preflight-only", action="store_true",
                        help="write and verify the first epoch sampling plan without training")
    args = parser.parse_args()

    experiment_dir = Path(__file__).resolve().parent
    project_root = experiment_dir.parent.parent
    config_path = args.config.resolve() if args.config else experiment_dir / "config.json"
    output_dir = args.output_dir.resolve() if args.output_dir else experiment_dir / "results"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    seed_everything(config["seed"])
    device = torch.device("cuda" if config["device"] == "auto" and torch.cuda.is_available() else "cpu")
    dataset_root = project_root / config["dataset_root"]
    manifest = project_root / config["prepared_dir"] / "manifest.csv"
    train_data = SeverstalSegmentationDataset(dataset_root / "train_images", manifest, "train",
                                               config["image_width"], config["image_height"])
    val_data = SeverstalSegmentationDataset(dataset_root / "train_images", manifest, "val",
                                             config["image_width"], config["image_height"])
    sampling = config.get("sampling")
    train_loader = (None if sampling else DataLoader(
        train_data, batch_size=config["batch_size"], shuffle=True,
        num_workers=config["num_workers"]))
    val_loader = DataLoader(val_data, batch_size=config["batch_size"], shuffle=False,
                            num_workers=config["num_workers"])
    model = SmallUNet(out_channels=len(CLASS_NAMES), base_channels=config["base_channels"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "run_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    if sampling:
        first_plan, first_stats = make_class_aware_plan(
            train_data.records, config["seed"], sampling["positive_fraction"],
            sampling["max_appearances_per_image"], sampling["rarity_power"])
        (output_dir / "sampling_preflight.json").write_text(
            json.dumps(first_stats, indent=2), encoding="utf-8")
        if args.preflight_only:
            print(json.dumps(first_stats, indent=2))
            print("Preflight complete; no optimizer updates were run.")
            return
    elif args.preflight_only:
        raise ValueError("--preflight-only requires a sampling configuration")

    history_path = output_dir / "training_history.csv"
    best_macro_dice = -1.0
    best_epoch = 0
    epochs_without_improvement = 0
    run_started = time.time()
    sampling_history_handle = None
    sampling_history_writer = None
    if sampling:
        sampling_history_handle = (output_dir / "sampling_history.csv").open(
            "w", newline="", encoding="utf-8")
        sampling_history_writer = csv.DictWriter(sampling_history_handle, fieldnames=[
            "epoch", "seed", "training_draws", "defect_positive_draws", "defect_free_draws",
            "unique_positive_images", "unique_defect_free_images_drawn", "unique_source_images_used",
            "repeat_draws_beyond_first_use", "maximum_appearances_of_any_source_image",
            "maximum_appearances_allowed",
            "class_1_image_draws", "class_2_image_draws", "class_3_image_draws", "class_4_image_draws"])
        sampling_history_writer.writeheader()
    columns = ["epoch", "train_loss", "train_bce", "train_dice_loss", "val_loss",
               "val_bce", "val_dice_loss", "val_macro_dice", "val_macro_iou"]
    columns += [f"val_dice_class_{i}" for i in range(1, 5)]
    columns += [f"val_iou_class_{i}" for i in range(1, 5)]

    with history_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for epoch in range(config["epochs"]):
            epoch_started = time.time()
            if sampling:
                epoch_plan, sampling_stats = make_class_aware_plan(
                    train_data.records, config["seed"] + epoch,
                    sampling["positive_fraction"], sampling["max_appearances_per_image"],
                    sampling["rarity_power"])
                epoch_train_loader = DataLoader(
                    Subset(train_data, epoch_plan), batch_size=config["batch_size"], shuffle=False,
                    num_workers=config["num_workers"])
                sample_row = {key: value for key, value in sampling_stats.items()
                              if key != "sampled_positive_image_draws_by_class" and
                              key != "per_class_rarity_weight_n_to_negative_power" and
                              key != "source_positive_image_counts_by_class"}
                sample_row.update({f"class_{i}_image_draws": sampling_stats[
                    "sampled_positive_image_draws_by_class"][str(i)] for i in range(1, 5)})
                sample_row["epoch"] = epoch + 1
                sampling_history_writer.writerow(sample_row)
                sampling_history_handle.flush()
            else:
                epoch_train_loader = train_loader
            model.train()
            train_sums = np.zeros(3, dtype=np.float64)
            train_seen = 0
            for images, masks, _ in epoch_train_loader:
                images, masks = images.to(device), masks.to(device)
                optimizer.zero_grad(set_to_none=True)
                loss, bce, dice_loss = bce_plus_soft_dice(model(images), masks)
                loss.backward()
                optimizer.step()
                count = images.size(0)
                train_sums += np.asarray([loss.item(), bce.item(), dice_loss.item()]) * count
                train_seen += count

            model.eval()
            val_sums = np.zeros(3, dtype=np.float64)
            val_seen = 0
            tp, fp, fn = torch.zeros(4), torch.zeros(4), torch.zeros(4)
            with torch.no_grad():
                for images, masks, _ in val_loader:
                    images, masks = images.to(device), masks.to(device)
                    logits = model(images)
                    loss, bce, dice_loss = bce_plus_soft_dice(logits, masks)
                    count = images.size(0)
                    val_sums += np.asarray([loss.item(), bce.item(), dice_loss.item()]) * count
                    predictions, truth = torch.sigmoid(logits) >= config["threshold"], masks >= 0.5
                    tp += (predictions & truth).sum((0, 2, 3)).cpu()
                    fp += (predictions & ~truth).sum((0, 2, 3)).cpu()
                    fn += (~predictions & truth).sum((0, 2, 3)).cpu()
                    val_seen += count

            class_dice, class_iou = pixel_metrics(tp, fp, fn)
            macro_dice = float(class_dice.mean())
            macro_iou = float(class_iou.mean())
            train_values = train_sums / train_seen
            val_values = val_sums / val_seen
            row = {"epoch": epoch + 1, "train_loss": train_values[0], "train_bce": train_values[1],
                   "train_dice_loss": train_values[2], "val_loss": val_values[0],
                   "val_bce": val_values[1], "val_dice_loss": val_values[2],
                   "val_macro_dice": macro_dice, "val_macro_iou": macro_iou}
            row.update({f"val_dice_class_{i+1}": float(class_dice[i]) for i in range(4)})
            row.update({f"val_iou_class_{i+1}": float(class_iou[i]) for i in range(4)})
            writer.writerow(row)
            handle.flush()

            improved = macro_dice > best_macro_dice
            if improved:
                best_macro_dice = macro_dice
                best_epoch = epoch + 1
                epochs_without_improvement = 0
                torch.save({"model_state_dict": model.state_dict(), "class_names": CLASS_NAMES,
                            "config": config, "epoch": best_epoch, "val_macro_dice": best_macro_dice,
                            "val_macro_iou": macro_iou, "val_loss": float(val_values[0]),
                            "checkpoint_selection": "highest validation macro Dice at default threshold"},
                           output_dir / "best_model.pt")
            else:
                epochs_without_improvement += 1
            print(f"Epoch {epoch+1}/{config['epochs']}: train_loss={train_values[0]:.5f} "
                  f"val_loss={val_values[0]:.5f} val_macro_Dice={macro_dice:.5f} "
                  f"val_macro_IoU={macro_iou:.5f} ({time.time()-epoch_started:.1f}s) "
                  f"{'best checkpoint saved' if improved else 'no Dice improvement'}", flush=True)
            if epochs_without_improvement >= config["early_stopping_patience"]:
                print(f"Early stopping after {epoch+1} epochs; best epoch was {best_epoch}.", flush=True)
                break

    duration = time.time() - run_started
    if sampling_history_handle:
        sampling_history_handle.close()
    run_info = {"status": "completed", "epochs_completed": epoch + 1, "best_epoch": best_epoch,
                "best_validation_macro_dice": best_macro_dice,
                "checkpoint_selection": "highest validation macro Dice at default threshold",
                "early_stopping_patience": config["early_stopping_patience"],
                "training_duration_seconds": duration, "device": str(device),
                "training_images": len(train_data), "validation_images": len(val_data),
                "checkpoint_is_smoke_test": False,
                "sampling_strategy": sampling or "uniform shuffle without replacement",
                "sampling_preflight": "sampling_preflight.json" if sampling else None}
    (output_dir / "training_run_info.json").write_text(json.dumps(run_info, indent=2), encoding="utf-8")
    print(f"Training complete in {duration:.1f}s. Best epoch: {best_epoch}; "
          f"checkpoint: {(output_dir / 'best_model.pt').resolve()}", flush=True)


if __name__ == "__main__":
    main()
