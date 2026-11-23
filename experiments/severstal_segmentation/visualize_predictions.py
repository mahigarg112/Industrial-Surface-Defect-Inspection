"""Save validation images, masks, and overlays for a completed trained checkpoint.

Uses Pillow rather than matplotlib so the script needs no plotting dependency.
Smoke-test checkpoints are deliberately rejected to avoid misleading examples.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader

from dataset import CLASS_NAMES, SeverstalSegmentationDataset
from model import SmallUNet


PALETTE = np.asarray([
    [255, 55, 55],   # class 1: red
    [55, 220, 80],   # class 2: green
    [55, 130, 255],  # class 3: blue
    [255, 220, 45],  # class 4: yellow
], dtype=np.float32)


def colorize(mask_stack: np.ndarray) -> Image.Image:
    """Combine four binary channels; overlapping active colors are averaged."""
    active = mask_stack.astype(bool)
    weights = active.sum(axis=0)
    rgb = (active.transpose(1, 2, 0).astype(np.float32) @ PALETTE)
    rgb = rgb / np.maximum(weights[..., None], 1)
    rgb[weights == 0] = 0
    return Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8), mode="RGB")


def overlay_masks(image: Image.Image, mask_stack: np.ndarray) -> Image.Image:
    colors = np.asarray(colorize(mask_stack), dtype=np.float32)
    original = np.asarray(image, dtype=np.float32).copy()
    active = mask_stack.astype(bool).any(axis=0)
    original[active] = 0.52 * original[active] + 0.48 * colors[active]
    return Image.fromarray(np.clip(original, 0, 255).astype(np.uint8))


def select_indices(dataset, count: int):
    """Prefer one positive validation image per class, then a defect-free example."""
    selected = []
    for class_index in range(len(CLASS_NAMES)):
        for index, row in enumerate(dataset.records):
            if row["rle"].split("|")[class_index]:
                if index not in selected:
                    selected.append(index)
                break
    for index, row in enumerate(dataset.records):
        if not any(row["rle"].split("|")):
            if index not in selected:
                selected.append(index)
            break
    for index in range(len(dataset)):
        if len(selected) >= count:
            break
        if index not in selected:
            selected.append(index)
    return selected[:count]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path,
                        default=Path("experiments/severstal_segmentation/results/best_model.pt"))
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument("--output-dir", type=Path,
                        default=Path("experiments/severstal_segmentation/results"))
    args = parser.parse_args()
    if args.count < 1:
        raise ValueError("--count must be at least 1")

    root = Path(__file__).resolve().parent.parent.parent
    checkpoint_path = args.checkpoint if args.checkpoint.is_absolute() else root / args.checkpoint
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Trained checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("smoke_test_only"):
        raise ValueError("Refusing to visualize smoke-test weights as model predictions.")
    if checkpoint.get("class_names") != CLASS_NAMES:
        raise ValueError(f"Unexpected class mapping: {checkpoint.get('class_names')}")
    if "model_state_dict" not in checkpoint:
        raise ValueError("Checkpoint is missing model_state_dict")

    config = checkpoint["config"]
    device = torch.device("cuda" if config.get("device") == "auto" and torch.cuda.is_available() else "cpu")
    data_root = root / config["dataset_root"]
    manifest = root / config["prepared_dir"] / "manifest.csv"
    dataset = SeverstalSegmentationDataset(data_root / "train_images", manifest, "val",
                                          config["image_width"], config["image_height"])
    model = SmallUNet(out_channels=4, base_channels=config["base_channels"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()

    indices = select_indices(dataset, min(args.count, len(dataset)))
    loader = DataLoader(torch.utils.data.Subset(dataset, indices), batch_size=1, shuffle=False,
                        num_workers=0)
    predictions = []
    with torch.no_grad():
        for images, _, _ in loader:
            logits = model(images.to(device))
            predictions.append((torch.sigmoid(logits)[0] >= config.get("threshold", 0.5)).cpu().numpy())

    tile_w, tile_h, label_h, gap = config["image_width"], config["image_height"], 26, 12
    panel_names = ["Original", "Ground truth masks", "Predicted masks", "Ground truth overlay", "Prediction overlay"]
    sheet_w = gap + len(panel_names) * (tile_w + gap)
    sheet_h = gap + len(indices) * (tile_h + label_h + gap)
    sheet = Image.new("RGB", (sheet_w, sheet_h), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    x_positions = [gap + i * (tile_w + gap) for i in range(len(panel_names))]
    for x, title in zip(x_positions, panel_names):
        draw.text((x, 4), title, fill="black", font=font)

    records = []
    for row_index, (dataset_index, prediction) in enumerate(zip(indices, predictions)):
        image_tensor, target_tensor, image_id = dataset[dataset_index]
        original = Image.fromarray(np.clip(image_tensor.permute(1, 2, 0).numpy() * 255, 0, 255).astype(np.uint8))
        target = target_tensor.numpy().astype(np.uint8)
        target_color, prediction_color = colorize(target), colorize(prediction.astype(np.uint8))
        gt_overlay = overlay_masks(original, target)
        pred_overlay = overlay_masks(original, prediction.astype(np.uint8))
        y = gap + row_index * (tile_h + label_h + gap) + label_h
        positives = [str(i + 1) for i, rle in enumerate(dataset.records[dataset_index]["rle"].split("|")) if rle]
        draw.text((gap, y - label_h + 4), f"{image_id} | true class IDs: {', '.join(positives) or 'none'}",
                  fill="black", font=font)
        for x, tile in zip(x_positions, [original, target_color, prediction_color, gt_overlay, pred_overlay]):
            sheet.paste(tile, (x, y))
        records.append({"image_id": image_id, "true_class_ids": positives,
                        "predicted_class_ids": [str(i + 1) for i in range(4) if prediction[i].any()]})

    output_dir = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    image_path = output_dir / "validation_predictions.png"
    sheet.save(image_path)
    manifest_path = output_dir / "visualization_manifest.json"
    manifest_path.write_text(json.dumps({"checkpoint": str(checkpoint_path.resolve()),
                                         "split": "val", "examples": records,
                                         "palette": {str(i + 1): PALETTE[i].astype(int).tolist()
                                                     for i in range(4)}}, indent=2), encoding="utf-8")
    print(f"Saved validation visualization: {image_path.resolve()}")
    print(f"Saved selected image IDs: {manifest_path.resolve()}")


if __name__ == "__main__":
    main()
