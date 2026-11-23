"""Create an image-level train/validation split and compact RLE manifest."""

import argparse
import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

from dataset import CLASS_NAMES, read_annotations


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare(dataset_root: Path, output_dir: Path, seed: int = 42,
            validation_fraction: float = 0.2) -> dict:
    dataset_root, output_dir = Path(dataset_root), Path(output_dir)
    image_dir = dataset_root / "train_images"
    csv_path = dataset_root / "train.csv"
    if not csv_path.is_file() or not image_dir.is_dir():
        raise FileNotFoundError(f"Expected train.csv and train_images/ under {dataset_root}")
    annotations = read_annotations(csv_path)
    image_paths = sorted(image_dir.glob("*.jpg"))
    if not image_paths:
        raise FileNotFoundError(f"No .jpg images found under {image_dir}")
    names = {path.name for path in image_paths}
    unknown_ids = set(annotations) - names
    if unknown_ids:
        raise ValueError(f"{len(unknown_ids)} annotation IDs have no image, e.g. {sorted(unknown_ids)[:3]}")

    # Hashes group byte-identical files so an exact duplicate cannot cross splits.
    groups = defaultdict(list)
    for path in image_paths:
        with Image.open(path) as image:
            if image.size != (1600, 256):
                raise ValueError(f"Expected 1600x256 image, got {image.size} for {path.name}")
            if image.mode not in ("RGB", "L"):
                raise ValueError(f"Unexpected image mode {image.mode!r} for {path.name}")
        groups[sha256(path)].append(path.name)
    rng = random.Random(seed)
    buckets = defaultdict(list)
    for digest, image_ids in groups.items():
        signatures = {tuple(bool(rle) for rle in annotations.get(name, [""] * 4)) for name in image_ids}
        if len(signatures) != 1:
            raise ValueError(f"Byte-identical images have inconsistent labels: {image_ids}")
        signature = next(iter(signatures))
        buckets[signature].append((digest, sorted(image_ids)))

    assignments = {}
    for signature, bucket in buckets.items():
        rng.shuffle(bucket)
        # Round while retaining at least one train sample when the bucket has multiple groups.
        val_count = round(len(bucket) * validation_fraction)
        if len(bucket) > 1:
            val_count = min(max(val_count, 1), len(bucket) - 1)
        else:
            val_count = 0
        for index, (_, image_ids) in enumerate(bucket):
            split = "val" if index < val_count else "train"
            for image_id in image_ids:
                assignments[image_id] = split

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = output_dir / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["image_id", "split", "rle"])
        for image_id in sorted(names):
            rles = annotations.get(image_id, [""] * len(CLASS_NAMES))
            writer.writerow([image_id, assignments[image_id], "|".join(rles)])

    split_counts = Counter(assignments.values())
    class_images = {str(i + 1): sum(bool(annotations.get(name, [""] * 4)[i]) for name in names)
                    for i in range(len(CLASS_NAMES))}
    split_class_images = {split: {str(i + 1): sum(
        assignments[name] == split and bool(annotations.get(name, [""] * 4)[i]) for name in names
    ) for i in range(len(CLASS_NAMES))} for split in ("train", "val")}
    summary = {
        "dataset_root": str(dataset_root.resolve()), "manifest": str(manifest.resolve()),
        "seed": seed, "validation_fraction": validation_fraction,
        "total_images": len(names), "labeled_images": sum(bool(any(annotations.get(n, []))) for n in names),
        "defect_free_images": sum(not any(annotations.get(n, [])) for n in names),
        "split_images": dict(split_counts), "class_positive_images_by_id": class_images,
        "split_class_positive_images_by_id": split_class_images,
        "exact_duplicate_groups": sum(len(group) > 1 for group in groups.values()),
        "exact_duplicate_extra_files": sum(len(group) - 1 for group in groups.values()),
        "class_names": CLASS_NAMES
    }
    (output_dir / "preparation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=Path("data/severstal-steel-defect-detection"))
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/severstal_segmentation/prepared"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    args = parser.parse_args()
    print(json.dumps(prepare(args.dataset_root, args.output_dir, args.seed, args.validation_fraction), indent=2))


if __name__ == "__main__":
    main()
