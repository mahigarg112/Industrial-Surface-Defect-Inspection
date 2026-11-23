"""Read-only Phase I verification and a separate duplicate-safe Phase II partition."""
import argparse
import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from .common import ROOT, OUT, LEGACY, IMAGES, sha, write_json, legacy_module


def protected_files():
    # Include every existing project source, configuration, metric, checkpoint and evidence artifact.
    for directory in ROOT.iterdir():
        if directory.is_dir() and directory.name not in {"phase2", ".venv-phase2", "data", ".publish-preparation", ".git"}:
            for p in directory.rglob("*"):
                if p.is_file() and "__pycache__" not in p.parts:
                    yield p
        elif directory.is_file():
            yield directory
    yield ROOT / "data/severstal-steel-defect-detection/train.csv"


def snapshot():
    path = OUT / "phase1_hashes.json"
    if path.exists():
        raise FileExistsError("Preservation snapshot already exists; use --verify")
    write_json(path, {str(p.relative_to(ROOT)): sha(p) for p in protected_files()})


def verify():
    before = json.loads((OUT / "phase1_hashes.json").read_text())
    changes = [p for p, digest in before.items() if not (ROOT / p).exists() or sha(ROOT / p) != digest]
    write_json(OUT / "preservation_verification.json", {"files_checked": len(before), "changed_or_missing": changes})
    if changes:
        raise RuntimeError(changes)
    print(f"Phase I preservation verified: {len(before)} files unchanged", flush=True)


def prepare(seed=42):
    if (OUT / "partition.csv").exists():
        raise FileExistsError("Partition already frozen")
    original = LEGACY / "prepared/manifest.csv"
    with original.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    annotations = legacy_module("dataset").read_annotations(ROOT / "data/severstal-steel-defect-detection/train.csv")
    groups, dimensions, inventory = defaultdict(list), Counter(), []
    # Byte hashes alone do not detect metadata or lossless re-encoding duplicates.
    for i, row in enumerate(rows):
        path = IMAGES / row["image_id"]
        with Image.open(path) as f:
            image = np.asarray(f.convert("RGB"))
        if image.shape != (256, 1600, 3):
            raise ValueError(path)
        dimensions[str(image.shape)] += 1
        expected = annotations.get(row["image_id"], [""] * 4)
        if row["rle"].split("|") != expected:
            raise ValueError(f"Manifest/CSV label mismatch: {row['image_id']}")
        digest = hashlib.sha256(image.tobytes()).hexdigest()
        groups[digest].append(row)
        inventory.append({"image_id": row["image_id"], "file_sha256": sha(path), "pixel_sha256": digest})
        if (i + 1) % 2000 == 0:
            print(f"Audited {i+1}/{len(rows)} native images", flush=True)
    assignments, buckets = {}, defaultdict(list)
    quarantined = []
    for digest, group in sorted(groups.items()):
        if len({r["rle"] for r in group}) != 1:
            raise ValueError("Identical pixels with contradictory annotations")
        splits = {r["split"] for r in group}
        if "val" in splits:
            for r in group:
                assignments[r["image_id"]] = "val" if r["split"] == "val" else "quarantine"
                if r["split"] != "val":
                    quarantined.append(r["image_id"])
        else:
            signature = tuple(bool(x) for x in group[0]["rle"].split("|"))
            buckets[signature].append((digest, group))
    rng = random.Random(seed)
    for signature, bucket in sorted(buckets.items()):
        rng.shuffle(bucket)
        n_test = round(len(bucket) * 0.1)
        n_test = min(max(n_test, 1), len(bucket)-1) if len(bucket) > 1 else 0
        for i, (_, group) in enumerate(bucket):
            for row in group:
                assignments[row["image_id"]] = "test" if i < n_test else "train"
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "partition.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image_id", "split", "original_split", "rle"])
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, "split": assignments[row["image_id"]], "original_split": row["split"]})
    for group in groups.values():
        active = {assignments[r["image_id"]] for r in group} - {"quarantine"}
        if len(active) != 1:
            raise AssertionError("Duplicate group crosses partitions")
    summary = {
        "seed": seed, "test_fraction_of_original_training_groups": 0.1,
        "original_manifest_sha256": sha(original), "partition_sha256": sha(OUT / "partition.csv"),
        "original_counts": dict(Counter(r["split"] for r in rows)),
        "phase2_counts": dict(Counter(assignments.values())),
        "class_positive_images": {s: [sum(assignments[r["image_id"]] == s and bool(r["rle"].split("|")[c]) for r in rows) for c in range(4)] for s in ["train", "val", "test"]},
        "native_dimensions": dict(dimensions), "duplicate_pixel_groups": sum(len(g)>1 for g in groups.values()),
        "quarantined_original_training_images": quarantined,
        "phase1_original_best_validation": {"checkpoint": "experiments/severstal_segmentation/dice_loss_v1/best_model.pt", "macro_dice": 0.11825690418481827, "macro_iou": 0.07744534313678741, "threshold": 0.5, "evaluation_size": [128,512]},
        "legacy_test_contamination": "Phase I models saw the original training partition, including the newly reserved test images. Their test scores cannot establish held-out generalization.",
        "duplicate_scope": "Exact RGB pixel identity and file SHA256; perceptual/near duplicates and unknown steel-coil group identity are not verified.",
        "patch_extension_audit": "No tiling/reconstruction implementation found in existing Python/model documentation search; added isolated phase2 implementation.",
    }
    write_json(OUT / "dataset_audit.json", summary)
    write_json(OUT / "image_hashes.json", inventory)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.verify:
        verify()
    else:
        snapshot()
        prepare()
