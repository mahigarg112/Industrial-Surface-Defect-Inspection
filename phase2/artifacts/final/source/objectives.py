"""A separately evaluated class-aware BCE change; training annotations only."""
import json
import argparse
from pathlib import Path

import numpy as np
import torch

from .common import OUT, decode_rle, sha, write_json
from .data import records


def weighted_loss(logits,target,positive_weights):
    logits,target = logits.float(),target.float()
    weights = positive_weights.to(device=logits.device,dtype=torch.float32).reshape(4,1,1)
    bce = torch.nn.functional.binary_cross_entropy_with_logits(logits,target,pos_weight=weights)
    probability = logits.sigmoid()
    dice = (2*(probability*target).sum((2,3))+1) / (probability.sum((2,3))+target.sum((2,3))+1)
    return 0.5*bce + 0.5*(1-dice.mean())


def present_dice_loss(logits,target):
    """Keep unweighted BCE; omit GT-empty pairs from the per-image/class Dice term."""
    logits,target = logits.float(),target.float()
    bce = torch.nn.functional.binary_cross_entropy_with_logits(logits,target)
    probability = logits.sigmoid()
    target_area = target.sum((2,3))
    dice = (2*(probability*target).sum((2,3))+1) / (probability.sum((2,3))+target_area+1)
    present = (target_area>0).float()
    dice_loss = ((1-dice)*present).sum()/present.sum().clamp_min(1)
    return 0.5*bce + 0.5*dice_loss


def balanced_present_dice_loss(logits,target,class_weights):
    """Inverse positive-pair-frequency Dice; unchanged BCE and no empty-pair reward."""
    logits,target = logits.float(),target.float()
    bce = torch.nn.functional.binary_cross_entropy_with_logits(logits,target)
    probability = logits.sigmoid()
    area = target.sum((2,3))
    dice = (2*(probability*target).sum((2,3))+1)/(probability.sum((2,3))+area+1)
    weights = class_weights.to(device=logits.device,dtype=torch.float32)[None,:]
    dice_loss = ((1-dice)*(area>0).float()*weights).mean()
    return 0.5*bce + 0.5*dice_loss


def prepare_pair_priors(directory):
    path = OUT/"dice_pair_priors.json"
    if path.exists():
        raise FileExistsError("Dice pair priors already frozen")
    info = json.loads((directory/"run_info.json").read_text())
    if info["status"] != "completed":
        raise ValueError("Finish the reference training crop schedule before calibrating priors")
    files = sorted(directory.glob("epoch_*_target_exposure.json"))
    values = [json.loads(file.read_text()) for file in files]
    count = sum(value["samples"] for value in values)
    positives = np.array([value["positive_samples_by_class"] for value in values],np.int64).sum(0)
    if len(files) != info["epochs_completed"] or np.any(positives<=0):
        raise ValueError("Incomplete class exposure logs")
    result = {"source_split":"train","source_run":directory.name,"training_draws":count,
              "calibration_epochs":len(files),"positive_pairs_by_class":positives.tolist(),
              "dice_class_weights":(count/positives).tolist(),"partition_sha256":sha(OUT/"partition.csv"),
              "source_config_sha256":sha(directory/"config.json"),
              "policy":"training draw count / observed positive image-class pairs, per class; mean over all B*C pairs; BCE unchanged; GT-empty Dice terms zero",
              "note":"Only deterministic training target exposure is used. No learned predictions, validation labels or test labels enter these weights."}
    write_json(path,result)
    print(json.dumps(result,indent=2),flush=True)


def prepare():
    path = OUT/"training_class_priors.json"
    if path.exists():
        raise FileExistsError("Training priors already frozen")
    rows = records("train")
    pixels = np.zeros(4,np.int64)
    for row in rows:
        for c,rle in enumerate(row["rle"].split("|")):
            if rle:
                pixels[c] += int(decode_rle(rle).sum())
    total = len(rows)*256*1600
    weights = np.minimum((total-pixels)/pixels,1000)
    result = {"source_split":"train","images":len(rows),"positive_pixels":pixels.tolist(),
              "total_pixels_per_class":total,"positive_weights":weights.tolist(),
              "policy":"min(negative_pixels / positive_pixels, 1000), fixed before training; all counts from native training masks only",
              "partition_sha256":sha(OUT/"partition.csv"),
              "rationale":"Historical Phase I and new B validation collapse to zero rare-class recall. Only BCE positive weights change; soft Dice, architecture, sampling, seed and optimizer stay fixed."}
    write_json(path,result)
    print(json.dumps(result,indent=2),flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--from-exposure",type=Path)
    args = parser.parse_args()
    prepare_pair_priors(args.from_exposure) if args.from_exposure else prepare()
