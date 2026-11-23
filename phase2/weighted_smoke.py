"""Eight real native-patch GPU steps for the separate weighted objective."""
import json
import argparse

import numpy as np
import torch
from PIL import Image

from .common import OUT, SmallUNet, seed_all, sha, write_json
from .data import native, records, TrainDataset
from .objectives import weighted_loss,present_dice_loss,balanced_present_dice_loss
from .patches import boxes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--objective",choices=["weighted","present","balanced"],default="weighted")
    parser.add_argument("--mode",choices=["patch","whole"],default="patch")
    parser.add_argument("--foreground-sampler",action="store_true")
    parser.add_argument("--dice-priors",type=str)
    args = parser.parse_args()
    if args.foreground_sampler and (args.objective not in ["present","balanced"] or args.mode != "patch"):
        raise ValueError("Foreground preflight uses the present-Dice patch model")
    if args.objective == "balanced" and (not args.foreground_sampler or not args.dice_priors):
        raise ValueError("Balanced-Dice preflight requires foreground sampler and frozen --dice-priors")
    seed_all(42)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    path = OUT/"training_class_priors.json"
    prior = json.loads(path.read_text()) if args.objective == "weighted" else None
    weights = torch.tensor(prior["positive_weights"],device="cuda",dtype=torch.float32) if prior else None
    dice_weights = None
    if args.objective == "balanced":
        from pathlib import Path
        dice_path = Path(args.dice_priors)
        dice_prior = json.loads(dice_path.read_text())
        if dice_prior["partition_sha256"] != sha(OUT/"partition.csv"):
            raise ValueError("Dice prior partition mismatch")
        dice_weights = torch.tensor(dice_prior["dice_class_weights"],device="cuda",dtype=torch.float32)
    chosen = []
    for c in range(4):
        row = next(r for r in records("train") if r["rle"].split("|")[c])
        image,mask = native(row)
        y,x,b,r = next(box for box in boxes(256,1600) if mask[c,box[0]:box[2],box[1]:box[3]].any())
        chosen.append((image[y:b,x:r],mask[:,y:b,x:r]) if args.mode == "patch" else (image,mask))
    if args.objective == "weighted":
        chosen = chosen*2
    else:
        negatives = [r for r in records("train") if not any(r["rle"].split("|"))][:4]
        for row in negatives:
            image,mask = native(row)
            chosen.append((image[:,:256],mask[:,:,:256]) if args.mode == "patch" else (image,mask))
    if args.mode == "whole":
        chosen = [(np.asarray(Image.fromarray(image).resize((512,128),Image.Resampling.BILINEAR)),
                   np.stack([np.asarray(Image.fromarray(m).resize((512,128),Image.Resampling.NEAREST)) for m in mask])) for image,mask in chosen]
    if args.foreground_sampler:
        rows = records("train")
        selected = [next(row for row in rows if [i for i,rle in enumerate(row["rle"].split("|")) if rle] == [c]) for c in range(4)]
        selected += [row for row in rows if not any(row["rle"].split("|"))][:4]
        dataset = TrainDataset(selected,"patch",foreground_crops=True)
        pairs = [dataset[i] for i in range(8)]
        images = torch.stack([p[0] for p in pairs]).cuda()
        targets = torch.stack([p[1] for p in pairs]).cuda()
    else:
        images = torch.from_numpy(np.stack([i.transpose(2,0,1) for i,m in chosen]).copy()).float().cuda()/255
        targets = torch.from_numpy(np.stack([m for i,m in chosen]).copy()).float().cuda()
    assert (targets.sum((0,2,3))>0).all()
    model = SmallUNet().cuda().train()
    before = model.output.weight.detach().clone()
    optimizer = torch.optim.Adam(model.parameters(),lr=0.001)
    scaler = torch.amp.GradScaler("cuda")
    losses,attempts,successful,scales = [],0,0,[]
    while successful < 8 and attempts < 24:
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda",dtype=torch.float16):
            logits = model(images)
            loss = (weighted_loss(logits,targets,weights) if prior else
                    balanced_present_dice_loss(logits,targets,dice_weights) if dice_weights is not None else
                    present_dice_loss(logits,targets))
        assert torch.isfinite(loss)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        finite = all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
        old_scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        attempts += 1
        successful += int(finite)
        scales.append({"before":old_scale,"after":scaler.get_scale(),"finite_gradients":finite})
        losses.append(float(loss.detach()))
    assert successful == 8, "AMP did not recover finite gradients within the short preflight budget"
    assert not torch.equal(before,model.output.weight)
    result = {"passed":True,"objective":args.objective,"mode":args.mode,"foreground_sampler":args.foreground_sampler,"prior_sha256":sha(path) if prior else None,"partition_sha256":sha(OUT/"partition.csv"),
              "successful_steps":successful,"attempted_steps":attempts,"skipped_steps":attempts-successful,
              "scales":scales,"all_four_classes_in_targets":True,"losses":losses,"batch_size":8,"amp":True,
              "note":"Real positive crops exercise all four channels; present-objective mode also includes negative crops. Mechanical preflight, not an accuracy evaluation."}
    filename = "weighted_preflight.json" if prior else ("present_preflight.json" if args.mode == "patch" else "present_whole_preflight.json")
    if args.foreground_sampler:
        filename = "foreground_preflight.json"
    if args.objective == "balanced":
        filename = "balanced_dice_preflight.json"
        result["dice_prior_sha256"] = sha(dice_path)
        result["dice_class_weights"] = dice_prior["dice_class_weights"]
    write_json(OUT/filename,result)
    print(json.dumps(result,indent=2),flush=True)


if __name__ == "__main__":
    main()
