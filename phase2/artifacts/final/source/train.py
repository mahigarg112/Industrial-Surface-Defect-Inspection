"""Matched-budget GPU whole/patch training, gated on successful preflight."""
import argparse
import csv
import json
import time
import shutil
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .common import CLASSES, OUT, SmallUNet, seed_all, sha, write_json
from .data import TrainDataset, native, records
from .metrics import counts, summarize
from .patches import predict
from .smoke import loss_fn


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["whole","patch"], required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--epochs",type=int,default=3)
    p.add_argument("--positive-weights",type=Path,help="separate class-weighted BCE experiment; requires weighted GPU preflight")
    p.add_argument("--present-dice",action="store_true",help="separate objective ablation: Dice only for GT-positive image/class pairs")
    p.add_argument("--foreground-crops",action="store_true",help="separate sampling ablation: rarest-present-class-positive native grid crops")
    p.add_argument("--dice-class-weights",type=Path,help="separate class-balanced positive-Dice ablation using frozen training pair priors")
    p.add_argument("--resume-run",type=Path,help="extend a completed Phase II run into a fresh directory; only total epoch budget may change")
    args = p.parse_args()
    gate = json.loads((OUT/"smoke_results.json").read_text())
    if not gate["passed"] or gate["partition_sha256"] != sha(OUT/"partition.csv"):
        raise ValueError("Preflight failed or partition changed")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    prior = None
    positive_weights = None
    dice_class_weights = None
    dice_prior = None
    if args.dice_class_weights:
        if not args.present_dice or not args.foreground_crops or args.positive_weights:
            raise ValueError("Balanced-Dice experiment retains the H foreground-crop pipeline and unweighted BCE")
        dice_prior = json.loads(args.dice_class_weights.read_text())
        dice_gate = json.loads((OUT/"balanced_dice_preflight.json").read_text())
        if not dice_gate["passed"] or dice_gate["dice_prior_sha256"] != sha(args.dice_class_weights) or dice_prior["partition_sha256"] != sha(OUT/"partition.csv"):
            raise ValueError("Balanced-Dice preflight/partition mismatch")
        dice_class_weights = torch.tensor(dice_prior["dice_class_weights"],device="cuda",dtype=torch.float32)
    if args.foreground_crops:
        if args.mode != "patch" or not args.present_dice or args.positive_weights:
            raise ValueError("Foreground crop experiment uses patch mode and present-class Dice only")
        sampling_gate = json.loads((OUT/"foreground_preflight.json").read_text())
        sampling_tests = json.loads((OUT/"foreground_sampling_tests.json").read_text())
        if not sampling_gate["passed"] or not sampling_tests["passed"] or sampling_gate["partition_sha256"] != sha(OUT/"partition.csv"):
            raise ValueError("Foreground sampler preflight failed")
    if args.present_dice:
        if args.positive_weights:
            raise ValueError("Do not combine the two objective ablations")
        present_gate = json.loads((OUT/("present_preflight.json" if args.mode == "patch" else "present_whole_preflight.json")).read_text())
        if not present_gate["passed"] or present_gate["partition_sha256"] != sha(OUT/"partition.csv"):
            raise ValueError("Present-Dice preflight/partition mismatch")
    if args.positive_weights:
        prior = json.loads(args.positive_weights.read_text())
        weighted_gate = json.loads((OUT/"weighted_preflight.json").read_text())
        if prior["partition_sha256"] != sha(OUT/"partition.csv") or not weighted_gate["passed"] or weighted_gate["prior_sha256"] != sha(args.positive_weights):
            raise ValueError("Weighted objective preflight/partition mismatch")
        positive_weights = torch.tensor(prior["positive_weights"],dtype=torch.float32,device="cuda")
    if args.output.exists():
        raise FileExistsError("Use a new run directory; results are never overwritten")
    args.output.mkdir(parents=True)
    seed_all(42)
    train_rows, val_rows = records("train"), records("val")
    config = {"mode":args.mode,"seed":42,"base_channels":16,"batch_size":8,"num_workers":2,
              "epochs":args.epochs,"learning_rate":0.001,"optimizer":"Adam","amp":"float16 forward / float32 loss",
              "loss":"0.5 BCE + 0.5 per-image per-class soft Dice, smoothing=1 (Phase I formula)",
              "early_stopping_patience":2,"augmentation":None,"gradient_clipping":None,"scheduler":None,
              "training_images":len(train_rows),"validation_images":len(val_rows),
              "whole_size":[128,512],"patch_size":[256,256],"stride":128,"blend":"hann",
              "train_sampling":"one sample per source image per epoch; uniform deterministic grid patch for patch mode",
              "training_pixels_per_sample":65536,"partition_sha256":sha(OUT/"partition.csv"),
              "selection":"highest full native-resolution validation macro Dice at threshold 0.5; first epoch wins ties"}
    if prior:
        config["loss"] = "0.5 positive-weighted BCE + 0.5 unchanged per-image/class soft Dice, smoothing=1"
        config["positive_weights"] = prior["positive_weights"]
        config["prior_sha256"] = sha(args.positive_weights)
    if args.present_dice:
        config["loss"] = "0.5 unchanged BCE + 0.5 soft Dice over GT-positive image/class pairs only; smoothing=1; empty batch Dice term=0"
    if args.foreground_crops:
        config["foreground_crops"] = True
        config["foreground_focus_source_image_counts"] = [sum(bool(row["rle"].split("|")[c]) for row in train_rows) for c in range(4)]
        config["train_sampling"] = "one sample per source image; choose its rarest present class by training image count, then a uniform native grid patch containing that class; negatives keep uniform grid sampling"
    if dice_prior:
        config["loss"] = "0.5 unchanged BCE + 0.5 positive-pair Dice weighted by inverse training positive-pair frequency; mean over B*C; empty Dice terms zero; smoothing=1"
        config["dice_class_weights"] = dice_prior["dice_class_weights"]
        config["dice_prior_sha256"] = sha(args.dice_class_weights)
    collect_histograms = bool(prior or args.present_dice)
    model = SmallUNet().cuda()
    optimizer = torch.optim.Adam(model.parameters(),lr=config["learning_rate"])
    scaler = torch.amp.GradScaler("cuda")
    best, best_epoch, bad_epochs, steps, skipped = -1.,0,0,0,0
    history = []
    start_epoch, prior_duration = 0,0.
    prior_peak_allocated,prior_peak_reserved = 0,0
    if args.resume_run:
        parent = args.resume_run.resolve()
        if not parent.is_relative_to(OUT.resolve()) or parent == args.output.resolve():
            raise ValueError("Resume only a preserved run inside this Phase II artifact directory")
        previous = json.loads((parent/"config.json").read_text())
        differences = {k for k in set(previous)|set(config) if previous.get(k) != config.get(k)}
        if differences-{"epochs"}:
            raise ValueError(f"Continuation may change only total epoch budget: {differences}")
        info = json.loads((parent/"run_info.json").read_text())
        if info["status"] != "completed" or args.epochs <= info["epochs_completed"]:
            raise ValueError("Extend a completed run with a larger total epoch cap")
        state_path = parent/"last_training_state.pt"
        resume_gate = json.loads((OUT/"resume_preflight.json").read_text())
        if not resume_gate["passed"] or resume_gate["parent_state_sha256"] != sha(state_path) or resume_gate["parent_config_sha256"] != sha(parent/"config.json"):
            raise ValueError("Run the exact-parent real-GPU continuation preflight first")
        state = torch.load(state_path,map_location="cpu",weights_only=True)
        start_epoch = int(state["epoch"])
        if start_epoch != info["epochs_completed"] or state["config"] != previous:
            raise ValueError("Parent training state/config mismatch")
        model.load_state_dict(state["model_state_dict"],strict=True)
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        torch.set_rng_state(state["torch_rng"])
        torch.cuda.set_rng_state_all(state["cuda_rng"])
        best,best_epoch = info["best_validation_macro_dice"],info["best_epoch"]
        bad_epochs = start_epoch-best_epoch
        if bad_epochs >= config["early_stopping_patience"]:
            raise ValueError("Parent already met the unchanged early-stopping rule; budget extension is not justified by that rule")
        steps = info.get("attempted_optimizer_steps",info["optimizer_steps"])
        skipped = info.get("skipped_nonfinite_steps",0)
        prior_duration = info["duration_seconds"]
        prior_peak_allocated,prior_peak_reserved = info["peak_allocated_bytes"],info["peak_reserved_bytes"]
        with (parent/"history.csv").open(newline="",encoding="utf-8") as f:
            for row in csv.DictReader(f):
                history.append({k:int(v) if k in ["epoch","attempted_optimizer_steps_total","skipped_nonfinite_steps_total"] else float(v) for k,v in row.items()})
        if len(history) != start_epoch:
            raise ValueError("Parent history incomplete")
        for file in parent.glob("epoch_*_target_exposure.json"):
            shutil.copy2(file,args.output/file.name)
        shutil.copy2(parent/"best_model.pt",args.output/"best_model.pt")
        shutil.copy2(parent/"thresholds.json",args.output/"thresholds.json")
        config["continuation_parent"] = parent.name
        config["continuation_state_sha256"] = sha(state_path)
        config["continuation_start_epoch"] = start_epoch
        print(f"Restored weights, Adam, AMP scaler and RNG at epoch {start_epoch}; total cap {args.epochs}. Parent evidence is unchanged.",flush=True)
        del state
    write_json(args.output/"config.json",config)
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(start_epoch,args.epochs):
        epoch_start = time.perf_counter()
        model.train()
        generator = torch.Generator().manual_seed(42+epoch)
        loader = DataLoader(TrainDataset(train_rows,args.mode,epoch=epoch,foreground_crops=args.foreground_crops),batch_size=8,shuffle=True,
                            generator=generator,num_workers=2,pin_memory=True)
        losses, seen = 0.,0
        positive_pixels = np.zeros(4,np.int64)
        positive_samples = np.zeros(4,np.int64)
        for i,(images,masks) in enumerate(loader):
            positive_pixels += masks.sum((0,2,3)).numpy().astype(np.int64)
            positive_samples += masks.flatten(2).any(2).sum(0).numpy().astype(np.int64)
            images,masks = images.cuda(non_blocking=True),masks.cuda(non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda",dtype=torch.float16):
                logits = model(images)
                if args.present_dice:
                    from .objectives import present_dice_loss,balanced_present_dice_loss
                    loss = (present_dice_loss(logits,masks) if dice_class_weights is None else
                            balanced_present_dice_loss(logits,masks,dice_class_weights))
                elif positive_weights is None:
                    loss = loss_fn(logits,masks)
                else:
                    from .objectives import weighted_loss
                    loss = weighted_loss(logits,masks,positive_weights)
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite training loss")
            scaler.scale(loss).backward()
            old_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            skipped += int(scaler.get_scale() < old_scale)
            losses += float(loss.detach())*len(images)
            seen += len(images)
            steps += 1
            if (i+1)%100 == 0:
                print(f"{args.mode} epoch {epoch+1} step {i+1}/{len(loader)} loss={losses/seen:.5f}",flush=True)
        train_seconds = time.perf_counter()-epoch_start
        model.eval()
        totals = np.zeros((3,4),np.int64)
        if collect_histograms:
            from .evaluate import THRESHOLDS,add_histogram,tune_from_histograms
            positive_hist = np.zeros((4,len(THRESHOLDS)+1),np.int64)
            negative_hist = np.zeros_like(positive_hist)
        val_start = time.perf_counter()
        for i,row in enumerate(val_rows):
            image,truth = native(row)
            probability = predict(model,image,args.mode)
            totals += counts(probability>=0.5,truth)
            if collect_histograms:
                add_histogram(probability,truth,positive_hist,negative_hist)
            if (i+1)%300 == 0:
                print(f"{args.mode} epoch {epoch+1} native validation {i+1}/{len(val_rows)}",flush=True)
        score = summarize(totals)
        entry = {"epoch":epoch+1,"train_loss":losses/seen,"val_macro_dice":score["macro_dice"],"val_macro_iou":score["macro_iou"],
                 "train_seconds":train_seconds,"validation_seconds":time.perf_counter()-val_start,
                 "attempted_optimizer_steps_total":steps,"skipped_nonfinite_steps_total":skipped,"scaler_scale":scaler.get_scale()}
        entry.update({f"dice_class_{c}":score["per_class"][str(c)]["dice"] for c in range(1,5)})
        history.append(entry)
        write_json(args.output/f"epoch_{epoch+1}_target_exposure.json",{
            "positive_pixels_by_class":positive_pixels.tolist(),"positive_samples_by_class":positive_samples.tolist(),
            "samples":seen,"input_pixels":seen*65536})
        with (args.output/"history.csv").open("w",newline="",encoding="utf-8") as f:
            writer = csv.DictWriter(f,fieldnames=list(entry))
            writer.writeheader()
            writer.writerows(history)
        if score["macro_dice"] > best:
            best,best_epoch,bad_epochs = score["macro_dice"],epoch+1,0
            torch.save({"model_state_dict":model.state_dict(),"config":config,"class_names":CLASSES,
                        "epoch":epoch+1,"val_macro_dice":best,"validation":score,"smoke_test_only":False},args.output/"best_model.pt")
            if collect_histograms:
                tune_from_histograms(positive_hist,negative_hist,args.mode,len(val_rows),args.output/"thresholds.json")
                threshold_data = json.loads((args.output/"thresholds.json").read_text())
                threshold_data["checkpoint_sha256"] = sha(args.output/"best_model.pt")
                threshold_data["collection"] = "Fixed grid counts accumulated during full validation of the selected epoch; same tuning rule, avoids another GPU pass."
                write_json(args.output/"thresholds.json",threshold_data)
        else:
            bad_epochs += 1
        # Full resumable state; best_model is the inference artifact.
        torch.save({"model_state_dict":model.state_dict(),"optimizer":optimizer.state_dict(),"scaler":scaler.state_dict(),
                    "epoch":epoch+1,"torch_rng":torch.get_rng_state(),"cuda_rng":torch.cuda.get_rng_state_all(),
                    "config":config},args.output/"last_training_state.pt")
        write_json(args.output/"run_info.json",{"status":"running","best_epoch":best_epoch,"epochs_completed":epoch+1,
                   "best_validation_macro_dice":best,"optimizer_steps":steps-skipped,"attempted_optimizer_steps":steps,
                   "skipped_nonfinite_steps":skipped,"duration_seconds":prior_duration+time.perf_counter()-started,
                   "continuation_only_seconds":time.perf_counter()-started if args.resume_run else None,
                   "peak_allocated_bytes":max(prior_peak_allocated,torch.cuda.max_memory_allocated()),"peak_reserved_bytes":max(prior_peak_reserved,torch.cuda.max_memory_reserved()),
                   "parameters":sum(p.numel() for p in model.parameters())})
        print(json.dumps(entry,indent=2),flush=True)
        if bad_epochs >=2:
            break
    info = json.loads((args.output/"run_info.json").read_text())
    info["status"] = "completed"
    info["checkpoint_sha256"] = sha(args.output/"best_model.pt")
    write_json(args.output/"run_info.json",info)


if __name__ == "__main__":
    main()
