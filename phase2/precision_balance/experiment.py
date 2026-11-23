"""One BCE/Dice coefficient intervention; historical sources are read only."""
from pathlib import Path
import csv
import ctypes
import json
import os
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.pop("SDI_PHASE2_OUTPUT", None)
from phase2.common import CLASSES, OUT, load_model, seed_all, sha, write_json
from phase2.data import TrainDataset, records
from phase2.objectives import balanced_present_dice_loss
from phase2.improvement_sprint import sprint as metrics

HERE = Path(__file__).resolve().parent


def precision_loss(logits, target, class_weights, bce_fraction=.75):
    logits, target = logits.float(), target.float()
    bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, target)
    probability = logits.sigmoid()
    area = target.sum((2, 3))
    dice = (2*(probability*target).sum((2, 3))+1)/(probability.sum((2, 3))+area+1)
    weights = class_weights.to(device=logits.device, dtype=torch.float32)[None, :]
    dice_loss = ((1-dice)*(area > 0).float()*weights).mean()
    return bce_fraction*bce+(1-bce_fraction)*dice_loss


def train():
    protocol = json.loads((HERE / "protocol.json").read_text())
    directory = HERE / "N_bce75_dice25"
    if directory.exists():
        raise FileExistsError("Preserve completed/interrupted experiment; never overwrite it")
    assert torch.cuda.is_available()
    assert sha(OUT / "final/model.pt") == protocol["initial_weights_sha256"]
    assert sha(OUT / "partition.csv") == protocol["partition_sha256"]
    assert sha(OUT / "K_extended_patch/last_training_state.pt") == protocol["initial_state_sha256"]
    directory.mkdir()
    seed_all(42)
    model, checkpoint = load_model(OUT / "final/model.pt", "cuda")
    state = torch.load(OUT / "K_extended_patch/last_training_state.pt", map_location="cpu", weights_only=True)
    assert state["epoch"] == checkpoint["epoch"] == 9
    assert all(torch.equal(v.cpu(), state["model_state_dict"][k]) for k, v in model.state_dict().items())
    optimizer = torch.optim.Adam(model.parameters(), lr=.0001)
    optimizer.load_state_dict(state["optimizer"])
    for group in optimizer.param_groups: group["lr"] = .0001
    scaler = torch.amp.GradScaler("cuda")
    scaler.load_state_dict(state["scaler"])
    torch.set_rng_state(state["torch_rng"])
    torch.cuda.set_rng_state_all(state["cuda_rng"])
    del state
    weights = torch.tensor(checkpoint["config"]["dice_class_weights"], device="cuda", dtype=torch.float32)
    config = {**checkpoint["config"], "learning_rate": .0001, "epochs": 12,
              "continuation_parent": "K_extended_patch", "continuation_start_epoch": 9,
              "continuation_state_sha256": protocol["initial_state_sha256"],
              "loss": "0.75 unweighted BCE +0.25 unchanged inverse-frequency positive-pair soft Dice; mean B*C; smoothing1",
              "bce_fraction": .75, "dice_fraction": .25,
              "selection": protocol["selection"], "fixed_thresholds": protocol["thresholds"],
              "source_sha256": sha(Path(__file__)), "protocol_sha256": sha(HERE / "protocol.json"),
              "reference": protocol["reference"], "optimizer_and_scaler_restored": True}
    write_json(directory / "config.json", config)
    (directory / "source_experiment.py").write_bytes(Path(__file__).read_bytes())
    rows = records("train")
    original = json.loads((ROOT / "phase2/improvement_sprint/postprocessing/results.json").read_text())["results"]["K_locked"]
    best, useful_best, bad, attempts, skipped = -1., original["macro_dice"], 0, 0, 0
    history = []
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(9, 12):
        model.train()  # Retained BN buffers update normally; no reset/freeze.
        loader = DataLoader(TrainDataset(rows, "patch", epoch=epoch, foreground_crops=True),
                            batch_size=8, num_workers=2, pin_memory=True, shuffle=True,
                            generator=torch.Generator().manual_seed(42+epoch))
        total_loss, seen = 0., 0
        pixels, pairs = np.zeros(4, np.int64), np.zeros(4, np.int64)
        train_start = time.perf_counter()
        for index, (images, masks) in enumerate(loader):
            pixels += masks.sum((0, 2, 3)).numpy().astype(np.int64)
            pairs += masks.flatten(2).any(2).sum(0).numpy().astype(np.int64)
            images, masks = images.cuda(non_blocking=True), masks.cuda(non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16):
                loss = precision_loss(model(images), masks, weights)
            if not torch.isfinite(loss): raise RuntimeError("Nonfinite training loss")
            scaler.scale(loss).backward()
            old_scale = scaler.get_scale()
            scaler.step(optimizer); scaler.update()
            skipped += int(scaler.get_scale() < old_scale); attempts += 1
            total_loss += float(loss.detach())*len(images); seen += len(images)
            if (index+1) % 100 == 0:
                print(f"N epoch{epoch+1} train {index+1}/{len(loader)} loss={total_loss/seen:.5f}", flush=True)
        train_seconds = time.perf_counter()-train_start
        exposure = {"samples": seen, "positive_samples_by_class": pairs.tolist(), "positive_pixels_by_class": pixels.tolist()}
        reference_exposure = json.loads((ROOT / f"phase2/improvement_sprint/L_low_lr/epoch_{epoch+1}_target_exposure.json").read_text())
        assert exposure == reference_exposure, "Sampling differs from the matched reference"
        write_json(directory / f"epoch_{epoch+1}_target_exposure.json", exposure)
        policies = [{**p, "name": "N_05" if i == 0 else "N_locked"} for i, p in enumerate(metrics.policies()[:2])]
        # Reuse exact native metrics and the SAME auxiliary original-K validation
        # objective as L; only training's loss coefficients are changed.
        evaluation = metrics.evaluate_validation(model, policies, f"N_epoch{epoch+1}")
        score = evaluation["results"]["N_locked"]
        score["safeguards_vs_K"] = metrics.safeguards(score, original)
        write_json(directory / f"epoch_{epoch+1}_validation.json", evaluation)
        entry = {"epoch": epoch+1, "train_loss_bce75_dice25": total_loss/seen,
                 "validation_auxiliary_original_K_loss": evaluation["validation_loss"],
                 "val_macro_dice": score["macro_dice"], "val_macro_iou": score["macro_iou"],
                 "class2_precision": score["per_class"]["2"]["precision"],
                 "class2_recall": score["per_class"]["2"]["recall"],
                 "normal_false_alarm_rate": score["normal_false_alarm_rate"],
                 "train_seconds": train_seconds, "validation_seconds": evaluation["seconds"],
                 "learning_rate": optimizer.param_groups[0]["lr"], "attempts": attempts, "skipped_steps": skipped}
        entry.update({f"dice_class_{c}": score["per_class"][str(c)]["dice"] for c in range(1, 5)})
        history.append(entry)
        with (directory / "history.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(entry)); writer.writeheader(); writer.writerows(history)
        if score["macro_dice"] > best:
            best = score["macro_dice"]
            torch.save({"model_state_dict": model.state_dict(), "config": config, "class_names": CLASSES,
                        "epoch": epoch+1, "val_macro_dice": best, "validation": score,
                        "smoke_test_only": False}, directory / "best_model.pt")
            write_json(directory / "best_validation.json", evaluation)
        if score["macro_dice"] >= useful_best+.002: useful_best, bad = score["macro_dice"], 0
        else: bad += 1
        torch.save({"model_state_dict": model.state_dict(), "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(),
                    "epoch": epoch+1, "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all(),
                    "config": config}, directory / "last_training_state.pt")
        info = {"status": "running", "epochs_completed": len(history), "best_validation_macro_dice": best,
                "attempted_steps": attempts, "successful_steps": attempts-skipped, "skipped_steps": skipped,
                "duration_seconds": time.perf_counter()-started,
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(), "checkpoint_sha256": sha(directory / "best_model.pt")}
        write_json(directory / "run_info.json", info)
        print(json.dumps(entry, indent=2), flush=True)
        if bad >= 2:
            print("Early stopping: same two-epoch useful-progress rule as L", flush=True); break
    info["status"] = "completed"; write_json(directory / "run_info.json", info)


if __name__ == "__main__":
    awake = bool(ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)) if os.name == "nt" else False
    try: train()
    finally:
        if awake: ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
