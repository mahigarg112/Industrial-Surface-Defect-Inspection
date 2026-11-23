"""Tune on validation, lock selection, then evaluate test. Never retrains using test outcomes."""
import argparse
import json
import shutil
import hashlib
from pathlib import Path

import numpy as np
from PIL import Image

from .common import OUT, ROOT, load_model, seed_all, sha, write_json
from .data import records
from .evaluate import evaluate, tune
from .infer import run as infer


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stage",choices=["validate","test"],required=True)
    p.add_argument("--include-weighted",action="store_true",help="include separately trained E_weighted_patch candidate")
    p.add_argument("--include-present",action="store_true",help="include F_present_dice_patch objective ablation")
    p.add_argument("--include-whole-present",action="store_true",help="include matching G_present_dice_whole counterpart")
    p.add_argument("--include-foreground",action="store_true",help="include H_foreground_patch sampling ablation")
    p.add_argument("--include-balanced",action="store_true",help="include J_balanced_dice_patch class-frequency ablation")
    p.add_argument("--include-extended",action="store_true",help="include K_extended_patch, a separate continuation-budget experiment")
    args = p.parse_args()
    seed_all(42)
    labels = ["B_whole","C_patch"] + (["E_weighted_patch"] if args.include_weighted else []) + (["F_present_dice_patch"] if args.include_present else []) + (["G_present_dice_whole"] if args.include_whole_present else []) + (["H_foreground_patch"] if args.include_foreground else []) + (["J_balanced_dice_patch"] if args.include_balanced else []) + (["K_extended_patch"] if args.include_extended else [])
    if args.stage == "validate":
        if args.include_extended:
            extended = json.loads((OUT/"K_extended_patch/config.json").read_text())
            parent = OUT/extended["continuation_parent"]
            previous = json.loads((parent/"config.json").read_text())
            differences = {key for key in set(previous)|set(extended) if previous.get(key) != extended.get(key)}
            permitted = {"epochs","continuation_parent","continuation_state_sha256","continuation_start_epoch"}
            if differences-permitted or sha(parent/"last_training_state.pt") != extended["continuation_state_sha256"]:
                raise ValueError("Continuation changed components besides its training budget, or its parent changed")
            for file in parent.glob("epoch_*_target_exposure.json"):
                if sha(file) != sha(OUT/"K_extended_patch"/file.name):
                    raise ValueError("Continuation did not preserve parent exposure evidence")
        for variant in [label for label in labels if label in ["E_weighted_patch","F_present_dice_patch"]]:
            # Objective ablation must retain exactly the same patch target sampling.
            c_exposure = sorted((OUT/"C_patch").glob("epoch_*_target_exposure.json"))
            e_exposure = sorted((OUT/variant).glob("epoch_*_target_exposure.json"))
            if len(c_exposure) != len(e_exposure) or any(json.loads(c.read_text()) != json.loads(e.read_text()) for c,e in zip(c_exposure,e_exposure)):
                raise ValueError(f"C/{variant} target exposure differs: objective ablation is uncontrolled")
        if args.include_whole_present:
            b_exposure = sorted((OUT/"B_whole").glob("epoch_*_target_exposure.json"))
            g_exposure = sorted((OUT/"G_present_dice_whole").glob("epoch_*_target_exposure.json"))
            if len(b_exposure) != len(g_exposure) or any(json.loads(b.read_text()) != json.loads(g.read_text()) for b,g in zip(b_exposure,g_exposure)):
                raise ValueError("B/G target exposure differs")
        if args.include_foreground:
            f_config = json.loads((OUT/"F_present_dice_patch/config.json").read_text())
            h_config = json.loads((OUT/"H_foreground_patch/config.json").read_text())
            permitted = {"train_sampling","foreground_crops","foreground_focus_source_image_counts"}
            differences = {key for key in set(f_config)|set(h_config) if f_config.get(key) != h_config.get(key)}
            if differences-permitted:
                raise ValueError(f"F/H changed more than crop sampling: {differences-permitted}")
        if args.include_balanced:
            h_config = json.loads((OUT/"H_foreground_patch/config.json").read_text())
            j_config = json.loads((OUT/"J_balanced_dice_patch/config.json").read_text())
            permitted = {"loss","dice_class_weights","dice_prior_sha256"}
            differences = {key for key in set(h_config)|set(j_config) if h_config.get(key) != j_config.get(key)}
            if differences-permitted:
                raise ValueError(f"H/J changed more than Dice reduction: {differences-permitted}")
            h_exposure = sorted((OUT/"H_foreground_patch").glob("epoch_*_target_exposure.json"))
            j_exposure = sorted((OUT/"J_balanced_dice_patch").glob("epoch_*_target_exposure.json"))
            if len(h_exposure) != len(j_exposure) or any(json.loads(h.read_text()) != json.loads(j.read_text()) for h,j in zip(h_exposure,j_exposure)):
                raise ValueError("H/J crop target exposure differs")
        for label in labels:
            directory = OUT/label
            info = json.loads((directory/"run_info.json").read_text())
            if info["status"] != "completed":
                raise RuntimeError(f"{label} training incomplete")
            path = directory/"thresholds.json"
            if path.exists():
                old = json.loads(path.read_text())
                if old.get("checkpoint_sha256") != sha(directory/"best_model.pt") or old["partition_sha256"] != sha(OUT/"partition.csv"):
                    raise ValueError("Existing threshold evidence has changed inputs")
                print(f"Reusing frozen validation thresholds for {label}",flush=True)
                continue
            model,ckpt = load_model(directory/"best_model.pt")
            tune(model,ckpt["config"]["mode"],records("val"),path)
            value = json.loads(path.read_text())
            value["checkpoint_sha256"] = sha(directory/"best_model.pt")
            write_json(path,value)
            del model
        summaries = {label:json.loads((OUT/label/"thresholds.json").read_text()) for label in labels}
        write_json(OUT/"validation_comparison.json",{label:{"thresholds":v["thresholds"],
                    "default":v["validation_default_metrics"],"tuned":v["validation_metrics"]} for label,v in summaries.items()})
        print(json.dumps(json.loads((OUT/"validation_comparison.json").read_text()),indent=2),flush=True)
        # Intentionally stop for assessment before the irreversible test opening.
        # Authorized --stage test performs concrete selection from these frozen validation scores.
        return
    if (OUT/"final").exists():
        raise FileExistsError("Final model is already locked; do not overwrite test evidence")
    candidates = {}
    for label in labels:
        data = json.loads((OUT/label/"thresholds.json").read_text())
        if data["partition_sha256"] != sha(OUT/"partition.csv") or data["checkpoint_sha256"] != sha(OUT/label/"best_model.pt"):
            raise ValueError("Locked validation evidence no longer matches")
        candidates[label] = data
    chosen = max(candidates,key=lambda label:candidates[label]["validation_metrics"]["macro_dice"])
    selected = OUT/"final"
    selected.mkdir()
    shutil.copy2(OUT/chosen/"best_model.pt",selected/"model.pt")
    lock = {"selected_candidate":chosen,"candidates":labels,"selection_split":"val","selection_metric":"highest validation macro Dice after per-class threshold tuning; earlier candidate order wins exact ties",
            "checkpoint_sha256":sha(selected/"model.pt"),"thresholds":candidates[chosen]["thresholds"],
            "threshold_file_sha256":sha(OUT/chosen/"thresholds.json"),"partition_sha256":sha(OUT/"partition.csv"),
            "stride":128,"blend":"hann","selected_validation_macro_dice":candidates[chosen]["validation_metrics"]["macro_dice"],
            "test_opened_at_selection":False,"code_sha256":{str(p.relative_to(ROOT)):sha(p) for p in (ROOT/"phase2").glob("*.py")}}
    write_json(selected/"lock.json",lock)
    source = selected/"source"
    source.mkdir()
    for path in (ROOT/"phase2").glob("*.py"):
        shutil.copy2(path,source/path.name)
    for filename in ["README.md","requirements.txt","requirements-lock.txt","SOURCE_AND_CONTRIBUTION_LOG.md","RESEARCH_DESIGN.md","EXPERIMENT_LOG.md","architecture.md"]:
        shutil.copy2(ROOT/"phase2"/filename,source/filename)
    write_json(OUT/"STATUS.json",{"phase2_complete":False,"stage":"selection locked; held-out test evaluation running","selected_candidate":chosen})
    rows = records("test")
    write_json(OUT/"test_image_ids.json",[r["image_id"] for r in rows])
    for label in labels:
        model,ckpt = load_model(OUT/label/"best_model.pt")
        result = evaluate(model,ckpt["config"]["mode"],rows,candidates[label]["thresholds"],OUT/label/"test")
        if label == "C_patch":
            # Same weights, images and thresholds: isolate reconstruction only.
            evaluate(model,"patch",rows,candidates[label]["thresholds"],OUT/"ablations"/"C_uniform_stride128",kind="uniform",examples=False)
            evaluate(model,"patch",rows,candidates[label]["thresholds"],OUT/"ablations"/"C_hann_stride256",stride=256,examples=False)
        del model
    inventory = json.loads((OUT/"image_hashes.json").read_text())
    seen_files = {row["file_sha256"] for row in inventory}
    seen_pixels = {row["pixel_sha256"] for row in inventory}
    image = None
    for candidate in sorted((ROOT/"data/severstal-steel-defect-detection/test_images").glob("*.jpg")):
        if sha(candidate) in seen_files:
            continue
        with Image.open(candidate) as f:
            pixel_sha = hashlib.sha256(np.asarray(f.convert("RGB")).tobytes()).hexdigest()
        if pixel_sha not in seen_pixels:
            image = candidate
            break
    if image is None:
        raise RuntimeError("No byte/pixel-distinct new steel image found")
    write_json(OUT/"new_image_novelty.json",{"image":str(image),"file_sha256":sha(image),"pixel_sha256":pixel_sha,"byte_and_pixel_distinct_from_all_labeled_images":True,"labeled_images_checked":len(inventory),"near_duplicates_not_checked":True})
    infer(selected,image,OUT/"new_image_inference")
    
    from .report import generate
    generate()


if __name__ == "__main__":
    main()
