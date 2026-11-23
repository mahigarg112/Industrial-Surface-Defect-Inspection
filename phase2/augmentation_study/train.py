"""Matched two-epoch augmentation-only continuation from K epoch9."""
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
sys.path.insert(0,str(ROOT)); os.environ.pop("SDI_PHASE2_OUTPUT",None)
from phase2.common import OUT, CLASSES, load_model, seed_all, sha, write_json
from phase2.data import records
from phase2.objectives import balanced_present_dice_loss
from phase2.augmentation_study.augmentation import AugmentedDataset
from phase2.augmentation_study.diagnostics import evaluate

HERE = Path(__file__).resolve().parent


def main():
    protocol = json.loads((HERE/"protocol.json").read_text())
    if not (HERE/"baseline_diagnosis.json").exists(): raise RuntimeError("Complete diagnosis before training")
    assert json.loads((HERE/"augmentation_tests.json").read_text())["passed"]
    directory = HERE/"O_photometric"
    if directory.exists(): raise FileExistsError("Preserve completed or interrupted run")
    assert torch.cuda.is_available()
    assert sha(OUT/"final/model.pt")==protocol["initial_checkpoint_sha256"]
    assert sha(OUT/"K_extended_patch/last_training_state.pt")==protocol["initial_state_sha256"]
    assert sha(OUT/"partition.csv")==protocol["partition_sha256"]
    directory.mkdir(); seed_all(42)
    model,checkpoint = load_model(OUT/"final/model.pt","cuda")
    state = torch.load(OUT/"K_extended_patch/last_training_state.pt",map_location="cpu",weights_only=True)
    assert state["epoch"]==checkpoint["epoch"]==9
    assert all(torch.equal(v.cpu(),state["model_state_dict"][k]) for k,v in model.state_dict().items())
    optimizer = torch.optim.Adam(model.parameters(),lr=.0001)
    optimizer.load_state_dict(state["optimizer"])
    for group in optimizer.param_groups: group["lr"] = .0001
    scaler = torch.amp.GradScaler("cuda"); scaler.load_state_dict(state["scaler"])
    torch.set_rng_state(state["torch_rng"]); torch.cuda.set_rng_state_all(state["cuda_rng"])
    del state
    weights = torch.tensor(checkpoint["config"]["dice_class_weights"],device="cuda",dtype=torch.float32)
    config = {**checkpoint["config"],"epochs":11,"learning_rate":.0001,
              "continuation_parent":"K_extended_patch","continuation_start_epoch":9,
              "continuation_state_sha256":protocol["initial_state_sha256"],
              "augmentation":protocol["augmentation"],"matched_reference":protocol["reference"],
              "max_additional_epochs":2,"attempted_update_budget":2262,
              "selection":protocol["selection"],"fixed_thresholds":protocol["thresholds"],
              "protocol_sha256":sha(HERE/"protocol.json"),
              "source_sha256":{n:sha(HERE/n) for n in ["train.py","augmentation.py","diagnostics.py"]}}
    write_json(directory/"config.json",config)
    for n in config["source_sha256"]: (directory/("source_"+n)).write_bytes((HERE/n).read_bytes())
    rows = records("train"); history=[]; best=-1.; bad=0; attempts=skipped=0
    useful = .23922878227237032
    started=time.perf_counter(); torch.cuda.reset_peak_memory_stats()
    for epoch in [9,10]:
        dataset=AugmentedDataset(rows,epoch=epoch)
        loader=DataLoader(dataset,batch_size=8,shuffle=True,num_workers=2,pin_memory=True,
                          generator=torch.Generator().manual_seed(42+epoch))
        model.train(); loss_sum=0.; seen=0; pixels=np.zeros(4,np.int64); pairs=np.zeros(4,np.int64)
        tick=time.perf_counter()
        for index,(images,masks) in enumerate(loader):
            pixels+=masks.sum((0,2,3)).numpy().astype(np.int64)
            pairs+=masks.flatten(2).any(2).sum(0).numpy().astype(np.int64)
            images,masks=images.cuda(non_blocking=True),masks.cuda(non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda",dtype=torch.float16):
                loss=balanced_present_dice_loss(model(images),masks,weights)
            if not torch.isfinite(loss):raise RuntimeError("Nonfinite training loss")
            scaler.scale(loss).backward(); old_scale=scaler.get_scale()
            scaler.step(optimizer);scaler.update();skipped+=int(scaler.get_scale()<old_scale);attempts+=1
            loss_sum+=float(loss.detach())*len(images);seen+=len(images)
            if (index+1)%100==0:print(f"O epoch{epoch+1} train {index+1}/{len(loader)} loss={loss_sum/seen:.5f}",flush=True)
        train_seconds=time.perf_counter()-tick
        exposure={"samples":seen,"positive_samples_by_class":pairs.tolist(),"positive_pixels_by_class":pixels.tolist()}
        ref=json.loads((ROOT/f"phase2/improvement_sprint/L_low_lr/epoch_{epoch+1}_target_exposure.json").read_text())
        assert exposure==ref,"Augmentation changed mask/crop/source exposure"
        write_json(directory/f"epoch_{epoch+1}_target_exposure.json",exposure)
        applied=sum(p[0] for p in dataset.augmentation_parameters)
        result=evaluate({"O":model},f"O_epoch{epoch+1}",reset_seed=False)
        metric=result["models"]["O"]["metrics"]
        write_json(directory/f"epoch_{epoch+1}_validation.json",result)
        row={"epoch":epoch+1,"train_loss":loss_sum/seen,"val_macro_dice":metric["macro_dice"],
             "val_macro_iou":metric["macro_iou"],"normal_false_alarm_rate":metric["normal_false_alarm_rate"],
             "class2_precision":metric["per_class"]["2"]["precision"],"class2_recall":metric["per_class"]["2"]["recall"],
             "augmented_draws":int(applied),"samples":seen,"learning_rate":.0001,"attempted_steps":attempts,
             "skipped_steps":skipped,"train_seconds":train_seconds,"validation_seconds":result["seconds"]}
        row.update({f"dice_class_{c}":metric["per_class"][str(c)]["dice"] for c in range(1,5)})
        history.append(row)
        with (directory/"history.csv").open("w",newline="",encoding="utf-8") as f:
            writer=csv.DictWriter(f,fieldnames=list(row));writer.writeheader();writer.writerows(history)
        if metric["macro_dice"]>best:
            best=metric["macro_dice"]
            torch.save({"model_state_dict":model.state_dict(),"config":config,"class_names":CLASSES,
                        "epoch":epoch+1,"val_macro_dice":best,"validation":metric,"smoke_test_only":False},directory/"best_model.pt")
            write_json(directory/"best_validation.json",result)
        if metric["macro_dice"]>=useful+.002:useful,bad=metric["macro_dice"],0
        else:bad+=1
        torch.save({"model_state_dict":model.state_dict(),"optimizer":optimizer.state_dict(),"scaler":scaler.state_dict(),
                    "epoch":epoch+1,"torch_rng":torch.get_rng_state(),"cuda_rng":torch.cuda.get_rng_state_all(),"config":config},directory/"last_training_state.pt")
        info={"status":"running","epochs_completed":len(history),"best_validation_macro_dice":best,
              "attempted_steps":attempts,"successful_steps":attempts-skipped,"skipped_steps":skipped,
              "duration_seconds":time.perf_counter()-started,"peak_allocated_bytes":torch.cuda.max_memory_allocated(),
              "peak_reserved_bytes":torch.cuda.max_memory_reserved(),"checkpoint_sha256":sha(directory/"best_model.pt")}
        write_json(directory/"run_info.json",info);print(json.dumps(row,indent=2),flush=True)
        if bad>=2:print("Same patience-two stopping rule as L; bounded at two additional epochs",flush=True);break
    info["status"]="completed";write_json(directory/"run_info.json",info)


if __name__=="__main__":
    awake=bool(ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)) if os.name=="nt" else False
    try:main()
    finally:
        if awake:ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
