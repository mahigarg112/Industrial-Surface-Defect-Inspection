"""Validation-only threshold selection and locked full-resolution evaluation."""
import argparse
import csv
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy import ndimage

from .common import OUT, load_model, sha, write_json
from .data import native, records
from .metrics import counts, summarize
from .patches import predict

THRESHOLDS = np.array([0.00001,0.0001,0.001,0.005,0.01,0.025,0.05,0.075,0.1,0.15,0.2,0.25,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95,0.975,0.99,0.995,0.999],np.float32)
_HISTOGRAM_POOL = ThreadPoolExecutor(max_workers=4,thread_name_prefix="threshold_histogram")


def tune(model, mode, rows, output, stride=128, kind="hann"):
    """Cumulative exact counts at a fixed threshold grid without saving huge probability volumes."""
    positive_hist = np.zeros((4,len(THRESHOLDS)+1),np.int64)
    negative_hist = np.zeros_like(positive_hist)
    for i, row in enumerate(rows):
        image, truth = native(row)
        probability = predict(model,image,mode,stride=stride,kind=kind)
        add_histogram(probability,truth,positive_hist,negative_hist)
        if (i+1)%100 == 0:
            print(f"Threshold validation {i+1}/{len(rows)}",flush=True)
    return tune_from_histograms(positive_hist,negative_hist,mode,len(rows),output,stride,kind)


def add_histogram(probability,truth,positive_hist,negative_hist):
    if not np.isfinite(probability).all():
        raise ValueError("Nonfinite prediction probabilities")
    for c,(positive,negative) in enumerate(_HISTOGRAM_POOL.map(_one_class_histogram,zip(probability,truth))):
        positive_hist[c] += positive
        negative_hist[c] += negative


def _one_class_histogram(pair):
    probability,truth = pair
    bins = np.searchsorted(THRESHOLDS,probability.ravel(),side="right")
    label = truth.ravel().astype(bool)
    return (np.bincount(bins[label],minlength=len(THRESHOLDS)+1),
            np.bincount(bins[~label],minlength=len(THRESHOLDS)+1))


def tune_from_histograms(positive_hist,negative_hist,mode,image_count,output,stride=128,kind="hann"):
    tp = np.cumsum(positive_hist[:,::-1],axis=1)[:,::-1][:,1:]
    fp = np.cumsum(negative_hist[:,::-1],axis=1)[:,::-1][:,1:]
    fn = positive_hist.sum(1)[:,None]-tp
    dice = np.divide(2*tp,2*tp+fp+fn,out=np.zeros_like(tp,dtype=float),where=(2*tp+fp+fn)>0)
    # Higher threshold wins exact ties; independently selected per class on validation only.
    indices = len(THRESHOLDS)-1 - np.argmax(dice[:,::-1],axis=1)
    thresholds = [float(THRESHOLDS[index]) for index in indices]
    selected_totals = np.stack([tp[np.arange(4),indices],fp[np.arange(4),indices],fn[np.arange(4),indices]])
    default_index = int(np.flatnonzero(THRESHOLDS == np.float32(0.5))[0])
    write_json(output,{"thresholds":thresholds,"source_split":"val","images":image_count,
                       "candidate_thresholds":[float(v) for v in THRESHOLDS],"validation_dice_by_class_and_threshold":dice.tolist(),
                       "policy":"per-class maximum aggregate validation Dice; higher threshold wins ties",
                       "validation_metrics":summarize(selected_totals),
                       "validation_default_metrics":summarize(np.stack([tp[:,default_index],fp[:,default_index],fn[:,default_index]])),
                       "mode":mode,"stride":stride,"blend":kind,"partition_sha256":sha(OUT/"partition.csv")})
    return thresholds


def size_stats(prediction,truth):
    """8-connected GT components; detection requires >=10% of the component's pixels."""
    stats = {name:np.zeros((4,4),np.int64) for name in ["small","medium","large"]}
    # columns: objects, detected objects, GT pixels, recovered GT pixels
    for c in range(4):
        labels,n = ndimage.label(truth[c],structure=np.ones((3,3)))
        if not n:
            continue
        areas = np.bincount(labels.ravel(),minlength=n+1)[1:]
        recovered = np.bincount(labels.ravel(),weights=prediction[c].ravel(),minlength=n+1)[1:]
        for name, select in [("small",areas<=1024),("medium",(areas>1024)&(areas<=8192)),("large",areas>8192)]:
            stats[name][c] = [select.sum(),((recovered >= 0.1*areas)&select).sum(),areas[select].sum(),recovered[select].sum()]
    return stats


def panel(image,truth,prediction,path):
    colors = np.array([[255,60,60],[50,255,100],[60,150,255],[255,220,50]],np.float32)
    def overlay(mask):
        color = np.einsum("chw,ck->hwk",mask.astype(np.float32),colors)
        count = mask.sum(0)
        color /= np.maximum(count[:,:,None],1)
        return np.where(count[:,:,None]>0,0.6*image+0.4*color,image).astype(np.uint8)
    Image.fromarray(np.concatenate([image,overlay(truth),overlay(prediction)],axis=0)).save(path)


def evaluate(model,mode,rows,thresholds,output,stride=128,kind="hann",examples=True):
    output = Path(output)
    output.mkdir(parents=True,exist_ok=True)
    totals = np.zeros((3,4),np.int64)
    sizes = {s:np.zeros((4,4),np.int64) for s in ["small","medium","large"]}
    latencies, per_image, candidates, normal_candidates = [],[],[],[]
    class_candidates = {c:[] for c in range(4)}
    threshold_array = np.asarray(thresholds,dtype=np.float32)[:,None,None]
    torch.cuda.empty_cache()
    # Warmup excluded from latency; includes CPU tile prep, GPU compute, transfer, blending thereafter.
    image,_ = native(rows[0])
    for _ in range(3):
        predict(model,image,mode,stride=stride,kind=kind)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    for i,row in enumerate(rows):
        image,truth = native(row)
        start = time.perf_counter()
        probability = predict(model,image,mode,stride=stride,kind=kind)
        if not np.isfinite(probability).all():
            raise ValueError("Nonfinite prediction probabilities")
        prediction = probability >= threshold_array
        torch.cuda.synchronize()
        latencies.append((time.perf_counter()-start)*1000)
        current = counts(prediction,truth)
        totals += current
        stats = size_stats(prediction,truth)
        for name in sizes:
            sizes[name] += stats[name]
        tp,fp,fn = current
        score = float(2*tp.sum() / (2*tp.sum()+fp.sum()+fn.sum())) if (2*tp.sum()+fp.sum()+fn.sum()) else 1.0
        per_image.append({"image_id":row["image_id"],"foreground_dice":score,
                          "true_positive_pixels":int(tp.sum()),"false_positive_pixels":int(fp.sum()),"false_negative_pixels":int(fn.sum()),
                          "class_counts_tp_fp_fn":current.tolist()})
        if truth.any():
            candidates.append((score,row))
        elif fp.sum()>0:
            normal_candidates.append((int(fp.sum()),row))
        for c in range(4):
            if truth[c].any():
                class_score = float(2*tp[c]/(2*tp[c]+fp[c]+fn[c]))
                class_candidates[c].append((class_score,row))
        if (i+1)%100 == 0:
            print(f"Evaluation {i+1}/{len(rows)} {mode} Dice={summarize(totals)['macro_dice']:.4f}",flush=True)
    result = summarize(totals)
    result.update({"images":len(rows),"thresholds":list(thresholds),"mode":mode,"stride":stride,"blend":kind,
                   "evaluation_size":[256,1600],"partition_sha256":sha(OUT/"partition.csv"),
                   "parameters":sum(p.numel() for p in model.parameters()),
                   "peak_allocated_bytes":torch.cuda.max_memory_allocated(),"peak_reserved_bytes":torch.cuda.max_memory_reserved(),
                   "latency_ms":{"mean":float(np.mean(latencies)),"median":float(np.median(latencies)),"p95":float(np.percentile(latencies,95)),
                                  "scope":"native RGB array to native binary masks, CPU preparation + GPU + transfer + blending + thresholds; excludes disk decoding and metrics; 3 warmups"},
                   "size_definition":"8-connected class components at original resolution; small<=1024 px, medium<=8192 px, large>8192 px; detected if >=10% GT component pixels recovered; thresholds fixed before test",
                   "performance_by_size":{name:{str(c+1):{"objects":int(v[c,0]),"detected_objects":int(v[c,1]),
                                                               "object_recall":float(v[c,1]/v[c,0]) if v[c,0] else None,
                                                               "pixel_recall":float(v[c,3]/v[c,2]) if v[c,2] else None,
                                                               "ground_truth_pixels":int(v[c,2]),"recovered_pixels":int(v[c,3])} for c in range(4)} for name,v in sizes.items()}})
    write_json(output/"metrics.json",result)
    write_json(output/"per_image_errors.json",per_image)
    if examples and candidates:
        candidates.sort(key=lambda pair:pair[0])
        selected = [("failure",score,r,None) for score,r in candidates[:3]] + [("success",score,r,None) for score,r in candidates[-3:]]
        for c,options in class_candidates.items():
            options.sort(key=lambda pair:pair[0])
            if options:
                for label,(score,row) in [("worst",options[0]),("best",options[-1])]:
                    selected.append((f"class_{c+1}_{label}",score,row,c+1))
        normal_candidates.sort(key=lambda pair:pair[0],reverse=True)
        selected.extend((f"normal_false_alarm_{i+1}",0.0,row,None) for i,(_,row) in enumerate(normal_candidates[:3]))
        manifest = []
        for label,score,row,focus_class in selected:
            image,truth = native(row)
            prediction = predict(model,image,mode,stride=stride,kind=kind)>=np.asarray(thresholds)[:,None,None]
            name = f"{label}_{Path(row['image_id']).stem}.png"
            panel(image,truth,prediction,output/name)
            manifest.append({"image_id":row["image_id"],"label":label,"file":name,"example_dice":score,"focus_class":focus_class,
                             "score_scope":"specified class" if focus_class else "all-class foreground",
                             "false_positive_pixels":int((prediction & ~truth.astype(bool)).sum()),
                             "ground_truth_empty":bool(not truth.any()),
                             "selection":"three lowest/highest foreground Dice on positive images; per-class lowest/highest positive images; up to three largest pixel false alarms on GT-empty images; relative examples, not deployment acceptance",
                             "layout":"top=input; middle=GT; bottom=prediction; red/green/blue/yellow=classes 1/2/3/4"})
        write_json(output/"examples.json",manifest)
    print(json.dumps({"macro_dice":result["macro_dice"],"macro_iou":result["macro_iou"],"images":len(rows)},indent=2),flush=True)
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--split",choices=["val","test"],default="val")
    p.add_argument("--thresholds",type=Path)
    p.add_argument("--tune",action="store_true")
    args = p.parse_args()
    model, ckpt = load_model(args.checkpoint)
    if ckpt.get("smoke_test_only"):
        raise ValueError("Smoke checkpoints cannot be used for research evaluation")
    mode = ckpt["config"]["mode"]
    if args.tune:
        if args.split != "val":
            raise ValueError("Test threshold tuning is forbidden")
        tune(model,mode,records("val"),args.output/"thresholds.json")
    else:
        if not args.thresholds:
            p.error("Provide validation-locked --thresholds")
        locked = json.loads(args.thresholds.read_text())
        if locked["partition_sha256"] != sha(OUT/"partition.csv"):
            raise ValueError("Partition mismatch")
        evaluate(model,mode,records(args.split),locked["thresholds"],args.output)
