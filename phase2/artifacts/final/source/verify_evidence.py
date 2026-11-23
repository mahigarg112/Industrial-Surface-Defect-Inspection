"""Independent post-lock arithmetic and artifact consistency checks; never selects a model."""
import json

import numpy as np
from PIL import Image

from .common import OUT, ROOT, load_model, sha, write_json
from .data import records


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def verify():
    lock = read(OUT/"final/lock.json")  # No access to held-out labels before a selection lock exists.
    rows = records("test")
    ids = [row["image_id"] for row in rows]
    assert ids == read(OUT/"test_image_ids.json") and len(ids) == len(set(ids)) == 1006
    assert sha(OUT/"partition.csv") == lock["partition_sha256"]
    # RLE run lengths independently give native GT areas without using segmentation metric code.
    gt_pixels = []
    for row in rows:
        gt_pixels.append([sum(int(v) for v in rle.split()[1::2]) for rle in row["rle"].split("|")])
    gt_pixels = np.asarray(gt_pixels,dtype=np.int64)
    paths = [OUT/label/"test" for label in lock["candidates"]]
    paths += [OUT/"ablations"/label for label in ["C_uniform_stride128","C_hann_stride256"]]
    for path in paths:
        data,metric = read(path/"per_image_errors.json"),read(path/"metrics.json")
        assert [row["image_id"] for row in data] == ids
        counts = np.asarray([row["class_counts_tp_fp_fn"] for row in data],dtype=np.int64)
        assert counts.shape == (1006,3,4) and np.all(counts>=0)
        assert np.array_equal(counts[:,0,:]+counts[:,2,:],gt_pixels)
        tp,fp,fn = counts.sum(0)
        dice,iou = 2*tp/(2*tp+fp+fn),tp/(tp+fp+fn)
        assert np.isclose(dice.mean(),metric["macro_dice"],rtol=0,atol=1e-12)
        assert np.isclose(iou.mean(),metric["macro_iou"],rtol=0,atol=1e-12)
        for c in range(4):
            measured = metric["per_class"][str(c+1)]
            for key,value in [("dice",dice[c]),("iou",iou[c]),
                              ("precision",tp[c]/(tp[c]+fp[c]) if tp[c]+fp[c] else None),
                              ("recall",tp[c]/(tp[c]+fn[c]))]:
                assert measured[key] is None if value is None else np.isclose(measured[key],value,rtol=0,atol=1e-12)
            for key,value in [("true_positive_pixels",tp[c]),("false_positive_pixels",fp[c]),("false_negative_pixels",fn[c])]:
                assert measured[key] == int(value)
        assert metric["images"] == len(ids) and metric["partition_sha256"] == lock["partition_sha256"]
    for relative,digest in lock["code_sha256"].items():
        # Source archived at model locking is immutable even if later analysis scripts are added.
        assert sha(OUT/"final/source"/(ROOT/relative).name) == digest
    checkpoint = OUT/"final/model.pt"
    assert sha(checkpoint) == lock["checkpoint_sha256"]
    model,_ = load_model(checkpoint,device="cpu")
    assert sum(p.numel() for p in model.parameters()) == 482788
    saved = np.load(OUT/"new_image_inference/prediction.npz")
    probability,masks = saved["probabilities"],saved["masks"]
    assert probability.shape == masks.shape == (4,256,1600)
    assert np.isfinite(probability).all() and probability.min()>=0 and probability.max()<=1
    assert np.array_equal(masks,probability>=np.asarray(lock["thresholds"])[:,None,None])
    for c in range(4):
        assert np.array_equal(np.asarray(Image.open(OUT/f"new_image_inference/class_{c+1}.png"))>0,masks[c])
    result = {"passed":True,"evaluations_checked":len(paths),"images_per_evaluation":len(ids),
              "independent_native_rle_area_check":True,"per_image_counts_reproduce_all_metrics":True,
              "archived_source_hashes_match_lock":True,"final_checkpoint_strict_cpu_reload":True,
              "new_image_masks_match_locked_thresholds_and_pngs":True,
              "checkpoint_sha256":lock["checkpoint_sha256"]}
    write_json(OUT/"evidence_verification.json",result)
    print(json.dumps(result,indent=2),flush=True)
    return result


if __name__ == "__main__":
    verify()
