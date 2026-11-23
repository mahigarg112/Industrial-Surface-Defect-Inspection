"""Re-evaluate checkpoint A on original validation, original size and threshold."""
import json
import argparse
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from .common import LEGACY, IMAGES, OUT, legacy_module, load_model, seed_all, sha, write_json
from .metrics import summarize


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device",choices=["cpu","cuda"],default="cuda")
    args = parser.parse_args()
    seed_all(42)
    started = time.perf_counter()
    path = LEGACY/"dice_loss_v1/best_model.pt"
    model,checkpoint = load_model(path,args.device)
    config = checkpoint["config"]
    dataset = legacy_module("dataset").SeverstalSegmentationDataset(IMAGES,LEGACY/"prepared/manifest.csv","val",config["image_width"],config["image_height"])
    totals = np.zeros((3,4),np.int64)
    with torch.inference_mode():
        for i,(images,masks,_) in enumerate(DataLoader(dataset,batch_size=config["batch_size"],num_workers=0)):
            # FP32 matches the original configuration, with no AMP or threshold tuning.
            prediction = model(images.to(args.device)).sigmoid()>=config["threshold"]
            truth = masks.to(args.device)>=0.5
            current = torch.stack([(prediction&truth).sum((0,2,3)),(prediction&~truth).sum((0,2,3)),(~prediction&truth).sum((0,2,3))])
            totals += current.cpu().numpy()
            if (i+1)%200 == 0:
                print(f"A original validation {(i+1)*config['batch_size']}/{len(dataset)}",flush=True)
    result = summarize(totals)
    result.update({"phase":"Phase I","images":len(dataset),"evaluation_size":[128,512],"threshold":0.5,
                   "checkpoint_sha256":sha(path),"original_reported_macro_dice":0.11825690418481827,
                   "original_reported_macro_iou":0.07744534313678741,
                   "device":args.device,"torch_version":torch.__version__,"precision":"FP32",
                   "duration_seconds":time.perf_counter()-started,
                   "test_eligibility":"Original training images contaminate newly carved test; A is historical validation context only"})
    result["absolute_difference_from_original_dice"] = abs(result["macro_dice"]-result["original_reported_macro_dice"])
    output = "A_phase1_original_validation_cpu.json" if args.device == "cpu" else "A_phase1_original_validation.json"
    write_json(OUT/output,result)
    print(json.dumps(result,indent=2),flush=True)


if __name__ == "__main__":
    main()
