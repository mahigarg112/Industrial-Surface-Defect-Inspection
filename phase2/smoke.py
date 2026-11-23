"""Short real-data GPU optimizer test; never a final research checkpoint."""
import io
import json
import platform
import subprocess
import sys
import time
import unittest

import numpy as np
import torch
from torch.utils.data import DataLoader

from .common import CLASSES, OUT, SmallUNet, load_model, seed_all, sha, write_json
from .data import TrainDataset, native, records
from .patches import predict
from .test_pipeline import PipelineTests


def loss_fn(logits, target):
    # Accumulate Dice sums in FP32 even under autocast (FP16 sums can overflow).
    logits, target = logits.float(), target.float()
    bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, target)
    prob = logits.sigmoid()
    dice = (2*(prob*target).sum((2,3))+1) / (prob.sum((2,3))+target.sum((2,3))+1)
    return 0.5*bce + 0.5*(1-dice.mean())


def main():
    stream = io.StringIO()
    tests = unittest.TextTestRunner(stream=stream,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(PipelineTests))
    test_report = {"passed": tests.wasSuccessful(), "count": tests.testsRun, "output": stream.getvalue()}
    write_json(OUT / "pipeline_tests.json", test_report)
    print(stream.getvalue(), flush=True)
    if not tests.wasSuccessful():
        raise RuntimeError("Pipeline tests failed; training blocked")
    environment = {"python": sys.version, "executable": sys.executable, "platform": platform.platform(),
                   "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
                   "cuda_available": torch.cuda.is_available(),
                   "nvidia_smi": subprocess.run(["nvidia-smi"],capture_output=True,text=True).stdout}
    write_json(OUT / "environment.json",environment)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; no expensive CPU fallback")
    seed_all(42)
    environment.update({"gpu": torch.cuda.get_device_name(0), "vram_bytes": torch.cuda.get_device_properties(0).total_memory,
                        "cudnn": torch.backends.cudnn.version(), "compute_capability": list(torch.cuda.get_device_capability(0))})
    write_json(OUT / "environment.json",environment)
    rows = records("train")
    chosen = []
    for c in range(4):
        chosen.extend([r for r in rows if r["rle"].split("|")[c]][:4])
    results = {}
    for mode in ["whole", "patch"]:
        seed_all(42)
        model = SmallUNet().cuda().train()
        before = model.output.weight.detach().clone()
        optimizer = torch.optim.Adam(model.parameters(),lr=0.001)
        scaler = torch.amp.GradScaler("cuda")
        loader = DataLoader(TrainDataset(chosen,mode), batch_size=8, num_workers=0)
        torch.cuda.reset_peak_memory_stats()
        losses = []
        start = time.perf_counter()
        for _ in range(4):
            for images, masks in loader:
                images, masks = images.cuda(), masks.cuda()
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast("cuda",dtype=torch.float16):
                    logits = model(images)
                    loss = loss_fn(logits,masks)
                if logits.shape != masks.shape or not torch.isfinite(loss):
                    raise RuntimeError("Nonfinite loss or output/target misalignment")
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                if not all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()):
                    raise RuntimeError("Nonfinite gradients")
                scaler.step(optimizer)
                scaler.update()
                losses.append(float(loss.detach()))
        torch.cuda.synchronize()
        if torch.equal(before,model.output.weight):
            raise RuntimeError("Optimizer did not update weights")
        checkpoint = OUT / f"smoke_{mode}.pt"
        torch.save({"model_state_dict": model.state_dict(), "class_names": CLASSES,
                    "config": {"base_channels":16,"mode":mode},"smoke_test_only":True},checkpoint)
        reloaded, _ = load_model(checkpoint)
        image, _ = native(chosen[0])
        probability = predict(reloaded,image,mode)
        assert probability.shape == (4,256,1600) and np.isfinite(probability).all()
        results[mode] = {"optimizer_steps": len(losses),"batch_size":8,"amp":True,"losses":losses,
                         "duration_seconds":time.perf_counter()-start,"peak_allocated_bytes":torch.cuda.max_memory_allocated(),
                         "peak_reserved_bytes":torch.cuda.max_memory_reserved(), "checkpoint_reload":True,
                         "native_prediction_shape":list(probability.shape), "parameters":sum(p.numel() for p in model.parameters())}
        print(json.dumps({mode:results[mode]},indent=2),flush=True)
        del model, reloaded, optimizer
        torch.cuda.empty_cache()
    write_json(OUT / "smoke_results.json",{"passed":True,"partition_sha256":sha(OUT / "partition.csv"),"tests_passed":True,"modes":results,
                                           "warning":"Eight steps per mode on 16 images; verifies mechanics, not accuracy or convergence."})


if __name__ == "__main__":
    main()
