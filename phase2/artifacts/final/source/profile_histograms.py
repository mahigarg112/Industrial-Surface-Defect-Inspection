"""Exact-counter microbenchmark; does not evaluate or select a segmentation model."""
import json
import time

import numpy as np
import torch

from .common import OUT,write_json
from .evaluate import THRESHOLDS,add_histogram,_one_class_histogram


@torch.inference_mode()
def cuda_histogram(probability,truth):
    p = torch.from_numpy(np.ascontiguousarray(probability)).cuda()
    t = torch.from_numpy(np.ascontiguousarray(truth)).cuda().bool()
    boundaries = torch.as_tensor(THRESHOLDS,device="cuda")
    bins = torch.bucketize(p,boundaries,right=True)
    bins += torch.arange(4,device="cuda")[:,None,None]*(len(THRESHOLDS)+1)
    length = 4*(len(THRESHOLDS)+1)
    positive = torch.bincount(bins[t],minlength=length).reshape(4,-1).cpu().numpy()
    negative = torch.bincount(bins[~t],minlength=length).reshape(4,-1).cpu().numpy()
    return positive,negative


def main():
    torch.set_num_threads(4)
    rng = np.random.default_rng(42)
    p = rng.random((4,256,1600),dtype=np.float32)
    t = np.stack([np.asfortranarray(rng.random((256,1600))<0.02) for _ in range(4)])
    def serial():
        results = [_one_class_histogram(pair) for pair in zip(p,t)]
        return np.stack([v[0] for v in results]),np.stack([v[1] for v in results])
    def parallel():
        a = np.zeros((4,len(THRESHOLDS)+1),np.int64)
        b = np.zeros_like(a)
        add_histogram(p,t,a,b)
        return a,b
    functions = {"cpu_serial":serial,"cpu_four_threads":parallel,"cuda_including_transfers":lambda:cuda_histogram(p,t)}
    expected = serial()
    for function in functions.values():
        for actual,reference in zip(function(),expected):
            np.testing.assert_array_equal(actual,reference)
    timings = {name:[] for name in functions}
    for _ in range(5):
        for name,function in functions.items():
            start = time.perf_counter()
            function()
            timings[name].append((time.perf_counter()-start)*1000)
    report = {"exact_counts_match":True,"shape":list(p.shape),"seed":42,"repeats":5,
              "milliseconds":{name:{"median":float(np.median(v)),"mean":float(np.mean(v))} for name,v in timings.items()},
              "scope":"Synthetic probability/GT histogram only; includes transfers for CUDA; concurrent model work can affect results; not inference latency."}
    write_json(OUT/"threshold_histogram_profile.json",report)
    print(json.dumps(report,indent=2),flush=True)


if __name__ == "__main__":
    main()
