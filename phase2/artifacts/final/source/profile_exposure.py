"""Check exactness and CPU cost of training-only target-exposure bookkeeping."""
import json
import time

import numpy as np
import torch

from .common import OUT,write_json


def exposure_numpy(masks):
    areas = masks.numpy().sum(axis=(2,3),dtype=np.int64)
    return areas.sum(0), (areas>0).sum(0)


def main():
    torch.set_num_threads(4)
    torch.manual_seed(42)
    masks = (torch.rand(8,4,256,256)<0.03).float()
    def original():
        return masks.sum((0,2,3)).numpy().astype(np.int64),masks.flatten(2).any(2).sum(0).numpy().astype(np.int64)
    def proposed():
        return exposure_numpy(masks)
    for a,b in zip(original(),proposed()):
        np.testing.assert_array_equal(a,b)
    for _ in range(3):
        original(); proposed()
    timings = {"torch_original":[],"numpy_contiguous_reduction":[]}
    for _ in range(20):
        for name,function in [("torch_original",original),("numpy_contiguous_reduction",proposed)]:
            start = time.perf_counter()
            function()
            timings[name].append((time.perf_counter()-start)*1000)
    report = {"exact_counts_match":True,"shape":list(masks.shape),"torch_threads":4,"repeats":20,
              "milliseconds":{name:{"median":float(np.median(values)),"mean":float(np.mean(values))} for name,values in timings.items()},
              "scope":"Synthetic binary-target CPU bookkeeping only; not neural inference or end-to-end training throughput. Concurrent research workload may affect timings."}
    write_json(OUT/"cpu_exposure_profile.json",report)
    print(json.dumps(report,indent=2),flush=True)


if __name__ == "__main__":
    main()
