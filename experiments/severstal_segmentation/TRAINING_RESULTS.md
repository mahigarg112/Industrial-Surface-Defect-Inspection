# Training Results

## Status

This file is a results log, not a claim that training succeeded. Populate it only from actual run output and files. Preparation and smoke-test results should be recorded after those steps finish. Full training requires an explicit launch after reviewing the CPU runtime estimate.

## Run record

- Preparation: completed with seed 42; 12,568 image IDs, 10,054 train and 2,514 validation. 6,666 images have defects; 5,902 are defect-free. Exact duplicate groups found: 0.
- Smoke test: passed on CPU. Input `(2, 3, 128, 512)`, target/logit `(2, 4, 128, 512)`, finite BCE loss `0.653464`, one optimizer update completed.
- Full training: not started
- Device: Python reports PyTorch `2.15.0+cpu`; CUDA is unavailable to this environment.
- Hardware check: 16 GB physical RAM, about 2.5 GB available at inspection, and 24 logical CPUs. The CPU-only environment is expected to make a full pass lengthy.
- Runtime estimate: smoke training step plus save took 0.9 seconds for 2 images. Scaling that single batch across 10,054 train and 2,514 validation images suggests roughly 1.5 hours for one epoch; real throughput may differ. The configured run is one epoch with batch size 2 and has not been launched.
- Smoke checkpoint: `results/smoke_checkpoint.pt` (smoke only; not an evaluated trained model)
- Validation metrics: pending

## Metrics to report

Record final/best validation loss, mean-per-image macro Dice and IoU, and Dice/IoU for class IDs 1–4 from `results/validation_metrics.json`. Also inspect the per-class pixel TP/FP/FN counts. The small rare-class prevalence and the absence of an independent external test set limit how broadly these measurements generalize. Segmentation metrics are not comparable to Phase I classification accuracy.
