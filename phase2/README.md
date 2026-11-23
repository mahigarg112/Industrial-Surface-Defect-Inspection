> Clean-release note (20 November 2026): the current app and preparation guide are in the root README.md. The research account/commands below are retained from the original study. Its opening no-GUI statement describes the initial research phase; this release includes phase2/release/app.py. Reproduce from this release root with fresh outputs and an explicitly supplied dataset. Local phase2/model.py and dataset_utils.py supply active imports; historical originals remain in experiments/.

# Phase II — Resolution-aware segmentation research

This directory is isolated from Phase I. No GUI, website, or presentation is part of this work.
Check `artifacts/STATUS.json` for actual completion state. The existence of scripts or a smoke checkpoint does not establish research success.

The Phase I reference is `experiments/severstal_segmentation/dice_loss_v1/best_model.pt`:
validation macro Dice **0.1182569042** and macro IoU **0.0774453431**, threshold 0.5,
512×128 masks, 2,514 original validation images. Only class 3 had nonzero Dice.
The original BCE and rare-class sampling checkpoints are other Phase I experiments.

The executed held-out study selected `artifacts/final/model.pt` (K, nine total epochs):
macro Dice **0.233996**, macro IoU **0.137937** on 1,006 images. Fresh B whole-image baseline
Dice is **0.123897**; matched C patch-only baseline is **0.091816**. K changes objective,
crop policy and compute budget, so its improvement cannot be attributed to patching alone.
Class-2 Dice is **0.082752** and 452/472 GT-empty images trigger an any-pixel false alarm:
this checkpoint is an evaluated research artifact, with industrial reliability unestablished.
The report includes source disclosures, all four classes, failed experiments and limitations.

- [Measured report and comparisons](artifacts/RESEARCH_REPORT.md)
- [Final checkpoint selection lock](artifacts/final/lock.json)
- [Architecture diagram](artifacts/architecture.png) and [verified tensor shapes](artifacts/architecture_verified.json)
- [Training curves](artifacts/training_curves.png) and [reviewed per-class figure](artifacts/post_review/heldout_per_class_reviewed.png)
- [Qualitative error review](artifacts/post_review/QUALITATIVE_ERROR_ANALYSIS.md) and [interleaved latency benchmark](artifacts/post_review/interleaved_runtime.json)
- [New-image inference](artifacts/new_image_inference/inference.json) and [overlay](artifacts/new_image_inference/overlay.png)
- [Source and Contribution Log](SOURCE_AND_CONTRIBUTION_LOG.md)

## Frozen data protocol

The original manifest is untouched. Phase II reserves 10% of its original training groups,
stratified by four-class presence signature, with seed 42. All original validation assignments remain.
There are 9,048 new training images, 2,514 original validation images and 1,006 held-out test images.
SHA-256 of decoded RGB pixels groups exact duplicates; file hashes are also recorded.
Near duplicates, consecutive coil frames, and unprovided acquisition groups cannot be ruled out.
Original train duplicates of validation would be quarantined, preserving original assignments; none were found.
All source annotation strings were checked against `train.csv`. Geometry and RLE tests use independent known masks.

**Phase I trained on the new test images. Its scores on them would be contaminated.**
A is re-evaluated on original validation at original resolution/threshold as historical context.
Only newly trained B/C candidates provide legitimate paired held-out comparisons.
The historical 0.1183 cannot be compared directly with full-resolution test Dice as proof of improvement.

## Controlled experiment

- A: preserved Phase I checkpoint; original CPU PyTorch 2.15 evaluation, original preprocessing and threshold 0.5. A secondary CUDA FP32 check is retained.
- B: Small U-Net from scratch, whole image resized to 512×128.
- C: same Small U-Net from the same seeded initialization, native 256×256 patches.
- C inference ablations: edge-aligned minimal overlap (stride 256), uniform overlap blending, positive Hann overlap blending.
- E (separate objective experiment, if run): same patch pipeline as C with positive-weighted BCE; unchanged soft Dice and all other training settings.
- F (separate objective experiment): same C patch pipeline and unweighted BCE, with soft Dice averaged over GT-positive image/class pairs; no Dice term for an all-negative batch.
- G: matching whole-image present-class Dice model, providing a second controlled whole/patch comparison with F.
- H: same F model/objective, selecting a native grid crop containing each positive source image's rarest present class; negative images retain uniform crops. This isolates training crop selection.
- J (if run): same H crop schedule and unchanged BCE, weighting positive-pair Dice by inverse empirical training pair frequency. This isolates the Dice reduction; weights use training exposures only.
- K (separate budget experiment): continue J's completed state to a nine-epoch total cap, retaining its loss, crops, optimizer, learning rate and patience-two early stopping. Restore Adam, AMP scaler and RNG; preserve J's three-epoch results.

B/C have the same 65,536 input pixels per training sample, seed 42, shuffled source-image order,
batch size 8, Adam learning rate 0.001, equal BCE/soft-Dice objective and initial epoch budget (3).
They use FP16 forward and FP32 loss reductions, two CPU workers and no augmentation, schedule,
gradient clipping, pretrained weights or class sampling. One grid patch is sampled uniformly
per source image per epoch for C. Training is **not exhaustive patch enumeration**.
Class exposure counts are recorded; a patch can omit a defect visible in its source image.
The per-image Dice smoothing of 1 has a spatial/empty-mask dependence; the formula is held fixed.

Select each checkpoint by highest **full native-resolution** validation macro Dice at threshold 0.5,
first epoch winning ties; early stopping after two unimproved epochs. Same protocol for B/C.
Three epochs is a modest single-seed budget, not evidence of convergence. Early stopping can change
actual update budgets; logs disclose updates, skipped AMP steps, wall time, and target exposure.
Phase I used batch size 2, FP32 CPU, the full 10,054 training images and five epochs: those budgets differ.

Historical Phase I and B validation show zero rare-class recall. This motivates a separate E experiment:
`pos_weight = min(training_negative_pixels / training_positive_pixels, 1000)` independently per class,
computed from native **training masks only**. The weights are 1000, 1000, 37.87235 and 184.13296.
Only BCE's positive term changes. No augmentation, architecture, sampling, schedule or soft-Dice change is bundled.
An eight-step weighted GPU preflight is required before E. E/C isolates this objective change; E/B does not isolate patching.
E accumulates threshold histograms during validation and retains those for the selected epoch,
using the same grid/tuning rule. This avoids another GPU validation pass; its validation wall time
includes that additional CPU histogram work. B/C collect their threshold histograms after training.
F tests the other loss component independently: original BCE is retained, and only Dice's handling
of empty GT image/class pairs changes. BCE still penalizes every negative pixel. No empty images
are removed from validation/test metrics. It is motivated by B/C collapse and E's excessive validation
false positives. F uses the same crop schedule as C, and its threshold histograms are collected like E.
H separately addresses the observed rare-positive exposure gap (33 versus 177 class-2 samples in
the first F/C versus whole-image epoch). Every source image still appears once per epoch. Among
classes with GT in a positive source, choose the one with the lowest training-image frequency,
then draw uniformly from original grid patches containing it. All four aligned GT channels are
retained. Negative source images keep uniform grid sampling. GT is never used for inference;
H validation/test still inspect every patch. H/F isolates this sampler, while F/G isolates representation.
If foreground sampling still leaves a rare class weak, J tests inverse positive-pair-frequency
weighting. Its fixed four weights equal training draws divided by positive image/class pairs,
measured over H's deterministic training crop schedule. Dice is then averaged over all B*C pairs
with GT-empty terms zero; BCE stays unchanged. This balances class contributions in expectation,
changes the loss normalization, and is not a newly invented objective. J repeats H from scratch
with the same seed and crops; their exposure logs must match exactly. No validation or test
predictions or labels enter these weights. J retains the initial three-epoch budget. K separately
tests a larger budget because J's first two validation scores and rare-class Dice were increasing
but weak. This decision was frozen before J's third validation pass and before held-out evaluation.
K's larger compute budget cannot establish a patching-only or loss-only improvement over B/G/H.
A real-GPU save/reload test must reproduce two uninterrupted optimizer updates before continuation.

After checkpoint selection, tune each class threshold on the full validation split using a fixed grid.
The common grid ranges from 0.00001 to 0.999, including high thresholds for positively weighted outputs.
Thresholds are constant over an image and all patches, and are applied **after probability reconstruction**.
Choose final candidate using validation, record checkpoint/config/threshold hashes, then open test results.
All new candidates use identical test image IDs, native-resolution GT, metric definitions and size buckets.
No test metric may drive retraining, threshold selection or final model selection in this study.
Additional experiments after opening test results require a new untouched final test set.

Patch selection, shape, native detail and field of view change together in B/C. This is a pipeline comparison;
it does not isolate native resolution as the sole cause. Reconstruction ablations isolate overlap/blending
using the same trained patch weights and the same locked C thresholds for all reconstruction ablations.

## Reconstruction and multi-scale terminology

The pipeline investigates a coarse whole-image representation versus native local detail.
It does not invent patching, introduce a new U-Net architecture, or claim a simultaneous multi-scale
feature-fusion network. The research contribution is the controlled resolution-aware design and measured
tradeoff in this steel-inspection system. Adaptive coarse-to-fine inference remains deferred.

Native 1600×256 images produce 12 patches with stride 128: starts 0,128,...,1280,1344.
The final start is edge-aligned, providing full coverage. No training mask interpolation occurs in C.
Four sigmoid channels preserve overlapping class labels. Inference batches at most eight patches.
Positive Hann weights (`max(hann,0.01)` on each axis) blend probabilities on the CPU,
normalizing by accumulated weights. Coordinates remain in the original image system.
B bilinearly upsamples probabilities to the original resolution before applying its locked thresholds.
The old resize from 1600×256 to 512×128 changes aspect ratio; the new B reproduces that preprocessing.

## Metrics and limitations

Accumulate TP/FP/FN in int64 over the entire split, then compute per-class Dice/IoU and their four-class mean.
Precision/recall with zero denominators are null. An absent class has null Dice/IoU; no empty-class bonus.
All full partitions contain all four classes, so macro scores are defined. Include every defect-free image.
False positives/negatives refer to pixels. Small-defect recall also counts GT 8-connected components,
separately by class: small ≤1,024 pixels; medium 1,025–8,192; large >8,192 at 1600×256.
An object is detected if at least 10% of its pixels are recovered. This is a declared localization proxy,
not instance matching or proof that a fragmented prediction is correct. Report pixel recall alongside it.

Latency excludes disk loading, GT decoding and metric computation. It includes CPU tile preparation,
GPU compute/transfers, probability reconstruction and final thresholding after three warmups. GPU synchronization is explicit.
Threshold histograms use four CPU threads with exact equivalence checked against brute-force thresholds.
Older E/F validation passes used serial class counting; their wall times are not an implementation-speed ablation.
Peak allocated/reserved VRAM are PyTorch process allocations; WDDM/nvidia-smi total usage is different.
Class 2 has only 20 positive test images: results will have substantial sampling uncertainty.
Single-seed comparisons cannot establish robustness across training seeds. Faculty approval of dataset use,
AI assistance and any publication remains the user's responsibility; no permissions are asserted.

## Windows PowerShell reproduction

Run from the project root. Preserve the original CPU environment.
For a fresh reproduction, choose an unused output directory. Existing evidence stays untouched.
The executed interpreter is 64-bit Python 3.14.5. The GPU environment uses PyTorch 2.11.0+cu128;
the preserved historical reference environment uses PyTorch 2.15.0+cpu. GPU, driver and CUDA
details are recorded by the real-device preflight. Reproduce the pinned dependency set below.

```powershell
$runDir = "phase2\reproduction_001"
$env:SDI_PHASE2_OUTPUT = Join-Path (Get-Location) $runDir
python -m venv .venv-phase2
.\.venv-phase2\Scripts\python.exe -m pip install torch==2.11.0+cu128 --index-url https://download.pytorch.org/whl/cu128
.\.venv-phase2\Scripts\python.exe -m pip install -r phase2\requirements-lock.txt
# Fresh artifact directory only; audit refuses to replace a frozen partition/snapshot.
.\.venv-phase2\Scripts\python.exe -B -m phase2.audit
.\.venv-phase2\Scripts\python.exe -B -m phase2.smoke
.\.venv-phase2\Scripts\python.exe -B -m phase2.test_evaluation
.\.venv-phase2\Scripts\python.exe -B -m phase2.inspect_architecture
.\.venv-phase2\Scripts\python.exe -B -m phase2.phase1_eval
# The preserved default Python contains the original CPU runtime:
python -B -m phase2.phase1_eval --device cpu
.\.venv-phase2\Scripts\python.exe -B -m phase2.train --mode whole --output "$runDir\B_whole" --epochs 3
.\.venv-phase2\Scripts\python.exe -B -m phase2.train --mode patch --output "$runDir\C_patch" --epochs 3
.\.venv-phase2\Scripts\python.exe -B -m phase2.finalize --stage validate
# Inspect validation_comparison.json before opening test; any new model experiment must precede test.
# Separate loss experiment if rare-class collapse warrants it:
.\.venv-phase2\Scripts\python.exe -B -m phase2.objectives
.\.venv-phase2\Scripts\python.exe -B -m phase2.weighted_smoke
.\.venv-phase2\Scripts\python.exe -B -m phase2.train --mode patch --positive-weights "$runDir\training_class_priors.json" --output "$runDir\E_weighted_patch" --epochs 3
.\.venv-phase2\Scripts\python.exe -B -m phase2.finalize --stage validate --include-weighted
.\.venv-phase2\Scripts\python.exe -B -m phase2.weighted_smoke --objective present
.\.venv-phase2\Scripts\python.exe -B -m phase2.train --mode patch --present-dice --output "$runDir\F_present_dice_patch" --epochs 3
.\.venv-phase2\Scripts\python.exe -B -m phase2.weighted_smoke --objective present --mode whole
.\.venv-phase2\Scripts\python.exe -B -m phase2.train --mode whole --present-dice --output "$runDir\G_present_dice_whole" --epochs 3
.\.venv-phase2\Scripts\python.exe -B -m phase2.test_sampling
.\.venv-phase2\Scripts\python.exe -B -m phase2.weighted_smoke --objective present --foreground-sampler
.\.venv-phase2\Scripts\python.exe -B -m phase2.train --mode patch --present-dice --foreground-crops --output "$runDir\H_foreground_patch" --epochs 3
.\.venv-phase2\Scripts\python.exe -B -m phase2.objectives --from-exposure "$runDir\H_foreground_patch"
.\.venv-phase2\Scripts\python.exe -B -m phase2.weighted_smoke --objective balanced --foreground-sampler --dice-priors "$runDir\dice_pair_priors.json"
.\.venv-phase2\Scripts\python.exe -B -m phase2.train --mode patch --present-dice --foreground-crops --dice-class-weights "$runDir\dice_pair_priors.json" --output "$runDir\J_balanced_dice_patch" --epochs 3
.\.venv-phase2\Scripts\python.exe -B -m phase2.resume_smoke --parent J_balanced_dice_patch
.\.venv-phase2\Scripts\python.exe -B -m phase2.train --mode patch --present-dice --foreground-crops --dice-class-weights "$runDir\dice_pair_priors.json" --resume-run "$runDir\J_balanced_dice_patch" --output "$runDir\K_extended_patch" --epochs 9
.\.venv-phase2\Scripts\python.exe -B -m phase2.finalize --stage validate --include-weighted --include-present --include-whole-present --include-foreground --include-balanced --include-extended
.\.venv-phase2\Scripts\python.exe -B -m phase2.finalize --stage test --include-weighted --include-present --include-whole-present --include-foreground --include-balanced --include-extended
.\.venv-phase2\Scripts\python.exe -B -m phase2.post_review
# Finalization automatically checks metric arithmetic, RLE areas, source hashes and exported masks.
# This command repeats those read-only consistency checks without reopening model selection:
.\.venv-phase2\Scripts\python.exe -B -m phase2.verify_evidence
.\.venv-phase2\Scripts\python.exe -B -m phase2.infer --package "$runDir\final" --image "C:\path\to\new_steel.jpg" --output "$runDir\another_image"
.\.venv-phase2\Scripts\python.exe -B -m phase2.audit --verify
```

Use `-B` for these commands so imports of Phase I do not create bytecode files there.
The executed study uses the default `phase2/artifacts`; the commands above use `phase2/reproduction_001`.
Choose another unused name for later reproductions. `SDI_PHASE2_OUTPUT` only selects output paths;
it does not change the dataset, partition RNG or model settings. Never delete evidence to rerun an experiment.
Pinned package versions (`requirements-lock.txt`) and GPU/driver details are recorded with the executed run.
Post-review adds a randomized interleaved latency measurement and clearer figure, preserving the
locked model, thresholds, partition and all test metric files. Its visual notes apply only to the
reviewed checkpoint/image IDs; a different reproduced checkpoint requires its own visual review.
The pre-test source snapshot remains in `artifacts/final/source`; the separately archived
post-review script and provenance identify reporting work performed after test opening.
