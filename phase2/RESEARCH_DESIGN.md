# Resolution-Aware Multi-Scale Steel Defect Segmentation Pipeline

## Hypothesis and scope of contribution

Preserving original spatial detail with overlapping patches and full-resolution probability reconstruction
may improve localization of small steel defects while retaining the existing compact U-Net.

The study compares coarse global image representation with native local representation. These are
different input scales/fields of view. There is no new simultaneous multi-scale feature-fusion module.
The contribution is this system's controlled implementation, validation and measured tradeoff, not
the invention of patching, weighted windows, class weighting or U-Net.

The [original U-Net paper](https://arxiv.org/pdf/1505.04597) already describes overlap-tile inference
for large images and discusses localization/context and memory tradeoffs. This is prior art.
Our unchanged SmallUNet uses padded convolutions and four sigmoid channels, rather than reproducing
that paper's exact architecture and original softmax/loss. No paper code or pretrained weights were imported.

## What changes spatially

The source image has 409,600 pixels (1600×256). Phase I and B resize it to 65,536 pixels
(512×128), retaining 16% of the pixel count. Horizontal scaling is 0.32 and vertical scaling 0.5.
The resize changes aspect ratio from 6.25 to 4. An old Phase I README's aspect-ratio-preservation
description is inaccurate; the original text/code remain preserved and this audit corrects the interpretation.

C presents one unresized 256×256 field per source image per training epoch. It receives the same number
of model input pixels as B, but its field of view, foreground-sample frequency, shape and pixel detail differ.
This intentionally measures a pipeline design. It cannot assign a result exclusively to resolution.

Exhaustive inference uses twelve overlapping fields, covering every native pixel. Thus C evaluates
786,432 model input pixels per full image, 12 times B's model input pixels. Overlap evaluates 1.92 times
the native image pixel count. Weighted probabilities are normalized in original coordinates before
the four independently locked thresholds. Parameter count stays 482,788.

## Controls and decision sequence

1. Freeze original Phase I source/checkpoint/result hashes and retain original validation assignments.
2. Reserve duplicate-group-separated test images only from original training. Old checkpoints are
   contaminated on this new test partition and cannot establish generalization there.
3. Check known RLE geometry, four channels, coverage, reconstruction, counts, threshold sweeps,
   component-size accounting, CUDA gradients, optimizer updates and checkpoint reloads.
4. Train B/C from seed 42, matching architecture, source-image order, optimizer, loss formula,
   batch size, input pixel budget and initial three-epoch budget. Select on full native validation Dice.
5. Record class target exposure. Uniform patches can omit rare defects even while foreground pixel
   counts resemble those in the resized baseline. Do not infer equal effective data from equal epoch counts.
6. Historical/new validation rare-class collapse motivates E, a separate change to BCE positive weights.
   Training-only priors define its weights; all other objective and training choices remain fixed.
   E's large false-positive counts motivate F as an independent ablation of the other loss component:
   retain original BCE and average soft Dice only over GT-positive image/class pairs. Empty batches
   have zero Dice loss but retain BCE. The observed collapse is consistent with empty-target Dice
   pressure; F tests this explanation rather than assuming it. All evaluation images remain included.
   G is F's whole-image counterpart. H separately changes F's training crop selection to retain
   a rare-class positive in each positive source; negatives keep uniform crops. H/F isolates the
   sampler, while F/G isolates representation under the revised loss. H/G changes two components
   and cannot isolate the effect of native resolution alone.
   If H retains rare positives but rare-class validation remains weak, J changes only Dice's
   weighting and normalization: inverse empirical positive-pair frequency, fixed from H's
   deterministic training targets, averaged over B*C pairs with GT-empty terms zero. BCE
   stays unchanged. H/J must have exactly matching training target exposure logs. This
   tests an established class-balancing idea; it does not claim a new loss algorithm.
   Before J's third validation pass, its increasing but weak first-two-epoch scores motivated
   a separately disclosed K continuation to a nine-epoch cap. Only total training budget changes;
   Adam/scaler/RNG and weights are restored, with a real-GPU resume equivalence test first.
   The original three-epoch run remains untouched. Patience-two early stopping is retained.
   K versus its parent tests the larger budget; K versus B/G/H changes multiple components.
7. Tune constant per-class thresholds on validation, choose the final candidate using validation,
   write a checkpoint/config/threshold selection lock, and only then evaluate the held-out test once.
8. Compare C's reconstruction ablations with identical weights and C thresholds: Hann stride 128,
   uniform stride 128, and edge-aligned minimal-overlap stride 256. The last configuration retains
   edge overlap to cover all pixels; do not describe it as strictly nonoverlapping.

## Interpretation rules for the viva

Primary comparisons are B/C for representation, C/E for objective, and C inference variants for
reconstruction. E/B changes two components and cannot establish the effect of patching alone.
Historical Phase I resized-validation scores cannot be used as a directly comparable held-out baseline.

Assess macro Dice/IoU together with every class's precision/recall and FP/FN pixels. Higher small-object
recall with excessive false positives does not by itself support improved segmentation. Size buckets
and object-recovery criteria are declared in README/metric artifacts, and pixel recall accompanies
the relaxed object-detection proxy. Include failures and all four classes in example selection.

Report measured signs and magnitudes, even when they contradict the hypothesis. The paired test-image
bootstrap describes sampling uncertainty of these fixed models; it does not replace repeat training seeds.
The initial three-epoch budget and one seed do not establish convergence or industrial robustness.
Adaptive coarse-to-fine selection is deferred so the exhaustive reference remains reliable and auditable.

The original runtime CPU re-evaluation reproduces Phase I to approximately 10^-9 in macro Dice.
GPU cross-check differences are recorded separately. New candidate inference latency is measured
serially after warmup, with synchronized GPU completion and no training running concurrently.
