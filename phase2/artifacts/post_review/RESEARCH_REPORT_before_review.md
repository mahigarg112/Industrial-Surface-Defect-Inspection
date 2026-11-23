# Phase II measured research report

Selected using validation only: **K_extended_patch**, checkpoint `final/model.pt`.
Held-out native-resolution macro Dice **0.233996**, IoU **0.137937** on 1,006 images.

Weakest class: **class 2**, Dice **0.082752**. Completed research evaluation does not establish industrial deployment reliability.

Phase I historical validation: Dice **0.1182569**, IoU **0.0774453** at 512×128 and threshold 0.5.
These historical scores use different training/evaluation conditions; they do not prove a Phase II improvement.

Fresh A evaluation under original resize/threshold on CPU FP32 / PyTorch 2.15.0+cpu: Dice 0.118256904, IoU 0.077445345. A secondary CUDA FP32 check is retained separately.

## Matched held-out comparison

| Candidate | Macro Dice | Macro IoU | Mean latency (ms) | Peak allocated MiB |
|---|---:|---:|---:|---:|
| B_whole | 0.123897 | 0.082356 | 31.23 | 19.9 |
| C_patch | 0.091816 | 0.054641 | 112.37 | 113.9 |
| C_uniform_stride128 | 0.088497 | 0.052301 | 106.38 | 113.9 |
| C_hann_stride256 | 0.086718 | 0.051074 | 48.71 | 99.9 |
| E_weighted_patch | 0.079376 | 0.044382 | 69.75 | 113.9 |
| F_present_dice_patch | 0.117008 | 0.071204 | 68.28 | 113.9 |
| G_present_dice_whole | 0.155507 | 0.093929 | 23.11 | 19.9 |
| H_foreground_patch | 0.144306 | 0.086331 | 67.87 | 113.9 |
| J_balanced_dice_patch | 0.131488 | 0.075340 | 69.16 | 113.9 |
| K_extended_patch | 0.233996 | 0.137937 | 70.33 | 113.9 |

E changes only BCE positive weights relative to C, using native training priors capped at 1,000. Soft Dice, seed, architecture, sampling, optimizer and initial epoch budget remain fixed.
E was introduced after historical/new validation showed rare-class collapse, before opening test. E-versus-B cannot isolate patching because objective and representation both differ.
The primary patching comparison is B-versus-C; E-versus-C is the objective experiment. E's objective-specific loss curve is plotted separately.
E minus C macro Dice: -0.012440; paired-image 95% interval [-0.018399, -0.005851].

F retains C's unweighted BCE and changes only the soft-Dice term to average over GT-positive image/class pairs. Dice contributes zero on an all-negative batch; BCE still penalizes all false positives.
No empty validation/test images or classes are excluded from the reported metrics. The mask applies only inside the training loss. F uses the same native patch sampling and initial training budget as C.
F minus C macro Dice: +0.025192; paired-image 95% interval [+0.020237, +0.030562].

G is the matching whole-image model with present-class Dice. B/C and G/F form a controlled representation-by-Dice-reduction comparison; E is a separate BCE-weighting experiment.
Under present-class Dice, F (patch) minus G (whole) macro Dice: -0.038499; paired-image 95% interval [-0.047228, -0.030377].
This separates representation effects from changes to the training objective. Field of view and resizing remain inherent parts of the representation design.

H retains F's objective, architecture, seed, source-image order and pixel/update budget, changing only training crop selection. It chooses a grid crop containing the rarest GT class present in each positive source image; negative source images remain uniformly cropped.
Ground truth guides training only. H inference remains exhaustive, with the same 12-patch reference and locked per-class thresholds. Sampling is established methodology, not a newly invented algorithm.
H minus F macro Dice: +0.027298; paired-image 95% interval [+0.021033, +0.034592].
H versus G changes representation and crop policy; the component comparisons are F/G for representation and H/F for crop selection.

J changes only H's Dice reduction. Each positive image/class Dice loss is weighted by the inverse empirical training positive-pair frequency, then averaged over all B*C pairs. Empty Dice terms remain zero; BCE is unchanged.
Weights come from H's deterministic training crop exposure logs. J repeats that same crop schedule from the same seeded initialization; no learned H weights, validation labels or test labels enter its class priors.
J minus H macro Dice: -0.012818; paired-image 95% interval [-0.019847, -0.005762].
This is an ablation of established inverse-frequency reweighting, not a newly invented loss. The reported segmentation metrics remain unweighted four-class pixel-count metrics.

K extends J_balanced_dice_patch from its completed three-epoch state, restoring weights, Adam, AMP scaler and RNG. The only changed training setting is the total epoch cap (9); early-stopping patience remains two. Parent artifacts remain unchanged.
K shares its first three epochs with J_balanced_dice_patch; they were not trained twice. Additional continuation time: 2663.2 seconds. The budget table's total time for K includes the shared parent prefix.
A real-GPU two-update test verified identical losses, weights, optimizer and scaler with versus without an intervening save/reload. The longer budget was motivated by validation underfitting before test opening.
K minus its parent macro Dice: +0.102507; paired-image 95% interval [+0.087135, +0.116295].
K has a larger compute budget than B/C/E/F/G/H/J. Its score cannot establish a patching-only or loss-only benefit against those other configurations. The three-epoch comparisons remain the controlled component experiments.

## Actual training budgets

| Candidate | Epochs run | Selected epoch | Attempted updates | Successful updates | Run seconds (train + validation) |
|---|---:|---:|---:|---:|---:|
| B_whole | 3 | 2 | 3393 | 3393 | 852.4 |
| C_patch | 3 | 1 | 3393 | 3393 | 1391.9 |
| E_weighted_patch | 3 | 3 | 3393 | 3390 | 1090.2 |
| F_present_dice_patch | 3 | 3 | 3393 | 3392 | 1608.7 |
| G_present_dice_whole | 3 | 3 | 3393 | 3391 | 1571.1 |
| H_foreground_patch | 3 | 3 | 3393 | 3391 | 2410.5 |
| J_balanced_dice_patch | 3 | 3 | 3393 | 3391 | 1482.4 |
| K_extended_patch | 9 | 9 | 10179 | 10174 | 4145.7 |
Checkpoint selection can retain different epochs even with the same total search budget. Per-epoch histories disclose updates up to each selected checkpoint. Phase I used five epochs, batch size 2 and the original larger training pool; its compute/data budget differs.

C minus B macro Dice: **-0.032081**, paired-image bootstrap 95% interval [-0.037678, -0.026610].
This interval describes test-image sampling for fixed weights; only one training seed was run.

Stride-256 remains edge-aligned, so its last patches overlap. Uniform/Hann/stride ablations share C weights and C's locked thresholds.
Overlap changes total inference work and field-of-view multiplicity; compute differences are part of the measured tradeoff.

## Selected model: all four classes

| Class | Dice | IoU | Precision | Recall | FP pixels | FN pixels |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 0.1902 | 0.1051 | 0.1889 | 0.1915 | 262446 | 258159 |
| 2 | 0.0828 | 0.0432 | 0.0556 | 0.1618 | 186257 | 56799 |
| 3 | 0.4156 | 0.2623 | 0.3977 | 0.4352 | 7251284 | 6213328 |
| 4 | 0.2474 | 0.1412 | 0.2068 | 0.3078 | 2534520 | 1485825 |

Normal-image false alarms: 452 / 472 images, counting any predicted pixel of any class as an alarm. No postprocessing filter is applied.

| Class | GT-positive images | Images with zero recovered GT pixels |
|---|---:|---:|
| 1 | 72 | 11 |
| 2 | 20 | 2 |
| 3 | 412 | 2 |
| 4 | 64 | 0 |

The measured patch pipeline does not improve macro Dice over B on these fixed held-out samples.
The paired-image interval excludes zero for these fixed weights; this is not a multi-seed result.

## Size-aware error analysis

Components use 8-connectivity per class. Small <=1,024 pixels; medium <=8,192; large >8,192 at native resolution.
Object detection requires recovery of >=10% of its GT pixels. It is not an instance-IoU metric.

| Size | Class | GT components | Detected components | Object recall | Pixel recall |
|---|---|---:|---:|---:|---:|
| small | 1 | 139 | 63 | 0.4532 | 0.2435 |
| small | 2 | 1 | 0 | 0.0000 | 0.0000 |
| small | 3 | 132 | 108 | 0.8182 | 0.6679 |
| small | 4 | 18 | 7 | 0.3889 | 0.0734 |
| medium | 1 | 104 | 57 | 0.5481 | 0.1791 |
| medium | 2 | 21 | 12 | 0.5714 | 0.1629 |
| medium | 3 | 733 | 670 | 0.9141 | 0.5500 |
| medium | 4 | 56 | 35 | 0.6250 | 0.2615 |
| large | 1 | 1 | 0 | 0.0000 | 0.0668 |
| large | 2 | 0 | 0 | undefined | undefined |
| large | 3 | 279 | 255 | 0.9140 | 0.3937 |
| large | 4 | 67 | 52 | 0.7761 | 0.3151 |

## Research contribution and limits

The contribution is a controlled resolution-aware segmentation design and measured accuracy/runtime tradeoff in this system. Patching, blending and U-Net are established methods.
The coarse resized representation and native patch representation are compared; no simultaneous multi-scale feature fusion or new patching algorithm is claimed.
B/C hold architecture, optimizer, objective, initialization, training input pixel budget and source-image order fixed. Patch field of view and target exposure differ inherently.
Native geometry, overlapping probability reconstruction, batching and four-class alignment passed tests; masks and predictions remain in original coordinates.
Selected classes with zero recall: none. Evaluation completion does not imply industrial deployment readiness.
The initial three-epoch runs and one training seed are a limited compute budget; any longer continuation is separately identified above. Model-selection validation was reused for threshold tuning, so validation is optimistic; held-out test was excluded from both.
Attempted and skipped AMP updates are logged; any difference in successful updates is disclosed rather than assumed equal. CPU reference A overlapped part of C's training/validation, so training wall times are not an isolated speed benchmark. Final candidate inference timings were measured serially.
Class 2 has only 20 positive test images. Near duplicates and steel-coil leakage could not be excluded because acquisition-group metadata is unavailable.
Random patch training can miss a source image's defects. See per-epoch target-exposure records. False-positive/false-negative counts and both successful/failed full-resolution panels are retained.
`training_target_exposure.csv` records positive sample and pixel counts for every class/epoch. Native patches can present fewer positive samples but similar total foreground pixels at the matched training input-pixel budget.
Adaptive coarse-to-fine, pretrained encoders and uncontrolled optimizer/augmentation bundles were not introduced.

## Evidence and reproduction

`comparison.csv` includes all four class metrics. Candidate test folders include per-image errors and native-resolution panels (input / GT / prediction).
`training_curves.png`, `heldout_per_class.png`, `paired_uncertainty.json`, checkpoint hashes, split hashes and environment/package versions accompany these results.
`new_image_inference` contains execution on a previously unlabeled Kaggle steel image; this verifies inference, not accuracy without GT.
Follow `../README.md` for exact Windows commands, and `../SOURCE_AND_CONTRIBUTION_LOG.md` for sources and AI assistance.
The original files were verified unchanged after the experiments. Reproduction scripts are provided; a second end-to-end training replication was not run.
