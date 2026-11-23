# Executed experiment log

All work is isolated under `phase2`; no GUI, presentation or website was produced.
The machine-readable status is `artifacts/STATUS.json`; this log does not declare completion.

| Stage | Actual evidence / decision |
|---|---|
| Repository audit | Correct pipeline: `experiments/severstal_segmentation`, four sigmoid channels, column-major RLE. No existing resolution-aware extension found in the searched implementation/documentation. Classification pipelines were excluded. |
| Preservation | SHA-256 snapshot of 507 original files; repeated verification found no changes. Images and original manifest are read-only inputs. |
| GPU setup | Default CPU PyTorch 2.15 preserved; separate Python 3.14 environment with PyTorch 2.11.0+cu128. RTX 3050 6 GB verified by real CUDA execution. |
| Dataset | Audited 12,568 native 1600×256 images and CSV/manifest agreement. No decoded-RGB exact duplicates. Frozen Phase II: 9,048 train, 2,514 original validation, 1,006 test. |
| Functional checks | Six geometry/RLE/count/batching tests, five threshold/component/objective tests, and three foreground-sampling tests: 14 functional tests currently pass. GPU whole/patch preflights each executed eight successful updates and checkpoint reloads. |
| Historical A | Original CPU runtime evaluation: Dice 0.1182569035, IoU 0.0774453452; reproduces recorded Phase I 0.1182569042 / 0.0774453431 to floating-point precision. CUDA cross-check retained separately. |
| B whole | Three epochs / 3,393 updates. Native validation Dice by epoch: 0.1008519, 0.1213525, 0.0. Best epoch 2. Classes 1/2/4 had zero recall at threshold 0.5. |
| C native patches | Same three-epoch budget / 3,393 updates. Native validation Dice: 0.0538405, 0.0, 0.0. Best epoch 1. Patching alone did not improve default-threshold validation under this budget. |
| Sampling evidence | In epoch 1, C contained class-2 positives in 33 samples versus B's 177, while positive pixel counts were similar (97,773 versus 94,716). Geometry inspection on validation showed correct placement and texture-related false positives. |
| E decision | Historical/new validation collapse motivates one objective change: positive-weighted BCE; unchanged soft Dice, architecture, optimizer, source order and patch sampling. Native training priors only; weights [1000, 1000, 37.87235, 184.13296]. |
| Weighted GPU gate | Eight successful updates in nine attempts, one AMP overflow skip followed by scale reduction/recovery. All four classes deliberately present in smoke targets. Full E training only started after this passed. |
| E outcome | Three epochs; default validation Dice 0.0503473, 0.0413191, 0.0589957. Selected epoch 3; validation-tuned Dice 0.0800660. Three skipped updates in 3,393 attempts. Recall rose but false positives were excessive; this did not solve four-class reliability. |
| F decision | A separate loss-component ablation retains original BCE and removes GT-empty image/class pairs from the soft-Dice average. This tests an explanation for background collapse; it does not mask evaluation images or combine the BCE weighting intervention. |
| F gate | Eight successful real-data GPU updates, zero skips, mixed positive/negative crops, all four classes represented. Four objective/evaluation unit tests plus six geometry tests passed. F training started after the gate. |
| F outcome | Three epochs; native default Dice 0.0910227, 0.0947596, 0.1051351. Selected epoch 3; one skipped AMP update in 3,393 attempts. Complete background collapse was avoided, but class 2 remained weak. |
| G decision/gate | Matching whole-image present-Dice counterpart, retaining all other training settings. Eight successful GPU updates, zero skips before full training. |
| H preparation | Three sampler tests verify rarest-class-positive selection, retention/alignment of other class channels, unchanged uniform default and unchanged negative sampling. GPU preflight/full results are separate artifacts when executed. |
| G outcome | Three epochs; default native validation Dice 0.0995632, 0.1302319, 0.1571062. Best epoch 3; tuned validation Dice 0.1617874. All four classes were nonzero, but class 2 Dice was only 0.0057068. Two skipped AMP updates in 3,393 attempts. |
| H interim evidence | First epoch foreground crops retained class-2 positives in 177 samples versus 33 in F/C; all four positive sample counts were [645,177,3520,577]. Epoch-2 default validation Dice 0.1334777 still had zero class-2 recall. These are validation observations, not held-out results. |
| J preparation | A fifth objective/evaluation test checks inverse positive-pair-frequency Dice reduction. J retains H's crops, initialization, BCE, optimizer and three-epoch budget; only Dice weighting/normalization changes. Weights are derived from completed H deterministic training exposure logs; a separate real-GPU preflight is required. No J accuracy result exists until its actual run artifacts are completed. |
| H outcome | Three epochs; default native validation Dice 0.1028103, 0.1334777, 0.1401044. Best epoch 3; tuned Dice 0.1508508. Two skipped updates in 3,393 attempts. All four classes nonzero at default threshold, but class-2 Dice was only 0.0087474. |
| J gate/start | Frozen training-only Dice weights [14.027907,51.118644,2.580964,15.681109] from 27,144 H training draws. Eight real-data GPU updates passed with finite gradients and zero skips. Three-epoch J training started only after this gate passed. |
| Longer-budget plan | Recorded before J epoch-3 validation: J default Dice rose from 0.0839591 to 0.1114176 over its first two epochs; class-2 Dice rose from 0.0033265 to 0.0146875. Extend J to a total nine-epoch cap if it has not already met unchanged patience-2 early stopping; H is the documented fallback if it has. This tests training budget separately, preserving the completed three-epoch comparisons. An exact-parent save/reload equivalence GPU gate must pass first. No held-out scores entered this plan. |
| J outcome | Three epochs; default native validation Dice 0.0839591, 0.1114176, 0.1189352. Selected epoch 3; class-2 Dice 0.0109533. Two skipped AMP updates in 3,393 attempts. The three-epoch class-balancing intervention did not outperform H on default macro Dice. |
| Resume gate correction | Initial equivalence check failed because independent test paths reused a loaded Adam dictionary whose CPU step tensors shared storage with each restored optimizer. A direct storage-pointer check confirmed this. Deep-copying the test's source optimizer dictionary isolated the paths; real training artifacts were not changed and continuation had not started. |
| Resume gate passed | Two successful real-GPU updates had exactly matching losses [0.594646156,0.565181017], model weights, Adam state and AMP scaler with versus without an intervening save/reload. Original J state SHA-256 remained unchanged. |
| K start | Extend J from epoch 3 to a total cap of 9, unchanged patience 2. All original J evidence stays intact. K restores the last training weights/Adam/scaler/RNG and retains J's best checkpoint as the initial selection reference. Extra compute is disclosed separately from the matched three-epoch experiments. |
| K outcome | Completed the nine-epoch cap. Default native validation Dice at epochs 4–9: 0.1142840, 0.1348211, 0.1342475, 0.1594569, 0.1341191, 0.1803962. Best epoch 9; IoU 0.1063137; per-class Dice [0.0988558,0.0347669,0.4101695,0.1777925]. Total 10,179 attempted / 10,174 successful updates, five AMP skips. Additional continuation time 2,663.24 seconds; total including shared J prefix 4,145.67 seconds. Class 2 remained weak, so no industrial reliability claim is made. |
| Training boundary | No further training planned after K. Finish validation threshold tuning, compare all candidates, lock selection, then open held-out test. None of the training decisions above used held-out model performance. |
| Validation selection | All controlled-condition checks passed. Tuned native validation macro Dice: B 0.1217797, C 0.0882805, E 0.0800660, F 0.1188392, G 0.1617874, H 0.1508508, J 0.1325860, K 0.2392288. K leads; selected four thresholds [0.95,0.975,0.5,0.95] (stored float32 values govern exact comparisons). The original test had not been evaluated at selection. Patching alone under the matched three-epoch protocol did not improve validation Dice. K differs in objective, sampler and training budget and cannot establish a patching-only benefit. |

Later stages — remaining training results, validation threshold selection, final checkpoint locking, held-out test,
reconstruction ablations and new-image inference — are recorded in their machine-readable artifacts
and the generated `artifacts/RESEARCH_REPORT.md` when actually completed. No test results informed
the decisions above. Old Phase I weights are contaminated on the newly carved test and are excluded
from held-out generalization comparisons.

CPU reference A ran concurrently with part of C's training/validation. Training wall times include
that contention and are not an isolated throughput benchmark. Final inference timings are measured
serially after warmup with synchronized GPU completion.
