> PORTABLE COPY: read START_HERE.txt first. Create a new .venv on this laptop; do not copy an old virtual environment. This package is for the already authorized private faculty demonstration. See REPRODUCIBILITY_LIMITATIONS.txt.

> Final freeze: K remains official; L/N/O/P are research comparisons only. Read FINAL_CLEAN_RELEASE_STATUS.txt, CODE_NAVIGATION_FOR_VIVA.txt and CLEANUP_SUMMARY.txt. Current consolidated results are in FINAL_RESEARCH_SUMMARY.txt. Older QA output paths removed as duplicates resolve in the documented external release-history archive; current app/training/evaluation sources remain standalone.

> Current GUI: start with **SIMPLE_DEMO_COMMANDS.txt** and **SIMPLE_GUI_VERIFICATION.txt**. The main screen is simplified; threshold controls and individual mask downloads are collapsed, with no sidebar/model switching. K weights, default thresholds and native predictions remain preserved. Earlier GUI verification records below describe their dated versions.

# Industrial Surface Defect Inspection — Phase II clean release

Local four-channel semantic segmentation of Severstal steel images. Prepared for the **individual evaluation on Monday, 23 November 2026: 8 minutes demonstration + 4 minutes viva**. This release reuses the verified implementation and evidence; cleanup does not retrain or select another model.

## Run locally

From PowerShell on the current machine:

```powershell
# Run from the extracted project root (folder containing START_HERE.txt).
$env:SDI_PHASE2_OUTPUT = $null
.\.venv\Scripts\python.exe -B -m streamlit run phase2\release\app.py --server.address 127.0.0.1 --server.port 8501 --server.headless true --browser.gatherUsageStats false --server.maxUploadSize 20 --client.toolbarMode minimal
```

Open <http://127.0.0.1:8501>. Upload JPG/JPEG/PNG or load a bundled image, click **Inspect image**, show the four masks, adjust thresholds, and download the overlay, individual mask PNGs and JSON. Disable **Adjust thresholds** to restore the exact locked settings. Stop with Ctrl+C. If port 8501 is occupied, choose 8502 and its matching URL.

The interpreter above supplies installed dependencies. **No source, weights or image corpus in the parent repository are required for inference.** There are no source symlinks/junctions. `phase2/common.py` resolves paths from this release's own location. The two sample images are already bundled. The model is a small independent physical copy (about 1.88 MiB), not a hard link to original weights.

## Five code files to open first

1. `phase2/model.py`: actual `SmallUNet.forward`, encoder, bottleneck and decoder.
2. `phase2/patches.py`: `boxes`, `weight_window`, `predict`; native tiling and reconstruction.
3. `phase2/release/inference.py`: validation, `Inspector`, thresholds, overlay and exports.
4. `phase2/data.py`: aligned RLE masks and foreground-aware training crops.
5. `phase2/train.py`: Adam, loss dispatch, backpropagation, validation and checkpoint selection. Its final K loss is `phase2/objectives.py:balanced_present_dice_loss`.

Application entry: `phase2/release/app.py`. RLE: `phase2/dataset_utils.py:decode_rle`. Metrics: `phase2/metrics.py`; threshold tuning/component evaluation: `phase2/evaluate.py`. Configuration: `phase2/artifacts/K_extended_patch/config.json`.

## Preparation documents

- `ARCHITECTURE_AND_CODE_FLOW.txt`: all 15 execution steps, verified layer table, architecture diagram reference and experiments.
- `MATHEMATICS_FOR_VIVA.txt`: formulas, symbols, examples and actual code links.
- `8_MINUTE_DEMO_SCRIPT.txt`: timed spoken demonstration and existing-slide recommendations.
- `FACULTY_VIVA_QA.txt`: 30 short answers.
- `LIVE_CODE_CHANGES.txt`: three safe exercises with tests and reversion.
- `CLEANUP_MANIFEST.txt`: keep/exclude/unknown inventory and exact packaging boundaries.
- `CLEAN_PROJECT_VERIFICATION.txt`: current tests, scope and limitations.

## Model and evidence

Small U-Net: **482,788 trainable parameters**, base width 16; no pretrained encoder. Native RGB patches `N×3×256×256` produce `N×4×256×256` logits. Sigmoid is outside the network. Twelve stride-128 patches cover a 256×1600 image; positive Hann probability blending reconstructs four native masks. Independent thresholds allow overlapping labels; background is implicit.

Selected model: `phase2/artifacts/final/model.pt`, candidate `K_extended_patch`, best epoch 9. SHA256:

```text
0ce6c174d27fff24551faa7ce94f7a2ec7f9a98294a0eac7746a8c61b89eed59
```

Exact thresholds in `final/lock.json`: `[0.949999988079071, 0.9750000238418579, 0.5, 0.949999988079071]`. They were selected using validation, before test opening. GUI adjustments are exploratory and do not replace them.

| Common native held-out protocol, 1,006 images | Macro Dice | Macro IoU |
|---|---:|---:|
| B: resized whole-image, 3 epochs | 0.123897 | 0.082356 |
| C: matched patch-only, 3 epochs | 0.091816 | 0.054641 |
| K: changed loss, foreground crops, 9 total epochs | 0.233996 | 0.137937 |

Patching alone underperformed. K's aggregate improvement cannot establish a patching-only effect. Historical Phase I Dice 0.118257 / IoU 0.077445 uses resized original validation and is a different protocol. Phase I trained on images later reserved as Phase II test, so its old weights cannot provide an independent baseline on that test.

Read `MODEL_CARD.md`, `FINAL_RESEARCH_SUMMARY.txt`, `phase2/artifacts/RESEARCH_REPORT.md`, `comparison.csv`, per-run `config.json`, `history.csv`, `run_info.json`, `thresholds.json`, `test/metrics.json`, exposure logs and `ablations/`. Training console logs are retained. Existing `architecture.png` and `architecture_verified.json` are reused; no generic replacement diagram is generated. The immutable pre-test source remains in `phase2/artifacts/final/source/`.

K class Dice: 0.190200, 0.082752, 0.415618, 0.247413. Any-pixel alarms on **452/472 GT-empty images** are a serious limitation. Upload accuracy is unknown without ground truth. Single seed, small budgets, rare classes, acquisition-group uncertainty and no external-domain evaluation limit generalization. This is a research prototype.

## Data and training protocol

12,568 labeled RGB images, normally 256×1600; one-based column-major RLE yields four masks. Frozen `partition.csv`: **9,048 train / 2,514 validation / 1,006 test**. Original validation is preserved; seed-42 stratified exact decoded-RGB duplicate groups split original training into new train/test. No exact RGB duplicates were found in the recorded audit. Near duplicates and steel-coil grouping were not established. Partition SHA256: `97135138fcb09211c850dc2a16e0ae1deadd3088d342fdccf0c9f9d6b04559f0`.

Images become RGB float32 /255, CHW. No ImageNet normalization or training augmentation is configured. Whole mode resizes image bilinearly and masks with nearest neighbor to 128×512. Patch mode keeps 256×256 native pixels. K selects a random grid crop containing the source image's rarest present class; negative sources retain uniform sampling. Each source contributes one sample per epoch. Inference exhaustively inspects all patches without labels.

K: Adam lr 0.001, batch 8, workers 2, seed 42, GPU FP16 forward / FP32 loss with GradScaler, patience 2, no scheduler or gradient clipping. Loss is 0.5 unweighted BCE + 0.5 inverse-training-positive-pair-frequency Dice, with GT-empty Dice terms zero and mean over all batch×class pairs. Weights: `[14.027906976744186, 51.11864406779661, 2.580964153275649, 15.68110918544194]`; smoothing 1. K restores J's first three epochs, Adam/scaler/RNG and continues to nine total epochs. It is one continuation, not nine fresh epochs after J.

Eight completed run directories represent seven fresh starts plus K continuation. Choose epochs on full native validation macro Dice at 0.5, first tie wins; tune class thresholds on validation only, higher threshold wins ties. Lock candidate by tuned validation macro Dice, then evaluate test. Test is now opened and cannot select further improvements while retaining an untouched-test claim. Evaluation sums int64 TP/FP/FN over the split per class, then takes four-class macro means; it does not average image Dice.

## Verification and CLI

```powershell
# From this clean folder; no dataset link required.
.\.venv\Scripts\python.exe -B -m unittest phase2.release.test_release phase2.test_pipeline phase2.test_evaluation phase2.test_sampling -v
.\.venv\Scripts\python.exe -B verify_clean.py
# Use a fresh output directory:
.\.venv\Scripts\python.exe -B -m phase2.release.inference --image phase2\release\demo_inputs\0007a71bf.jpg --device cpu --output verification\my_cli_demo
```

`verify_clean.py` writes fresh integration records, leaf-layer hooks, CPU PNG/JSON exports and HTTP startup results under `verification/run_*/`. It also checks existing metrics arithmetically without full-dataset inference, routing output away from historical records. Unit tests exercise real Streamlit controls and cached inference. Optional installed-Edge browser harness: `phase2/release/browser_smoke.py`, run against a server on port 8501 with a fresh `--evidence-dir` under its `evidence/` folder. Browser automation is not an app dependency.

`RELEASE_MANIFEST_SHA256.json` records release integrity, excluding itself and later-generated files. It is not a signature. `verification/ORIGINAL_PRESERVATION_BASELINE.json` records originals before cleanup; the final preservation result is in verification. No original files were deleted or edited.

## Reproduce on another machine

Recorded environment: Windows, Python 3.14.5, torch 2.11.0+cu128, Streamlit 1.67.0, NumPy 2.4.6, Pillow 12.2.0, SciPy 1.18.1, Matplotlib 3.11.2; RTX 3050 Laptop 6 GiB. See `RELEASE_ENVIRONMENT.json` and `requirements-release-lock.txt`. `torchvision` is not used.

```powershell
# From the clean folder; set up before the presentation.
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.11.0+cu128 --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install -r requirements-release-lock.txt
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -B -m streamlit run phase2\release\app.py --server.headless true
```

For CPU-only, install `torch==2.11.0` from `https://download.pytorch.org/whl/cpu`, then `requirements-app.txt`; use that local interpreter in all commands. CPU/GPU rounding can differ slightly. No dependency installation was performed during cleanup.

Full research reproduction requires an explicitly supplied authorized Severstal image corpus and `train.csv`. The app does not require them. Use `LINK_DATA_FOR_REPRODUCTION.ps1 -DatasetRoot 'D:\authorized\severstal-steel-defect-detection'` to create a documented optional junction, or place the data under this release's `data/severstal-steel-defect-detection`. This is an explicit data dependency for training, never a hidden source dependency. Do not copy the full corpus for the demo.

Follow the full eight-run commands in `phase2/README.md` from this folder, using a **fresh** `SDI_PHASE2_OUTPUT` such as `phase2/reproduction_001`. Audit/preflight regenerates gates against the fresh partition; do not reuse gates from another partition. Keep all settings and J→K continuation steps. `phase2/train.py` refuses existing run directories and requires CUDA. Historical Phase I reevaluation uses its separately documented CPU runtime. Selected historical Phase I weights and manifest are included; other candidate checkpoints are referenced, not duplicated. Saved metrics for all runs are included. Re-evaluating another candidate requires its explicitly referenced checkpoint or retraining. No training is needed for cleanup/viva.

## Sources, assistance and official requirements

`SOURCE_AND_CONTRIBUTION_LOG.md` records inherited Phase I architecture/RLE, established U-Net/tiling/Hann/class-weighting methods, research references and substantial Codex assistance. Describe the work you personally understand, reviewed, executed and can defend; do not claim unaided authorship of generated code. Review/sign `ORIGINALITY_DECLARATION_TEMPLATE.txt` personally.

Both official evaluation PDFs were found locally and copied into `docs/`. They permit disclosed AI support, recommend 5–8 slides, and emphasize independent mathematical/implementation understanding. This does not certify your personal understanding or satisfy an unavailable separate Project02 brief. `RUBRIC_REVIEW.txt` distinguishes confirmed evaluation expectations from unresolved project-specific requirements. C++/Java integration is not implemented; whether required remains unknown. Private faculty demonstration approval for the existing samples is recorded in prior project evidence; public redistribution/licensing is not established. Existing research references are retained without new literature or performance claims.
