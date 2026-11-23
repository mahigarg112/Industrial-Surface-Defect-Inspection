# Source and Contribution Log

Prepared 2026-11-19; research continued 2026-11-20. This records sources and assistance; it does not assert faculty approval or a new algorithm.

| Source | Actual use / contribution |
|---|---|
| Existing `experiments/severstal_segmentation/model.py` | SmallUNet imported unchanged; state-dict compatibility retained. Original provenance remains subject to existing project disclosures. |
| Existing `experiments/severstal_segmentation/dataset.py` | Column-major RLE decoder reused and independently tested; original dataset loader reused for historical A evaluation. |
| Existing `dice_loss_v1` configuration/checkpoint/results | Historical Phase I reference; original validation and recorded scores preserved. |
| Existing `train.py` loss | Equal BCE + per-image/class soft Dice with smoothing 1; new AMP-safe FP32 reductions. |
| Phase II present-class Dice ablation | An explicit reduction variant of the existing soft-Dice formula, implemented with AI assistance to test empty-target behavior. Original BCE remains; this is not claimed as a newly invented loss function. |
| Phase II balanced positive-pair Dice ablation | Inverse empirical training-pair-frequency weighting of the existing Dice terms, with unchanged BCE. Implemented with AI assistance; only deterministic training target counts set the weights. This is a controlled class-balancing experiment, not a claim of inventing class weighting or Dice. |
| [Severstal dataset](https://www.kaggle.com/competitions/severstal-steel-defect-detection/data) | Existing locally supplied images/annotations; no additional data downloaded, no redistribution performed. |
| [U-Net, Ronneberger et al. 2015](https://arxiv.org/pdf/1505.04597) | Architectural background and established overlap-tile prior art (paper section 1/Fig. 2); no external implementation or pretrained weights downloaded. |
| [PyTorch official installation guide](https://pytorch.org/get-started/locally/) | Consulted for CUDA/Windows installation; actual compatible wheel verified against official CUDA 12.8 index. |
| [PyTorch 2.11 AMP docs](https://docs.pytorch.org/docs/2.11/amp.html) | Reference for mixed precision; runtime smoke checks determine support. |
| [PyTorch BCEWithLogitsLoss docs](https://docs.pytorch.org/docs/2.11/generated/torch.nn.BCEWithLogitsLoss.html) | Positive-class weighting and per-class broadcasting background for the separately evaluated E objective. The cap of 1,000 is a study design choice, not a library default or claimed optimum. |
| [Official nnU-Net trainer](https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunetv2/training/nnUNetTrainer/nnUNetTrainer.py) | Consulted 2026-11-20 as foreground-sampling prior art. No nnU-Net package, implementation or weights used. H's rarity-first existing-grid policy is this project's controlled implementation of an established sampling idea. |
| NumPy / Pillow / SciPy / Matplotlib | Numerical operations, image loading, connected components, exportable plots. Exact installed versions are frozen. |
| OpenAI Codex AI assistance | Repository/environment audit; new Phase II tiling, split, test, training/evaluation/inference code and documentation. The student must understand, review and disclose this assistance according to faculty policy. |
| Post-evaluation visual review and runtime analysis | AI-assisted inspection of saved four-class examples, explicit class-confusion/false-alarm notes, a clearer chart and an interleaved timing benchmark. These were performed after locking/evaluation and changed no model, threshold, partition or test metrics. Archived post-review source/provenance are separate from the pre-test source snapshot. Visual notes are bound to the reviewed checkpoint and image IDs; a different reproduction needs its own visual review. |

Original contribution in this project: isolated controlled whole-versus-native-patch comparison;
duplicate-aware new test partition preserving Phase I validation; weighted native reconstruction;
four-class and size-aware error/runtime evidence; explicitly locked model/threshold selection;
controlled reconstruction ablations. Overlapping patches, Hann windows and U-Net are established methods.
No pretrained encoder or adaptive inference region selector has been introduced. Foreground-conditioned crops affect training only. Claims of performance improvement
must be restricted to the measured matched evaluation; unrun experiments are not results.


## Finalization sprint additions

New phase2/release source reuses the preserved SmallUNet, strict loader and patches.predict. Codex AI assistance contributed to shared input/inference/export utilities, Streamlit UI, unit/AppTest/browser checks, package assembly and documentation. No new architecture, dataset, training, threshold selection or accuracy measurement was introduced. Streamlit and required dependencies were installed only in .venv-phase2; the original research lockfile remains unchanged. Browser QA used installed Microsoft Edge and websockets, not downloaded model weights. Failed browser-harness attempts and Windows restrictions are recorded in QA_HISTORY.txt. Faculty AI/data permissions, originality/ownership decisions, possible mandatory C++/Java work and redistribution rights remain unresolved. The student must review and sign declarations personally.


## Clean release and viva preparation ? 20 November 2026

Codex assisted clean assembly, direct local model/RLE imports, four GUI mask downloads, naming the existing overlay alpha, focused tests and viva documents. SmallUNet and dataset utilities were copied byte-for-byte to `phase2/model.py` and `phase2/dataset_utils.py`. Research/model/threshold evidence and immutable snapshots were preserved. No training or full-dataset inference was performed. Cleanup does not establish unaided student authorship; personal understanding and review remain the student?s responsibility.

The two official evaluation PDFs in Downloads were read and copied to `docs/`. They permit disclosed AI support and prioritize mathematics/individual defense. The full separate Project02 brief remains unavailable. General AI support permission is now documented; individual compliance, signature and public redistribution rights are not certified. Existing references remain; no new literature ranking, licensing permission or faculty acceptance is asserted. The earlier Phase I deck was inspected only for a shorter selection recommendation; no PowerPoint changed.

Active source differs from the pre-test snapshot only for documented packaging/UI details. `verification/COPY_PROVENANCE.json` records original source hashes; `RELEASE_MANIFEST_SHA256.json` covers the adapted release. Historical checkpoint references are optional reproduction evidence and never inference-time source dependencies.


## Simple academic GUI update

Codex assisted the minimal Streamlit layout, neutral prediction/empty-state text, requested research-only notice, collapsed threshold/download controls, explicit EXIF orientation handling and focused tests/screenshots. The original GUI sources are preserved under the parent project?s `phase2/precision_balance/gui_before`; original `phase2/release` and the prior submission are unchanged. No model weights, default thresholds or historical research metrics were changed by this presentation update. Actual K masks for both existing samples matched the earlier renderer on CPU/CUDA. Synthetic empty/overlap/orientation unit fixtures are tests, not fabricated demonstration predictions.

A separately authorized, single loss-balancing experiment is recorded outside this clean GUI under `phase2/precision_balance`. It does not select or replace this app?s checkpoint. The experiment changes only BCE/Dice coefficients relative to a documented matched reference; its validation-only results and assistance are disclosed separately.


Final freeze: Codex assisted reference/import classification,candidate-release testing,release-only duplicate cleanup,viva navigation and consolidation of saved L/N/O/P evidence. No retraining,model/threshold change,new performance estimate or unaided-authorship claim. Original and selected checkpoints,Phase I/II research and existing assistance disclosures remain preserved.
