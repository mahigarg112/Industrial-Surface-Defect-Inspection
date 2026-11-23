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

Original contribution in this project: isolated controlled whole-versus-native-patch comparison;
duplicate-aware new test partition preserving Phase I validation; weighted native reconstruction;
four-class and size-aware error/runtime evidence; explicitly locked model/threshold selection;
controlled reconstruction ablations. Overlapping patches, Hann windows and U-Net are established methods.
No pretrained encoder or adaptive inference region selector has been introduced. Foreground-conditioned crops affect training only. Claims of performance improvement
must be restricted to the measured matched evaluation; unrun experiments are not results.
