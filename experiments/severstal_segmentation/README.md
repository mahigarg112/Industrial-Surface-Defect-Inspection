# Severstal Steel Defect Segmentation Baseline

This is a new, isolated experiment for pixel-level segmentation. It does not alter NEU-CLS, either Phase I CNN, the existing `experiments/new_dataset/` experiment, or the original downloaded images. The official Kaggle test images are not used: their ground-truth masks are unavailable.

## Data and labels

The source directory is `data/severstal-steel-defect-detection/` and is expected to contain `train.csv` and `train_images/`. Each image has four independent binary defect masks represented by class IDs 1–4. The public training CSV identifies classes by ID; this experiment does not assign unofficial names. Images with empty annotations are retained as background-only examples.

## Preparation

From the project root, in PowerShell:

```powershell
python experiments/severstal_segmentation/prepare_data.py
```

This verifies annotation image IDs, decodes no source files in place, hashes images to group byte-identical files, and writes a fixed-seed 80/20 image-level train/validation manifest to `prepared/manifest.csv`. The validation split is for model selection and reporting only.

## Smoke test

```powershell
python experiments/severstal_segmentation/train.py --smoke
```

The smoke run loads two training samples, computes masks and logits, calculates binary cross-entropy, performs one optimizer update, then saves `results/smoke_checkpoint.pt`. This checkpoint is only evidence that the pipeline runs; it is not a trained model.

## Baseline training and validation

After reviewing the smoke output and CPU runtime estimate, launch the configured one-epoch baseline:

```powershell
python experiments/severstal_segmentation/train.py
python experiments/severstal_segmentation/validate.py
```

For an even smaller batch size (or to override the epoch count):

```powershell
python experiments/severstal_segmentation/train.py --epochs 1 --batch-size 1
```

Training saves its history, run configuration, and lowest-validation-loss checkpoint under `results/`. Validation reports per-class Dice/IoU and pixel TP/FP/FN for the validation split. Results are measured on resized masks. Do not compare them directly with Phase I image-classification accuracy.

## Design and preprocessing

The model is a small U-Net. It outputs four logits at each pixel; sigmoid plus binary cross-entropy treats the four class masks independently and allows overlap. Inputs and targets are resized together from 1600×256 to 512×128, retaining aspect ratio. RGB inputs are converted to `[0,1]` tensors. Masks use nearest-neighbor resizing so labels remain binary. No crop, padding, augmentation, or validation-image test evaluation is performed.

See `DATASET_PREPARATION.md`, `TRAINING_RESULTS.md`, and `MODEL_EXPLANATION_FOR_VIVA.md` for details.
