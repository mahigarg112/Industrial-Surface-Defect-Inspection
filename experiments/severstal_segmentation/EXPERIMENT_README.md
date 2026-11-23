# Experiment Guide

The Severstal dataset provides pixel masks, so this folder implements a separate four-channel segmentation baseline. The U-Net produces a binary mask per CSV class ID (1–4); it does not reuse the original six-class image classifier. The Kaggle test images are excluded because their ground-truth labels are unavailable.

## Exact commands

Run from the project root in PowerShell:

```powershell
python experiments/severstal_segmentation/prepare_data.py
python experiments/severstal_segmentation/train.py --smoke
python experiments/severstal_segmentation/train.py
python experiments/severstal_segmentation/validate.py
```

The first command validates paths, groups exact duplicate images, and makes the reproducible train/validation manifest. Smoke test performs one update and writes a clearly marked smoke checkpoint. Full training uses the conservative one-epoch, batch-size-2 configuration in `config.json`. Validation reads only the labeled validation split. Add `--epochs 1 --batch-size 1` to use an even smaller batch.

## Outputs

- `prepared/manifest.csv` and `prepared/preparation_summary.json`
- `results/smoke_checkpoint.pt` from the smoke command
- `results/training_history.csv`, `results/run_config.json`, and `results/best_model.pt` from training
- `results/validation_metrics.json` from validation

Training loss is unweighted binary cross-entropy over four masks. Metrics are Dice and IoU per class, plus their class macro average. Input and target dimensions are 512×128, resized from 1600×256 with bilinear and nearest-neighbor interpolation respectively. See `DATASET_PREPARATION.md` and `MODEL_EXPLANATION_FOR_VIVA.md` for the design and limitations. Do not report metrics until the corresponding run has completed and the output file exists.
