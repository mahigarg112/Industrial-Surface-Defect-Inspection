# Prediction Visualization Guide

## Current status

The visualization script is ready, but there is no valid image to display yet: only smoke-test weights exist. It deliberately refuses `smoke_checkpoint.pt`, and no prediction PNG was generated. This avoids showing one-step, effectively untrained outputs as model behavior. Matplotlib is not installed; the script uses Pillow and NumPy instead.

## After full training

Run from the project root in PowerShell:

```powershell
python experiments/severstal_segmentation/visualize_predictions.py `
  --checkpoint experiments/severstal_segmentation/results/best_model.pt `
  --count 8
```

The script samples validation images, preferring examples with each class and one defect-free image. It writes:

- `results/validation_predictions.png`: rows of examples with five panels: original image, ground-truth masks, predicted masks, ground-truth overlay, and prediction overlay.
- `results/visualization_manifest.json`: selected image IDs, ground-truth class IDs, predicted class IDs, checkpoint path, and color palette.

Class colors are class 1 red, class 2 green, class 3 blue, and class 4 yellow. Overlapping classes are blended in the mask color. Overlays retain the original image outside marked pixels. The script uses the checkpoint's image size, threshold, and base channel setting, and reads examples only from the validation split.

These pictures help explain localization errors: compare the predicted mask with the ground truth. Extra colored regions are false positives; missing ground-truth regions are false negatives. The visual samples are illustrative and do not replace split-wide metrics.
