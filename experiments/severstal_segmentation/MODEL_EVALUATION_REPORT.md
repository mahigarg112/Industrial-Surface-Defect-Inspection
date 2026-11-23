# Severstal Model Training and Evaluation Report

## Status

Training completed successfully for **one configured epoch** on the training split. Validation used only the held-out validation split. This is an actual trained checkpoint, not the smoke-test checkpoint. However, its validation segmentation quality is very poor, so it is **not ready for application integration**.

The run used seed 42, batch size 2, 512 × 128 resized images, base channel width 16, Adam with learning rate 0.001, and binary cross-entropy with logits. The machine had a CPU-only PyTorch build (2.15.0+cpu; CUDA unavailable). The epoch took **946.2 seconds (15 minutes 46.2 seconds)**. The manifest contains 10,054 training images and 2,514 validation images.

## Training history

| Epoch | Training loss | Validation loss | Macro Dice | Macro IoU |
|---:|---:|---:|---:|---:|
| 1 | 0.048009 | 0.029247 | 0.005424 | 0.002742 |

Loss is the average binary cross-entropy across mask pixels and output classes. A low loss by itself does not mean defect segmentation is good: most pixels are background, so predicting background can still keep the average loss low.

## Validation results

Metrics were calculated at threshold 0.5 by accumulating pixel counts over all 2,514 validation images, then calculating per-class scores. The four class IDs are the numeric Severstal CSV classes; the local dataset does not provide semantic names for them.

| Class ID | Dice | IoU | Precision | Recall |
|---:|---:|---:|---:|---:|
| 1 | 0.0000 | 0.0000 | N/A (no predicted pixels) | 0.0000 |
| 2 | 0.0000 | 0.0000 | N/A (no predicted pixels) | 0.0000 |
| 3 | 0.0217 | 0.0110 | 0.7227 | 0.0110 |
| 4 | 0.0000 | 0.0000 | N/A (no predicted pixels) | 0.0000 |
| **Macro average** | **0.0054** | **0.0027** | — | — |

The model predicted no pixels for classes 1, 2, and 4 across validation. For class 3, most pixels it did predict were correct (precision 72.3%), but it found only about 1.1% of actual class-3 defect pixels. In plain terms, it misses nearly every defect. The macro Dice and IoU confirm that the predictions do not overlap the ground truth well.

## Visual inspection

The saved comparison sheet shows the original validation images, ground-truth masks, predicted masks, and overlays. The predicted-mask column is mostly empty, matching the poor recall scores. These are model outputs from the saved checkpoint, not fabricated examples.

## Limitations and recommended next work

One epoch is a pipeline completion run, not enough training to claim a useful model. The dataset is heavily imbalanced: 5,902 of 12,568 images have no defect mask, and class 2 has only 247 positive images overall. The current plain binary cross-entropy objective gives background pixels a strong influence; this is a likely reason for the model's mostly empty predictions. That is an explanation consistent with the evidence, not proof of the sole cause.

The next training iteration should address imbalance with a validated class-weighted or focal/Dice-based loss, train for multiple epochs with early stopping on validation metrics, and track per-class recall/Dice. Keep the validation split unchanged while tuning. Reassess the resulting per-class metrics and prediction sheet before considering application integration. Do not present this checkpoint as application-ready.

## Reproducibility and artifact paths

- Best checkpoint: `results/best_model.pt` (epoch 1, validation loss 0.0292467; load verified; not a smoke checkpoint)
- Configuration snapshot: `results/run_config.json`
- Epoch history: `results/training_history.csv`
- Actual duration and run summary: `results/training_run_info.json`
- Validation metrics: `results/validation_metrics.json`
- Loss curve: not generated because matplotlib is not installed; CSV history is present
- Validation visualization: `results/validation_predictions.png`
- Selected visualization examples: `results/visualization_manifest.json`
- The original `results/smoke_checkpoint.pt` is preserved and remains explicitly marked as a smoke test.

All metric values above come from the saved training history or full validation evaluation. No Kaggle test images were used.
