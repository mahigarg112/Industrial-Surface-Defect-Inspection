# dice_loss_v1 Evaluation Report

## Summary

The separate BCE + soft Dice run completed all five configured epochs on CPU. Its best checkpoint is epoch 3, selected by **validation macro Dice at threshold 0.5**. Epochs 4 and 5 did not improve that score; patience was two epochs. Validation was used only for checkpoint selection and threshold tuning, never for gradient updates.

The new loss substantially improved aggregate validation results and class 3 detection. However, classes 1, 2, and 4 remain completely missed at the default threshold, and the default class 3 false-positive count increased. The model is suitable for further experiments, but **not for application integration**.

## Configuration and loss

The experiment kept the original U-Net architecture, 512 × 128 image dimensions, Adam optimizer, learning rate 0.001, seed 42, batch size 2, dataset split, and preprocessing. It added no class weights or sampling changes. The loss is:

`total = 0.5 × BCEWithLogitsLoss + 0.5 × Dice_loss`

`soft_Dice = (2 × sum(probability × target) + 1) / (sum(probability) + sum(target) + 1)`

`Dice_loss = 1 − soft_Dice`

The smoothing constant 1 keeps the calculation defined for empty masks. See `README.md` for the beginner-friendly explanation and checkpoint rule.

## Training history

| Epoch | Train loss | Train BCE | Train Dice loss | Validation loss | Validation BCE | Validation Dice loss | Val macro Dice @ 0.5 | Val macro IoU @ 0.5 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.26029 | 0.06312 | 0.45746 | 0.14698 | 0.05010 | 0.24386 | 0.09292 | 0.05706 |
| 2 | 0.14508 | 0.04947 | 0.24070 | 0.14206 | 0.04950 | 0.23461 | 0.10347 | 0.06524 |
| 3 | 0.14061 | 0.04585 | 0.23538 | 0.14006 | 0.04691 | 0.23321 | 0.11826 | 0.07745 |
| 4 | 0.12219 | 0.07234 | 0.17203 | 0.12313 | 0.09301 | 0.15325 | 0.00746 | 0.00379 |
| 5 | 0.10913 | 0.07504 | 0.14322 | 0.11504 | 0.06326 | 0.16682 | 0.11183 | 0.07202 |

Training and validation loss combine the two loss terms equally. Epoch 4 demonstrates why lower loss alone was not used to select the checkpoint: its validation loss decreased, but its macro Dice dropped sharply. Epoch 3 had the best validation macro Dice and was preserved.

Total measured training duration was **6,850.0 seconds (1 hour, 54 minutes, 10 seconds)**. The run used 10,054 training images and 2,514 validation images. No GPU was available.

## Validation results at default threshold 0.5

| Class | Dice | IoU | Precision | Recall | False-positive pixels | Missed defect pixels |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.0000 | 0.0000 | N/A | 0.0000 | 0 | 129,177 |
| 2 | 0.0000 | 0.0000 | N/A | 0.0000 | 0 | 27,727 |
| 3 | 0.4730 | 0.3098 | 0.4135 | 0.5525 | 3,130,734 | 1,787,699 |
| 4 | 0.0000 | 0.0000 | N/A | 0.0000 | 0 | 859,080 |
| **Macro** | **0.1183** | **0.0774** | — | — | **3,130,734** | **2,803,683** |

## Validation-only threshold tuning

The evaluator chose a separate threshold per class from a fixed candidate list by maximizing Dice on this validation split. Ties prefer the higher threshold. Since the same validation split chooses and scores these thresholds, these results may be optimistic and must remain labeled as validation results.

| Class | Selected threshold | Dice | IoU | Precision | Recall | False-positive pixels | Missed defect pixels |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.90 | 0.0000 | 0.0000 | N/A | 0.0000 | 0 | 129,177 |
| 2 | 0.90 | 0.0000 | 0.0000 | N/A | 0.0000 | 0 | 27,727 |
| 3 | 0.80 | 0.4803 | 0.3161 | 0.4771 | 0.4835 | 2,116,891 | 2,063,374 |
| 4 | 0.90 | 0.0000 | 0.0000 | N/A | 0.0000 | 0 | 859,080 |
| **Macro** | — | **0.1201** | **0.0790** | — | — | **2,116,891** | **3,079,358** |

Threshold tuning found a better class 3 balance at 0.80, reducing false positives while maintaining useful recall. It could not recover classes 1, 2, or 4.

## Comparison with the original baseline

Both checkpoints were evaluated on the same 2,514 validation images with the same evaluator and threshold candidates.

| Evaluation | Baseline macro Dice | dice_loss_v1 macro Dice | Baseline macro IoU | dice_loss_v1 macro IoU |
|---|---:|---:|---:|---:|
| Default threshold 0.5 | 0.0054 | **0.1183** | 0.0027 | **0.0774** |
| Validation-tuned thresholds | 0.0946 | **0.1201** | 0.0547 | **0.0790** |

At the default threshold, class 3 improved from Dice 0.0217 / IoU 0.0110 / recall 0.0110 to Dice 0.4730 / IoU 0.3098 / recall 0.5525. Its precision fell from 0.7227 to 0.4135, and false positives rose from 16,887 to 3,130,734 pixels. This is a clear recall gain with a substantial false-alarm cost. Classes 1, 2, and 4 remained completely missed at threshold 0.5.

With validation-tuned thresholds, class 3 improved from Dice 0.3124 / IoU 0.1851 / recall 0.3642 to Dice 0.4803 / IoU 0.3161 / recall 0.4835. Class 1's tuned Dice fell from 0.0115 to 0, class 2 remained at 0, and class 4's tuned Dice fell from 0.0545 to 0. The tuned model reduced class 3 false positives (3,866,217 to 2,116,891), but continued to miss all class 1, 2, and 4 pixels.

The validator also computes a common BCE + soft Dice validation loss for both checkpoints to keep its output schema consistent. The baseline was originally trained with BCE alone; therefore its original reported validation BCE loss (0.02925) is not directly comparable to dice_loss_v1's combined validation loss (0.14006).

## Conclusion and next step

BCE + soft Dice substantially helped class 3, showing that the baseline's background-dominated BCE objective was a major problem. It did not produce broad four-class defect detection. High false positives at threshold 0.5, three entirely missed classes, and validation-optimized thresholds mean this model is **not ready for application integration**. It is suitable for further controlled experimentation. A next study should investigate the rare classes and threshold generalization, then confirm results on an independent test set before any deployment decision.

## Artifacts

- Best trained checkpoint: `best_model.pt` (epoch 3; load verified; not a smoke checkpoint)
- Configuration snapshot: `run_config.json`; source config: `config.json`
- Per-epoch history and per-class default-threshold metrics: `training_history.csv`
- Run duration and checkpoint selection details: `training_run_info.json`
- Default and tuned full metrics: `evaluation/validation_metrics_default.json`, `evaluation/validation_metrics_tuned.json`
- Baseline comparison metrics: `baseline_comparison/validation_metrics_default.json`, `baseline_comparison/validation_metrics_tuned.json`
- New model validation visualization: `visualizations/validation_predictions.png`
- Baseline visualization for side-by-side review: `baseline_comparison/visualizations/validation_predictions.png`
- The parent baseline and smoke-checkpoint files were preserved.
