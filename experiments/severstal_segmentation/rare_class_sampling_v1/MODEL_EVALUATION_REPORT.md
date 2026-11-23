# rare_class_sampling_v1 Evaluation Report

## Summary

The controlled sampling experiment completed five epochs. The best checkpoint is epoch 3, selected by validation macro Dice at threshold 0.5. It did **not** improve default-threshold detection for classes 1, 2, or 4: all three remain undetected. Compared with `dice_loss_v1`, class 3 false positives decreased, but class 3 recall and overall macro Dice also decreased. Rare-class sampling did not meet its primary objective.

The result is suitable for further study, but it is not ready for application integration.

## Sampling algorithm

Each epoch has exactly 10,054 training draws. The sampler uses every one of the 5,332 defect-containing source images once, then selects 1,203 of those images for one additional draw. The extra draw is sampled without replacement, so a source image can appear at most twice. It selects 3,519 defect-free images uniformly without replacement from 4,722 available, then shuffles the combined list. Validation sampling stays unchanged.

For class `c`, `N_c` is the number of unique training images containing that class. The rarity factor is `N_c^(-0.5)`. For an image with active classes `S`, its weight is the **average**, not sum, of those factors:

`image_weight = sum(N_c^(-0.5) for c in S) / number_of_active_classes`

This favors images with rarer classes while preventing an image with multiple labels from getting an inflated weight just because it has more labels. The extra-draw weights for classes 1–4 were respectively 0.03735, 0.07125, 0.01558, and 0.03950. Class 2 therefore received the largest per-image rarity factor.

The class draw counts below count each label on a multi-label image; their sum can exceed the number of defect-positive draws.

| Epoch | Training draws | Positive draws | Defect-free draws | Class 1 draws | Class 2 draws | Class 3 draws | Class 4 draws | Unique images used | Repeats beyond first use | Max appearances |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 10,054 | 6,535 | 3,519 | 961 | 306 | 4,855 | 846 | 8,851 | 1,203 | 2 |
| 2 | 10,054 | 6,535 | 3,519 | 947 | 308 | 4,865 | 860 | 8,851 | 1,203 | 2 |
| 3 | 10,054 | 6,535 | 3,519 | 955 | 299 | 4,851 | 872 | 8,851 | 1,203 | 2 |
| 4 | 10,054 | 6,535 | 3,519 | 971 | 314 | 4,836 | 847 | 8,851 | 1,203 | 2 |
| 5 | 10,054 | 6,535 | 3,519 | 976 | 306 | 4,854 | 844 | 8,851 | 1,203 | 2 |

Every epoch met the repetition cap. All 5,332 positive source images were used once per epoch; 3,519 of the 4,722 clean images were selected without replacement per epoch.

## Training setup and history

The experiment preserved the `dice_loss_v1` U-Net, BCE + soft Dice loss, 512 × 128 preprocessing, Adam optimizer, learning rate 0.001, batch size 2, seed 42, and train/validation split. No class-weighted loss or validation sampling changes were added. Early stopping patience was two epochs; checkpoint selection was highest validation macro Dice at threshold 0.5.

| Epoch | Train loss | Train BCE | Train Dice loss | Validation loss | Validation BCE | Validation Dice loss | Val macro Dice @ 0.5 | Val macro IoU @ 0.5 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.26760 | 0.06920 | 0.46599 | 0.14555 | 0.04660 | 0.24449 | 0.08889 | 0.05405 |
| 2 | 0.14997 | 0.05753 | 0.24241 | 0.14111 | 0.04649 | 0.23572 | 0.10823 | 0.06906 |
| 3 | 0.14688 | 0.05503 | 0.23873 | 0.13970 | 0.04698 | 0.23242 | **0.11279** | **0.07282** |
| 4 | 0.14413 | 0.06770 | 0.22056 | 0.12065 | 0.08162 | 0.15968 | 0.05421 | 0.03040 |
| 5 | 0.12776 | 0.08481 | 0.17072 | 0.12091 | 0.08257 | 0.15926 | 0.01702 | 0.00881 |

Training loss decreased each epoch, and validation loss also decreased overall. However, threshold-0.5 validation Dice peaked at epoch 3 and then fell sharply. This is a sign of validation performance instability and possible overfitting or probability calibration drift; it is not conclusive proof of classic overfitting because validation loss did not rise.

Total training duration was **4,783.7 seconds (1 hour, 19 minutes, 44 seconds)** on CPU. The best checkpoint is epoch 3 and is not a smoke checkpoint.

## Default threshold 0.5 validation metrics

| Class | Dice | IoU | Precision | Recall | False-positive pixels | Missed defect pixels |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.0000 | 0.0000 | N/A | 0.0000 | 0 | 129,177 |
| 2 | 0.0000 | 0.0000 | N/A | 0.0000 | 0 | 27,727 |
| 3 | 0.4511 | 0.2913 | 0.4996 | 0.4113 | 1,645,987 | 2,352,014 |
| 4 | 0.0000 | 0.0000 | N/A | 0.0000 | 0 | 859,080 |
| **Macro** | **0.1128** | **0.0728** | — | — | **1,645,987** | **3,367,998** |

At threshold 0.5, classes 1, 2, and 4 still have zero recall. Class 3 precision improved compared with `dice_loss_v1`, while its recall declined.

## Validation-only tuned thresholds

The evaluator chooses each class threshold by maximum Dice over a fixed grid from 0.00001 through 0.9; ties prefer the higher threshold. These are validation-tuned results and may be optimistic because the same validation split selects and scores the thresholds.

| Class | Threshold | Dice | IoU | Precision | Recall | False-positive pixels | Missed defect pixels |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.90 | 0.0000 | 0.0000 | N/A | 0.0000 | 0 | 129,177 |
| 2 | 0.90 | 0.0000 | 0.0000 | N/A | 0.0000 | 0 | 27,727 |
| 3 | 0.30 | 0.4539 | 0.2936 | 0.4606 | 0.4473 | 2,092,596 | 2,207,959 |
| 4 | 0.00001 | 0.0012 | 0.0006 | 0.1001 | 0.0006 | 4,604 | 858,568 |
| **Macro** | — | **0.1138** | **0.0735** | — | — | **2,097,200** | **3,223,431** |

Class 4 has a few predictions only at the very lowest tested threshold, but recall is about 0.06%. This is not meaningful rare-class detection.

## Comparison with the BCE baseline and dice_loss_v1

All three checkpoints were evaluated on the same validation split with the same threshold grid. BCE baseline used plain BCE during training; its values below are only common validation metrics, not its original training loss.

| Model | Default macro Dice | Default macro IoU | Tuned macro Dice | Tuned macro IoU |
|---|---:|---:|---:|---:|
| Original BCE baseline | 0.0054 | 0.0027 | 0.0951 | 0.0550 |
| `dice_loss_v1` | **0.1183** | **0.0774** | **0.1221** | **0.0800** |
| `rare_class_sampling_v1` | 0.1128 | 0.0728 | 0.1138 | 0.0735 |

Per-class comparison at default threshold:

| Class | Original BCE Dice / Recall | `dice_loss_v1` Dice / Recall | Rare-sampling Dice / Recall | Rare-sampling change vs `dice_loss_v1` |
|---:|---:|---:|---:|---|
| 1 | 0 / 0 | 0 / 0 | 0 / 0 | No improvement |
| 2 | 0 / 0 | 0 / 0 | 0 / 0 | No improvement |
| 3 | 0.0217 / 0.0110 | 0.4730 / 0.5525 | 0.4511 / 0.4113 | Worse Dice and recall |
| 4 | 0 / 0 | 0 / 0 | 0 / 0 | No improvement |

Class 3 false positives at threshold 0.5 were 16,887 for the BCE baseline, 3,130,734 for `dice_loss_v1`, and **1,645,987** for rare sampling. Thus rare sampling reduced class 3 false positives by about 47% versus `dice_loss_v1`, but also reduced class 3 recall from 55.25% to 41.13%. It did not improve default recall for classes 1, 2, or 4.

With validation-tuned thresholds, rare sampling's class 3 Dice/IoU/precision/recall were 0.4539 / 0.2936 / 0.4606 / 0.4473. `dice_loss_v1` reached 0.4803 / 0.3161 / 0.4771 / 0.4835. The tuned rare-class model therefore remained below `dice_loss_v1` on class 3 and aggregate scores. Classes 1 and 2 remained undetected, while class 4 recall was only 0.0006.

## Conclusion

The sampler met its 65/35 target and two-appearance cap, and it increased exposure to images containing rare classes. But the experiment did **not** improve rare-class recall at the default threshold, and macro Dice was lower than `dice_loss_v1`. It did reduce class 3 false positives, at the cost of class 3 recall. The results do not justify integrating the model into the application.

The code-level sampling error on the first launch was `ValueError: dict contains fields not in fieldnames: 'maximum_appearances_allowed'`. It occurred before any optimizer update. The CSV header was corrected, the same configuration was rerun, and the successful run completed as documented here.

## Artifacts

- Checkpoint: `best_model.pt` (epoch 3; load verified; not a smoke checkpoint)
- Run config and duration: `run_config.json`, `training_run_info.json`
- Train/validation loss and Dice/IoU history: `training_history.csv`
- Per-epoch sampled class counts and repetition checks: `sampling_history.csv`
- Preflight sample distribution: `sampling_preflight.json`
- Default and tuned metrics: `evaluation/validation_metrics_default.json`, `evaluation/validation_metrics_tuned.json`
- Same-grid baseline metrics: `baseline_comparison/original_bce/` and `baseline_comparison/dice_loss_v1/`
- Validation predictions: `visualizations/validation_predictions.png`
- Training curves: `training_curves.svg`
- No original baseline checkpoint, smoke checkpoint, `dice_loss_v1` artifact, NEU-CLS project, or application file was overwritten.
