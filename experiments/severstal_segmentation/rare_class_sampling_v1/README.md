# rare_class_sampling_v1

This run keeps the `dice_loss_v1` U-Net, BCE + soft Dice loss, preprocessing,
optimizer, learning rate, batch size, seed, and data split. Only the training
image sampler changes. Validation remains sequential, with no repeated or
weighted validation images.

## Exact sampling plan

Each 10,054-draw epoch contains 6,535 defect-containing image draws (65%) and
3,519 defect-free draws (35%). Every defect-containing training image is used
once. Of the 5,332 unique defect-containing training images, 1,203 are then
selected for one additional draw. The extra-draw selection is weighted without
replacement, so a source image appears at most twice. The 3,519 defect-free
images are sampled uniformly without replacement from the 4,722 available.
The combined draw list is shuffled before batching.

For each class `c`, let `N_c` be the number of unique training images containing
that class. Its rarity factor is `N_c^(-0.5)`. For a positive image with active
class set `S`, its image weight is the **average** rarity factor:

`image_weight = sum(N_c^(-0.5) for c in S) / number_of_active_classes`

The average prevents a multi-class image from receiving a large weight merely
because it has several labels. Sampling the extra positive draws without
replacement applies the cap of two appearances per source image. Epoch seeds
are `seed + epoch_index`, making the plan reproducible while changing which
positive images receive the one repeat in later epochs.

The preflight command writes `sampling_preflight.json` before any optimizer
updates. It verifies the draw count and repetition cap and records actual
per-class draw counts. `sampling_history.csv` records the class distribution,
unique source images, and repetition count for every training epoch.

## Evaluation and interpretation

Checkpoint selection is the highest validation macro Dice at threshold 0.5;
early stopping patience is two epochs. Validation is never used for gradient
updates. Compare default-threshold and validation-tuned metrics, and watch class
3 false-positive pixels because increasing defect exposure can increase false
alarms. The tuned threshold scores are validation results and may be optimistic.
