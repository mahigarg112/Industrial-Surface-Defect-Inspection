# dice_loss_v1

This experiment keeps the baseline U-Net, input size, optimizer, learning rate,
seed, preprocessing, batch size, and train/validation split. It changes the
training loss to an equal combination of binary cross-entropy (BCE) and soft
Dice loss. It does not use class weights or defect-aware sampling.

## Loss in beginner-friendly terms

The total loss is:

`total_loss = 0.5 * BCE + 0.5 * Dice_loss`

BCE checks each predicted pixel probability against its 0/1 target. Soft Dice
compares overlap without first rounding probabilities to 0 or 1:

`soft_Dice = (2 * sum(probability * target) + smooth) / (sum(probability) + sum(target) + smooth)`

`Dice_loss = 1 - soft_Dice`, with `smooth = 1` to keep the formula defined for
an empty target mask. BCE helps discourage false alarms, while Dice gives the
small defect regions more influence than plain pixel-averaged BCE alone.

## Checkpoint rule

After each epoch, the script evaluates the validation split at the configured
default threshold (0.5). It saves `best_model.pt` only when validation macro
Dice improves. If macro Dice does not improve for two consecutive epochs,
training stops. The checkpoint therefore contains the best validation macro
Dice epoch, not necessarily the last epoch. Validation is never used for
gradient updates.

## Threshold results

`validate.py` writes default-threshold metrics and a separate validation-tuned
metrics file. The tuned file chooses the per-class threshold with the highest
Dice from a fixed candidate list, preferring the higher threshold on ties.
These tuned scores are validation results and may be optimistic because the
same validation split selects the thresholds.

All checkpoints, histories, configuration snapshots, metrics, and visualization
files for this experiment belong in this directory. The original baseline and
smoke checkpoint remain in the parent experiment's `results/` directory.
