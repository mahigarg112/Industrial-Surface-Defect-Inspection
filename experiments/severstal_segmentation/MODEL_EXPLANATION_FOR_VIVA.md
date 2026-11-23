# Viva Explanation: Severstal Segmentation Experiment

## What problem does it solve?

The Severstal steel dataset provides defect masks, so the goal is to identify *where* each defect is in an image. The earlier NEU-CLS experiments classify a whole image; this experiment predicts a mask for every pixel and is a separate task.

## How are labels represented?

The CSV stores masks as run-length encoding (RLE). The decoder expands each run into a binary image using the dataset's one-based, column-first convention. The model predicts four mask channels, one for each CSV class ID (1 through 4), plus implicit background when every channel is zero. Independent channels are appropriate because annotations need not be mutually exclusive.

## What does U-Net do?

The encoder repeatedly reduces the image size while learning visual features. The decoder increases spatial resolution again. Skip connections copy fine details from encoder layers to matching decoder layers, which helps the network localize narrow defect regions. A final 1×1 convolution creates four pixel-level logits. Sigmoid converts each logit into an independent defect probability.

## How is it trained and evaluated?

Binary cross-entropy compares the four predicted masks with the four target masks. Images are split by image ID into fixed-seed training and validation groups; exact byte duplicates are kept together. The optimizer learns from training images, while validation loss selects the best checkpoint. Dice and IoU summarize overlap between predicted and true masks, with per-class results to expose imbalance. Kaggle test images are not evaluated because their masks are not public.

## What should I say about limitations?

This is a small baseline, not a production inspection system. Resizing to 512×128 loses some fine detail. The classes are imbalanced, particularly class 2, and background pixels are numerous; unweighted loss may underperform on rare defects. Validation metrics measure only the chosen internal split and do not establish performance on new factories or camera conditions. A future study could compare class-weighted or Dice-based losses, augmentations, higher resolution, and a truly labeled external test set.
