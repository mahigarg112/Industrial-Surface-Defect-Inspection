# Dataset Preparation and Audit

## Observed source layout

The inspected source is `data/severstal-steel-defect-detection/`, with `train.csv`, `sample_submission.csv`, `train_images/`, and `test_images/`. `train.csv` has columns `ImageId`, `ClassId`, and `EncodedPixels`; there are 7,095 CSV rows and 12,568 training images. Every CSV image ID has a matching file. All 12,568 training images were opened and verified as RGB, 1600×256, with no decode errors. CSV class IDs are integers 1–4. Kaggle test images have no training masks and are excluded from splitting and evaluation.

An annotation row with an empty `EncodedPixels` is a negative for that image/class. Classes occur in these numbers of distinct positive training images: class 1: 897, class 2: 247, class 3: 5,150, class 4: 801. There are 6,666 images with at least one mask and 5,902 images with no defect mask. Image-level class totals overlap because one image can have more than one class.

## RLE decoding

Kaggle's encoding consists of alternating 1-based start positions and run lengths, traversed column-first. `decode_rle` subtracts one from each start, fills the corresponding flat runs, and reshapes to `(height, width)` in Fortran/column-major order. An empty RLE becomes an all-zero mask. Out-of-bounds runs raise an error. The four class channels are decoded separately, preserving possible mask overlap.

## Split and duplicate handling

Preparation assigns all training image IDs to train or validation with an 80/20 target and seed 42. Assignment is stratified by the image's four-class presence signature. SHA-256 groups byte-identical images before splitting, preventing exact duplicates from crossing the boundary. The split unit is the image, never a CSV annotation row. Similar-looking images with different bytes are not detected by this exact-duplicate check. No Kaggle test image is used.

The generated `prepared/manifest.csv` keeps image ID, split, and four RLE strings. `preparation_summary.json` records observed split counts, class prevalence, and exact duplicate totals. Raw CSV and image bytes remain read-only inputs.

## Transformations and imbalance

The source aspect ratio is 6.25:1. Images and masks are resized to 512×128, which preserves that ratio. RGB is converted to float `[0,1]` (no mean/std normalization). Images use bilinear interpolation; binary masks use nearest-neighbor interpolation. No padding or crop is used. Defect-free images remain in the dataset. Class 2 is especially rare, and class 3 is common; the baseline uses unweighted binary cross-entropy, so its scores may favor common/background pixels. Report per-class metrics and treat rare-class results as uncertain.
