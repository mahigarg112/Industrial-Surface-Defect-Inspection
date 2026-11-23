# Application Integration Plan

## Current application behavior

The current Streamlit application is `notebooks/SDI/app.py`. It is a whole-image classifier: it accepts an uploaded image, runs a softmax over a single set of class logits, displays one predicted class and its score, and plots class scores. Its inference helper is `notebooks/SDI/src/predict.py`; model construction is in `notebooks/SDI/src/models.py`; preprocessing is in `notebooks/SDI/src/data.py`.

The classifier loader expects a checkpoint schema with fields such as `model_type`, `image_size`, and `model_state`. The Severstal segmentation checkpoint uses a different U-Net, a `model_state_dict`, four output masks, and an elongated 512×128 input. The existing loader and UI cannot consume it safely as-is. No application file has been changed.

## Feasibility

Integration is technically feasible as a separate segmentation mode after a full model has been trained and evaluated. It is not ready now: the only checkpoint is a smoke checkpoint, so the application has no useful trained segmentation weights to load.

## Files likely to change in a future implementation

1. Add `notebooks/SDI/src/segmentation_predict.py` with a segmentation-specific checkpoint loader and image preprocessing. Keep the existing classification loader intact.
2. Update `notebooks/SDI/app.py` to let a user choose classification or segmentation, load the corresponding model type, and show a pixel mask for the segmentation option.
3. Optionally add a small palette/overlay helper next to the segmentation inference module. Do not force segmentation checkpoints into the existing `models/*/best.pt` classifier discovery pattern.

## Suggested user interface

For segmentation, display the uploaded image beside a colored mask overlay. Add class toggles for IDs 1–4 and an overlay-opacity control. Show each class's predicted pixel area or percentage with a clear label that it is derived from the mask. Do not show a single softmax “predicted class”: the four sigmoid masks are independent and may overlap. Preserve the current classification screen and its checkpoint handling as a separate mode.

## Integration sequence

1. Train the U-Net and keep its checkpoint and config together.
2. Verify validation metrics and inspect saved validation overlays.
3. Implement the separate inference helper and UI mode.
4. Manually check input sizing, mask colors, and class toggles on labeled examples.
5. Document that this remains a student research prototype, not a production inspection system.

This document is a plan only; no application files were edited.
