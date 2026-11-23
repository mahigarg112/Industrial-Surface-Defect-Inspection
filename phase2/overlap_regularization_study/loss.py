"""A single auxiliary term; original K objective stays unchanged."""
import math
import torch
from phase2.objectives import balanced_present_dice_loss


def pairwise_probability_overlap(probabilities):
    """Mean of p_i*p_j over six unordered class pairs and B,H,W."""
    if probabilities.ndim != 4 or probabilities.shape[1] != 4:
        raise ValueError("Expected B,4,H,W probabilities")
    return torch.stack([probabilities[:,i]*probabilities[:,j]
                        for i in range(4) for j in range(i+1,4)]).mean()


def loss_terms(logits, target, class_weights, coefficient=.10):
    if not math.isfinite(coefficient) or coefficient < 0:
        raise ValueError("Overlap coefficient must be finite and nonnegative")
    baseline = balanced_present_dice_loss(logits,target,class_weights)
    # FP32 probabilities and reductions even during CUDA FP16 forward.
    auxiliary = pairwise_probability_overlap(logits.float().sigmoid())
    return baseline+coefficient*auxiliary,baseline,auxiliary
