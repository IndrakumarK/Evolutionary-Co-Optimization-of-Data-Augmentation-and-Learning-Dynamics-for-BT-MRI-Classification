import torch
import numpy as np
from sklearn.metrics import f1_score as sk_f1
from sklearn.metrics import roc_auc_score


# -----------------------------
# Accuracy
# -----------------------------
def accuracy_score(preds, targets):
    """
    Classification accuracy.
    """
    preds = preds.view(-1)
    targets = targets.view(-1)
    correct = (preds == targets).sum().item()
    total = targets.numel()
    return correct / (total + 1e-8)


# -----------------------------
# F1 Score
# -----------------------------
def f1_score(preds, targets, average="macro"):
    """
    Multi-class F1 score.
    """
    preds = preds.cpu().numpy()
    targets = targets.cpu().numpy()
    return sk_f1(targets, preds, average=average)


# -----------------------------
# Dice Score (Segmentation)
# -----------------------------
def dice_score(preds, targets, epsilon=1e-6):
    """
    Dice coefficient for segmentation.
    """
    preds = preds.float()
    targets = targets.float()

    intersection = (preds * targets).sum()
    union = preds.sum() + targets.sum()

    return ((2.0 * intersection + epsilon) /
            (union + epsilon)).item()


# -----------------------------
# AUROC
# -----------------------------
def auroc_score(logits, targets):
    """
    Compute AUROC for classification.
    logits: raw model outputs
    """
    probs = torch.softmax(logits, dim=1)
    probs = probs.detach().cpu().numpy()
    targets = targets.detach().cpu().numpy()

    # Multi-class AUROC
    return roc_auc_score(targets, probs, multi_class="ovr")