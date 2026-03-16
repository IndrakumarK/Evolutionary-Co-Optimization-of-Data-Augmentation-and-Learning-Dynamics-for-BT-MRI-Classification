import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize


def compute_binary_roc(y_true, y_scores):
    """
    Compute ROC curve and AUC for binary classification.
    
    y_true   : Ground truth labels (0/1)
    y_scores : Predicted probabilities for positive class
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    return fpr, tpr, roc_auc


def compute_multiclass_roc(y_true, y_scores, num_classes):
    """
    Compute ROC and AUC for multi-class classification (one-vs-rest).
    
    y_true   : Ground truth labels
    y_scores : Predicted probabilities (N x C)
    """
    y_true_bin = label_binarize(y_true, classes=list(range(num_classes)))

    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_scores[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    return fpr, tpr, roc_auc


def plot_binary_roc(fpr, tpr, roc_auc,
                    title="ROC Curve",
                    save_path=None):
    """
    Plot binary ROC curve.
    """
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--")

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend()
    plt.grid(True)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


def roc(y_true, y_scores, num_classes=2,
        plot=True, save_path=None):
    """
    Unified ROC function.
    Automatically selects binary or multiclass.
    """

    if num_classes == 2:
        fpr, tpr, roc_auc = compute_binary_roc(y_true, y_scores)

        if plot:
            plot_binary_roc(fpr, tpr, roc_auc, save_path=save_path)

        return fpr, tpr, roc_auc

    else:
        fpr, tpr, roc_auc = compute_multiclass_roc(
            y_true, y_scores, num_classes
        )

        if plot:
            plt.figure(figsize=(6, 5))
            for i in range(num_classes):
                plt.plot(fpr[i], tpr[i],
                         label=f"Class {i} AUC={roc_auc[i]:.3f}")

            plt.plot([0, 1], [0, 1], linestyle="--")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title("Multi-Class ROC")
            plt.legend()
            plt.grid(True)

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches="tight")

            plt.show()

        return fpr, tpr, roc_auc