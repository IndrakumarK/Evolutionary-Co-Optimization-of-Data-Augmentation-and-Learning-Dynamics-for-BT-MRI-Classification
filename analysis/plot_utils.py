import matplotlib.pyplot as plt
import numpy as np


def plot_loss_curves(train_loss, val_loss, title="Training Curve", save_path=None):
    """
    Plot training and validation loss curves.
    """
    epochs = np.arange(1, len(train_loss) + 1)

    plt.figure(figsize=(6, 4))
    plt.plot(epochs, train_loss, label="Train Loss")
    plt.plot(epochs, val_loss, label="Validation Loss")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.grid(True)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


def plot_histogram(data_baseline, data_etdacvo,
                   title="Convergence Distribution",
                   save_path=None):
    """
    Plot histogram comparison for convergence epochs.
    """
    plt.figure(figsize=(6, 4))

    plt.hist(data_baseline, alpha=0.6, label="Baseline")
    plt.hist(data_etdacvo, alpha=0.6, label="ETDACVO")

    plt.xlabel("Epoch")
    plt.ylabel("Frequency")
    plt.title(title)
    plt.legend()
    plt.grid(True)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


def plot_boxplot(data_baseline, data_etdacvo,
                 title="Convergence Boxplot",
                 save_path=None):
    """
    Boxplot comparison between methods.
    """
    plt.figure(figsize=(5, 4))
    plt.boxplot([data_baseline, data_etdacvo],
                labels=["Baseline", "ETDACVO"])

    plt.ylabel("Epoch")
    plt.title(title)
    plt.grid(True)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


def plot_roc_curve(fpr, tpr, auc_score,
                   title="ROC Curve",
                   save_path=None):
    """
    Plot ROC curve.
    """
    plt.figure(figsize=(6, 4))

    plt.plot(fpr, tpr, label=f"AUC = {auc_score:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--")

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend()
    plt.grid(True)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()