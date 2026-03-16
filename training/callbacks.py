import torch
import numpy as np


class Callback:
    """
    Base callback class.
    """

    def on_epoch_begin(self, epoch):
        pass

    def on_epoch_end(self, epoch, logs=None):
        pass


class ModelCheckpoint(Callback):
    """
    Save best model based on validation loss.
    """

    def __init__(self, filepath):
        self.filepath = filepath
        self.best_loss = float("inf")

    def on_epoch_end(self, epoch, logs=None):
        val_loss = logs.get("val_loss")

        if val_loss < self.best_loss:
            self.best_loss = val_loss
            torch.save(logs["model_state"], self.filepath)
            print(f"Saved best model at epoch {epoch}")


class LossTracker(Callback):
    """
    Track loss smoothness and oscillation amplitude.
    """

    def __init__(self):
        self.history = []

    def on_epoch_end(self, epoch, logs=None):
        loss = logs.get("val_loss")
        self.history.append(loss)

    def oscillation_amplitude(self):
        if len(self.history) < 2:
            return 0
        return max(self.history) - min(self.history)

    def coefficient_of_variation(self):
        if len(self.history) < 2:
            return 0
        arr = np.array(self.history)
        return arr.std() / (arr.mean() + 1e-8)


class EarlyStopping(Callback):
    """
    Stop training if no improvement.
    """

    def __init__(self, patience=10):
        self.patience = patience
        self.counter = 0
        self.best_loss = float("inf")
        self.stop = False

    def on_epoch_end(self, epoch, logs=None):
        val_loss = logs.get("val_loss")

        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1

        if self.counter >= self.patience:
            self.stop = True
            print("Early stopping triggered.")