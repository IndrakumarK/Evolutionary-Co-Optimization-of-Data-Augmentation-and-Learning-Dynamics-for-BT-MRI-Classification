import numpy as np


def convergence_epoch(loss_history):
    """
    Convergence defined as first epoch where loss
    drops to 10% of initial value.
    This corresponds to 90% reduction,
    matching the paper definition.
    """
    threshold = 0.1 * loss_history[0]
    for i, loss in enumerate(loss_history):
        if loss <= threshold:
            return i
    return len(loss_history)


def oscillation_amplitude(loss_history):
    return max(loss_history) - min(loss_history)


def curvature_metric(loss_history):
    diffs = np.diff(loss_history)
    second_diffs = np.diff(diffs)
    return np.mean(np.abs(second_diffs))