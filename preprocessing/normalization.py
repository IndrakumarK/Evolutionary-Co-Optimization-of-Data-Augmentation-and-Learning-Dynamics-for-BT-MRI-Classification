import numpy as np
import torch


def normalize(x, mask_background=True):
    """
    Z-score normalization for MRI volumes.

    x: numpy array or torch tensor
    mask_background: normalize only non-zero voxels
    """

    if isinstance(x, torch.Tensor):
        x_np = x.cpu().numpy()
    else:
        x_np = x

    if mask_background:
        mask = x_np > 0
        if np.sum(mask) == 0:
            return x
        mean = x_np[mask].mean()
        std = x_np[mask].std()
    else:
        mean = x_np.mean()
        std = x_np.std()

    normalized = (x_np - mean) / (std + 1e-8)

    if isinstance(x, torch.Tensor):
        return torch.tensor(normalized, dtype=x.dtype, device=x.device)
    else:
        return normalized