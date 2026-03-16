import torch
import numpy as np
import random
from scipy.ndimage import gaussian_filter, map_coordinates


def elastic(x, alpha=20, sigma=3):
    """
    Apply elastic deformation to a 2D MRI slice.

    x: torch tensor (C, H, W)
    alpha: scaling factor for deformation intensity
    sigma: smoothing factor (controls deformation smoothness)
    """

    if alpha <= 0:
        return x

    x_np = x.numpy()
    c, h, w = x_np.shape

    # Generate random displacement fields
    dx = gaussian_filter(
        (np.random.rand(h, w) * 2 - 1),
        sigma,
        mode="reflect"
    ) * alpha

    dy = gaussian_filter(
        (np.random.rand(h, w) * 2 - 1),
        sigma,
        mode="reflect"
    ) * alpha

    x_coords, y_coords = np.meshgrid(np.arange(w), np.arange(h))

    indices = (
        np.reshape(y_coords + dy, (-1, 1)),
        np.reshape(x_coords + dx, (-1, 1))
    )

    deformed = np.zeros_like(x_np)

    for i in range(c):
        deformed[i] = map_coordinates(
            x_np[i],
            indices,
            order=1,
            mode='reflect'
        ).reshape(h, w)

    return torch.tensor(deformed, dtype=x.dtype)