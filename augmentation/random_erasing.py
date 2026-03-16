import torch
import random


def random_erasing(x, min_size=8, max_size=32, p=0.5):
    """
    Apply random erasing (cutout) augmentation.

    x: torch tensor (C, H, W)
    min_size: minimum cutout size
    max_size: maximum cutout size
    p: probability of applying erasing
    """

    if random.random() > p:
        return x

    _, h, w = x.shape

    cutout_size = random.randint(min_size, max_size)

    # Ensure cutout fits inside image
    if cutout_size >= h or cutout_size >= w:
        return x

    y = random.randint(0, h - cutout_size)
    x_coord = random.randint(0, w - cutout_size)

    x = x.clone()  # avoid in-place modification

    x[:, y:y + cutout_size, x_coord:x_coord + cutout_size] = 0.0

    return x