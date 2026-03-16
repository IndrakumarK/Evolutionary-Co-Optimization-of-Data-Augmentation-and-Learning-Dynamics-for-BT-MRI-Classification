import torch
import torchvision.transforms.functional as TF
import random


def rotate(x, max_angle=15):
    """
    Apply random rotation within ±max_angle degrees.

    x: torch tensor (C, H, W)
    max_angle: maximum rotation angle (e.g., 15)
    """

    if max_angle <= 0:
        return x

    angle = random.uniform(-max_angle, max_angle)

    # Use bilinear interpolation, reflect padding to preserve structure
    x = TF.rotate(
        x,
        angle=angle,
        interpolation=TF.InterpolationMode.BILINEAR,
        expand=False,
        fill=0
    )

    return x