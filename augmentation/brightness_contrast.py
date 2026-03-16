import torch
import torchvision.transforms.functional as TF
import random


def adjust(x, brightness=0.2, contrast=0.2):
    """
    Apply brightness and contrast adjustment.
    
    x: torch tensor (C, H, W) normalized in [0, 1]
    brightness: max brightness scaling range (e.g., 0.2 -> ±20%)
    contrast: max contrast scaling range
    """

    # Brightness adjustment
    if brightness > 0:
        brightness_factor = random.uniform(1 - brightness, 1 + brightness)
        x = TF.adjust_brightness(x, brightness_factor)

    # Contrast adjustment
    if contrast > 0:
        contrast_factor = random.uniform(1 - contrast, 1 + contrast)
        x = TF.adjust_contrast(x, contrast_factor)

    # Clamp to valid intensity range
    x = torch.clamp(x, 0.0, 1.0)

    return x