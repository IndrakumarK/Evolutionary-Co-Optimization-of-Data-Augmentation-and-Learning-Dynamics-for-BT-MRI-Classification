import torch
import torch.nn as nn
import torchvision.models as models


class ViTModel(nn.Module):
    """
    Vision Transformer model for medical image classification.
    """

    def __init__(self,
                 num_classes=2,
                 image_size=224,
                 in_channels=1,
                 variant="vit_b_16"):
        super().__init__()

        if variant == "vit_b_16":
            self.model = models.vit_b_16(weights=None)
            embed_dim = 768
        else:
            raise ValueError("Unsupported ViT variant")

        # Adjust patch embedding layer for grayscale MRI
        if in_channels != 3:
            self.model.conv_proj = nn.Conv2d(
                in_channels,
                embed_dim,
                kernel_size=16,
                stride=16
            )

        # Replace classifier head
        self.model.heads = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        return self.model(x)