import torch
import torch.nn as nn
import torchvision.models as models


class ResNetModel(nn.Module):
    """
    ResNet baseline model for medical image classification.
    """

    def __init__(self, num_classes=2, backbone="resnet18", in_channels=1):
        super().__init__()

        if backbone == "resnet18":
            self.model = models.resnet18(weights=None)
            feature_dim = 512
        elif backbone == "resnet50":
            self.model = models.resnet50(weights=None)
            feature_dim = 2048
        else:
            raise ValueError("Unsupported backbone")

        # Adjust first convolution for grayscale MRI
        if in_channels != 3:
            self.model.conv1 = nn.Conv2d(
                in_channels,
                64,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False
            )

        # Replace classifier
        self.model.fc = nn.Linear(feature_dim, num_classes)

    def forward(self, x):
        return self.model(x)