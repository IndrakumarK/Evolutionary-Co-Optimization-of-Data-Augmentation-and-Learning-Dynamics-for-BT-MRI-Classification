import torch
import torch.nn as nn
import torchvision.models as models
from .fusion import AttentionWeightedFusion


class HybridModel(nn.Module):
    """
    Hybrid CNN + Transformer model
    (Local + Global feature fusion)
    """

    def __init__(self, num_classes=2):
        super().__init__()

        # ---------------------------
        # CNN Backbone (Local Features)
        # ---------------------------
        self.cnn = models.resnet18(weights=None)
        self.cnn.fc = nn.Identity()  # Remove classifier

        cnn_feature_dim = 512

        # ---------------------------
        # Transformer Branch (Global Features)
        # ---------------------------
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=512,
                nhead=8,
                batch_first=True
            ),
            num_layers=2
        )

        self.transformer_pool = nn.AdaptiveAvgPool1d(1)

        # ---------------------------
        # Fusion (Attention-Weighted)
        # ---------------------------
        self.fusion = AttentionWeightedFusion(feature_dim=512)

        # ---------------------------
        # Final Classifier
        # ---------------------------
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        """
        x: (B, C, H, W)
        """

        # CNN branch
        cnn_features = self.cnn(x)  # (B, 512)

        # Transformer branch
        # Flatten spatial features for transformer
        B, C, H, W = x.shape
        flattened = x.view(B, C, -1).permute(0, 2, 1)  # (B, HW, C)

        transformer_features = self.transformer(flattened)
        transformer_features = transformer_features.mean(dim=1)  # Global pooling

        # Fusion
        fused = self.fusion(cnn_features, transformer_features)

        # Classification
        logits = self.classifier(fused)

        return logits