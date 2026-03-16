import torch
import torch.nn as nn


class ConcatenationFusion(nn.Module):
    """
    Simple concatenation fusion.
    """
    def __init__(self, in_dim1, in_dim2, out_dim):
        super().__init__()
        self.fc = nn.Linear(in_dim1 + in_dim2, out_dim)

    def forward(self, x, y):
        fused = torch.cat([x, y], dim=1)
        return self.fc(fused)


class AdditiveFusion(nn.Module):
    """
    Element-wise additive fusion.
    """
    def forward(self, x, y):
        return x + y


class AttentionWeightedFusion(nn.Module):
    """
    Proposed attention-weighted fusion (?1, ?2 learnable).
    """

    def __init__(self, feature_dim):
        super().__init__()

        self.gamma1 = nn.Parameter(torch.tensor(0.5))
        self.gamma2 = nn.Parameter(torch.tensor(0.5))

        self.norm = nn.Softmax(dim=0)

    def forward(self, x, y):
        weights = torch.stack([self.gamma1, self.gamma2])
        weights = self.norm(weights)

        fused = weights[0] * x + weights[1] * y
        return fused


def fusion(x, y, method="attention", feature_dim=None):
    """
    Wrapper function for fusion methods.
    """

    if method == "concat":
        module = ConcatenationFusion(
            x.shape[1],
            y.shape[1],
            feature_dim if feature_dim else x.shape[1]
        )
        return module(x, y)

    elif method == "add":
        return AdditiveFusion()(x, y)

    elif method == "attention":
        module = AttentionWeightedFusion(x.shape[1])
        return module(x, y)

    else:
        raise ValueError("Unknown fusion method.")