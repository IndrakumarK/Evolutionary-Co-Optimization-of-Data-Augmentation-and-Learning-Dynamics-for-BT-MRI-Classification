import torch


class EWMA:
    """
    Exponentially Weighted Moving Average smoothing
    """

    def __init__(self, dim, beta=0.3, device="cpu"):
        """
        dim: dimensionality of parameter vector
        beta: smoothing coefficient
        """
        self.beta = beta
        self.device = device
        self.phi = torch.zeros(dim, device=device)
        self.initialized = False

    def update(self, theta):
        """
        theta: tensor shape [dim]
        """

        if not self.initialized:
            self.phi = theta.clone()
            self.initialized = True
        else:
            self.phi = (
                self.beta * self.phi
                + (1 - self.beta) * theta
            )

        return self.phi