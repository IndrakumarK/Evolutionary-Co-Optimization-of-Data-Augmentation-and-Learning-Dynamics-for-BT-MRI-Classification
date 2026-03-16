import torch
import math


class TDO:
    """
    Tasmanian Devil Optimization (TDO)
    Lévy-flight-based exploration mechanism.

    Generates stochastic exploration steps using the
    Mantegna algorithm for Lévy distributions.
    """

    def __init__(
        self,
        dim: int,
        levy_exponent: float = 1.5,
        step_scale: float = 0.01,
        device: str = "cpu",
    ):

        self.dim = dim
        self.beta = levy_exponent
        self.step_scale = step_scale
        self.device = device

    def levy_flight(self):
        """
        Generate Lévy flight step using the Mantegna algorithm.
        """

        beta = self.beta

        # Lévy sigma calculation
        sigma_u = (
            math.gamma(1 + beta)
            * math.sin(math.pi * beta / 2)
            / (
                math.gamma((1 + beta) / 2)
                * beta
                * 2 ** ((beta - 1) / 2)
            )
        ) ** (1 / beta)

        # random variables
        u = torch.randn(self.dim, device=self.device) * sigma_u
        v = torch.randn(self.dim, device=self.device)

        # prevent division instability
        step = u / (torch.abs(v) ** (1 / beta) + 1e-8)

        return step

    def update(self):
        """
        Return Lévy exploration step.

        This step will be combined with other components
        inside the ETDACVO optimizer.
        """

        levy_step = self.levy_flight()

        return self.step_scale * levy_step