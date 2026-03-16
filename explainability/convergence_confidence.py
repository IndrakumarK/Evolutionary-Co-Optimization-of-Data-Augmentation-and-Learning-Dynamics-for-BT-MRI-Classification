import torch


class ConvergenceConfidence:
    """
    Evolutionary Convergence Confidence (ECC)

    Measures optimization stability from the ETDACVO
    parameter trajectory.

    Higher values indicate stronger convergence.
    """

    def __init__(self, beta=0.3, window=10, device="cpu"):
        """
        beta: stability sensitivity coefficient
        window: number of recent generations used for stability estimation
        """

        self.beta = beta
        self.window = window
        self.device = device

        self.trajectory = []

    def update(self, theta):
        """
        Store parameter vector from current generation
        """

        self.trajectory.append(theta.detach().clone().to(self.device))

        # keep only recent history
        if len(self.trajectory) > self.window + 1:
            self.trajectory.pop(0)

    def compute(self):
        """
        Compute convergence confidence ?
        """

        if len(self.trajectory) < 2:
            return torch.tensor(0.0, device=self.device)

        weights = []

        for i in range(1, len(self.trajectory)):

            diff = torch.norm(
                self.trajectory[i] - self.trajectory[i - 1]
            ) ** 2

            weight = torch.exp(-self.beta * diff)

            weights.append(weight)

        psi = torch.mean(torch.stack(weights))

        return psi