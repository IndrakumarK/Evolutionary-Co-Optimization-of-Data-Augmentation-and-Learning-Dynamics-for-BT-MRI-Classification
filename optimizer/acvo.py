import torch


class ACVO:
    """
    Anti-Conservative Variable Optimization (ACVO)

    Injects diversity into the evolutionary population by
    sampling perturbations from a population-derived
    covariance structure.
    """

    def __init__(self,
                 dim: int,
                 gamma: float = 0.2,
                 sigma: float = 0.1,
                 device: str = "cpu"):

        self.dim = dim
        self.gamma = gamma
        self.sigma = sigma
        self.device = device

    def compute_correlation(self, population):
        """
        Estimate covariance / correlation matrix from population.
        """

        pop_tensor = torch.stack(population).to(self.device)

        # Compute covariance matrix
        cov = torch.cov(pop_tensor.T)

        # Numerical stability
        cov = cov + 1e-5 * torch.eye(self.dim, device=self.device)

        return cov

    def update(self, population):
        """
        Generate ACVO diversity step.

        population: list of parameter vectors
        """

        R = self.compute_correlation(population)

        epsilon = torch.randn(self.dim, device=self.device) * self.sigma

        perturbation = self.gamma * torch.matmul(R, epsilon)

        return perturbation