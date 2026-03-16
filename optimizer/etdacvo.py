import torch
from tdo import TDO
from acvo import ACVO


class ETDACVO:
    """
    Enhanced Tasmanian Devil Anti-Conservative Variable Optimization (ETDACVO)

    Joint evolutionary optimizer for:
    - augmentation parameters
    - optimizer hyperparameters
    """

    def __init__(
        self,
        dim=9,
        population_size=20,
        alpha1=1.6,
        alpha2=0.9,
        beta=0.3,
        device="cpu"
    ):

        self.dim = dim
        self.population_size = population_size
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.beta = beta
        self.device = device

        # initialize population
        self.population = torch.randn(population_size, dim, device=device)

        # best individual
        self.best = None

        # EWMA vector
        self.phi = torch.zeros(dim, device=device)

        # optimization modules
        self.tdo = TDO(dim=dim, device=device)
        self.acvo = ACVO(dim=dim, device=device)

    def decode_theta(self, theta):
        """
        Convert parameter vector to augmentation + optimizer config
        """

        return {
            "brightness": torch.clamp(theta[0], -0.2, 0.2).item(),
            "contrast": torch.clamp(theta[1], -0.2, 0.2).item(),
            "rotation": torch.clamp(theta[2], -15, 15).item(),
            "deform_alpha": torch.clamp(theta[3], 0, 30).item(),
            "deform_sigma": torch.clamp(theta[4], 1, 5).item(),
            "noise": torch.clamp(theta[5], 0, 0.05).item(),
            "lr": torch.clamp(theta[6], 1e-5, 1e-2).item(),
            "momentum": torch.clamp(theta[7], 0.8, 0.99).item(),
            "weight_decay": torch.clamp(theta[8], 1e-6, 1e-3).item(),
        }

    def update_population(self, fitness_scores):
        """
        Perform one ETDACVO evolutionary step
        """

        fitness_scores = torch.tensor(fitness_scores, device=self.device)

        # select best individual
        best_idx = torch.argmin(fitness_scores)
        self.best = self.population[best_idx].clone()

        new_population = []

        for theta in self.population:

            # ---- TDO exploration ----
            tdo_step = self.tdo.update()

            # ---- ACVO diversity ----
            acvo_step = self.acvo.update(self.population)

            # ---- EWMA smoothing ----
            self.phi = self.beta * self.phi + (1 - self.beta) * theta
            ewma_step = self.phi - theta

            # ---- unified update ----
            new_theta = (
                theta
                + self.alpha1 * tdo_step
                + self.alpha2 * acvo_step
                + ewma_step
            )

            new_population.append(new_theta)

        self.population = torch.stack(new_population)

    def get_best(self):
        """
        Return best parameter vector
        """
        return self.best