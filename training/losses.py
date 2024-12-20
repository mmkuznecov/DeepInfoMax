import torch
import torch.nn as nn
import torch.nn.functional as F


class DeepInfoMaxLoss(nn.Module):
    """Implementation of Deep InfoMax loss."""

    def __init__(
        self,
        global_disc,
        local_disc,
        prior_disc,
        alpha: float = 0.5,
        beta: float = 1.0,
        gamma: float = 0.1,
    ):
        """
        Initialize Deep InfoMax loss.

        Args:
            global_disc: Global discriminator module
            local_disc: Local discriminator module
            prior_disc: Prior discriminator module
            alpha: Weight for global loss term
            beta: Weight for local loss term
            gamma: Weight for prior loss term
        """
        super().__init__()
        self.global_d = global_disc
        self.local_d = local_disc
        self.prior_d = prior_disc
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def forward(self, y, M, M_prime):
        # Expand encoded representation for concatenation
        y_exp = y.unsqueeze(-1).unsqueeze(-1)
        y_exp = y_exp.expand(-1, -1, 26, 26)

        # Concatenate features with expanded encoding
        y_M = torch.cat((M, y_exp), dim=1)
        y_M_prime = torch.cat((M_prime, y_exp), dim=1)

        # Local loss
        Ej = -F.softplus(-self.local_d(y_M)).mean()
        Em = F.softplus(self.local_d(y_M_prime)).mean()
        LOCAL = (Em - Ej) * self.beta

        # Global loss
        Ej = -F.softplus(-self.global_d(y, M)).mean()
        Em = F.softplus(self.global_d(y, M_prime)).mean()
        GLOBAL = (Em - Ej) * self.alpha

        # Prior loss
        prior = torch.rand_like(y)
        eps = 1e-10  # for avoiding nans
        term_a = torch.log(self.prior_d(prior) + eps).mean()
        term_b = torch.log(1.0 - self.prior_d(y) + eps).mean()
        PRIOR = -(term_a + term_b) * self.gamma

        return LOCAL + GLOBAL + PRIOR


class ReconstructionLoss(nn.Module):
    """Simple MSE reconstruction loss for autoencoder training."""

    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, reconstruction, target):
        return self.mse(reconstruction, target)
