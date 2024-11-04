import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .nf_utils import Flow


class Radial(Flow):
    """Radial transformation.

    Args:
        dim: dimension of input/output data, int
    """

    def __init__(self, dim: int = 2):
        """Create and initialize an affine transformation."""
        super().__init__()

        self.dim = dim

        self.x0 = nn.Parameter(
            torch.Tensor(
                self.dim,
            )
        )  # Vector used to parametrize z_0
        self.pre_alpha = nn.Parameter(
            torch.Tensor(
                1,
            )
        )  # Scalar used to indirectly parametrized \alpha
        self.pre_beta = nn.Parameter(
            torch.Tensor(
                1,
            )
        )  # Scaler used to indireclty parametrized \beta

        stdv = 1.0 / math.sqrt(self.dim)
        self.pre_alpha.data.uniform_(-stdv, stdv)
        self.pre_beta.data.uniform_(-stdv, stdv)
        self.x0.data.uniform_(-stdv, stdv)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Compute the forward transformation for the given input x.

        Args:
            x: input sample, shape [batch_size, dim]

        Returns:
            y: sample after forward transformation, shape [batch_size, dim]
            log_det_jac: log determinant of the jacobian of the forward transformation, shape [batch_size]
        """
        B, D = x.shape

        ##########################################################
        # YOUR CODE HERE
        alpha = F.softplus(self.pre_alpha)
        beta = -alpha + F.softplus(self.pre_beta)

        diff = x - self.x0
        r = diff.norm(dim=-1, keepdim=True)
        h = 1. / (alpha + r)
        y = x + beta * h * diff
        
        h_prime = - (h ** 2)
        log_det_jac = ((self.dim - 1) * torch.log1p(beta * h) + torch.log1p(beta * h + beta * h_prime * r)).sum(-1)
        ##########################################################

        assert y.shape == (B, D)
        assert log_det_jac.shape == (B,)

        return y, log_det_jac

    def inverse(self, y: Tensor) -> None:
        """Compute the inverse transformation given an input y.

        Args:
            y: input sample. shape [batch_size, dim]

        Returns:
            x: sample after inverse transformation. shape [batch_size, dim]
            inv_log_det_jac: log determinant of the jacobian of the inverse transformation, shape [batch_size]
        """
        raise ValueError("The inverse transformation is not known in closed form.")
