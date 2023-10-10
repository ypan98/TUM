from typing import List, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torch.distributions.multivariate_normal import MultivariateNormal

from .nf_utils import Flow


class StackedFlows(nn.Module):
    """Stack a list of transformations with a given based distribtuion.

    Args:
        transforms: list fo stacked transformations. list of Flows
        dim: dimension of input/output data. int
        base_dist: name of the base distribution. options: ['Normal']
    """

    def __init__(
        self,
        transforms: List[Flow],
        dim: int = 2,
        base_dist: str = "Normal",
        device="cpu",
    ):
        super().__init__()

        if isinstance(transforms, Flow):
            self.transforms = nn.ModuleList(
                [
                    transforms,
                ]
            )
        elif isinstance(transforms, list):
            if not all(isinstance(t, Flow) for t in transforms):
                raise ValueError("transforms must be a Flow or a list of Flows")
            self.transforms = nn.ModuleList(transforms)
        else:
            raise ValueError(
                f"transforms must a Flow or a list, but was {type(transforms)}"
            )

        self.dim = dim
        if base_dist == "Normal":
            self.base_dist = MultivariateNormal(
                torch.zeros(self.dim).to(device), torch.eye(self.dim).to(device)
            )
        else:
            raise NotImplementedError

    def log_prob(self, x: Tensor) -> Tensor:
        """Compute log probability of a batch of data (slide 27).

        Args:
            x: input sample. shape [batch_size, dim]

        Returns:
            log_prob: Log probability of the data, shape [batch_size]
        """

        B, D = x.shape

        ##########################################################
        # YOUR CODE HERE
        inv_log_det = 0.0
        for transform in reversed(self.transforms):
            z, inv_log_det_jacobian = transform.inverse(x)
            inv_log_det += inv_log_det_jacobian
            x = z
        log_base = self.base_dist.log_prob(x)
        log_prob = (inv_log_det + log_base)
        ##########################################################

        assert log_prob.shape == (B,)

        return log_prob

    def rsample(self, batch_size: int) -> Tuple[Tensor, Tensor]:
        """Sample from the transformed distribution (slide 31).

        Returns:
            x: sample after forward transformation, shape [batch_size, dim]
            log_prob: Log probability of x, shape [batch_size]
        """
        ##########################################################
        # YOUR CODE HERE
        x = self.base_dist.rsample([batch_size])
        log_base = self.base_dist.log_prob(x)
        log_det = 0.0
        for transform in self.transforms:
            x, log_det_jacobian = transform.forward(x)
            log_det += log_det_jacobian
        log_prob = - log_det + log_base
        ##########################################################

        assert x.shape == (batch_size, self.dim)
        assert log_prob.shape == (batch_size,)

        return x, log_prob
