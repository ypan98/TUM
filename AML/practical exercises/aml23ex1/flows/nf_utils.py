from typing import Tuple

import torch.nn as nn
from torch import Tensor


# Base class for all normalizing flows
class Flow(nn.Module):
    """Base class for transforms with learnable parameters."""

    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Compute f(x) and log_abs_det_jac(x)."""
        raise NotImplementedError

    def inverse(self, y: Tensor) -> Tuple[Tensor, Tensor]:
        """Compute f^-1(y) and inv_log_abs_det_jac(y)."""
        raise NotImplementedError

    def get_inverse(self):
        """Get inverse transformation."""
        return InverseFlow(self)


class InverseFlow(Flow):
    """Change the forward and inverse transformations."""

    def __init__(self, base_flow: Flow):
        """Create the inverse flow from a base flow.

        Args:
            base_flow: flow to reverse.
        """
        super().__init__()
        self.base_flow = base_flow
        if hasattr(base_flow, "domain"):
            self.codomain = base_flow.domain
        if hasattr(base_flow, "codomain"):
            self.domain = base_flow.codomain

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Compute the forward transformation given an input x.

        Args:
            x: input sample. shape [batch_size, dim]

        Returns:
            y: sample after forward tranformation. shape [batch_size, dim]
            log_det_jac: log determinant of the jacobian of the forward tranformation, shape [batch_size]
        """
        y, log_det_jac = self.base_flow.inverse(x)
        return y, log_det_jac

    def inverse(self, y: Tensor) -> Tuple[Tensor, Tensor]:
        """Compute the inverse transformation given an input y.

        Args:
            y: input sample. shape [batch_size, dim]

        Returns:
            x: sample after inverse tranformation. shape [batch_size, dim]
            inv_log_det_jac: log determinant of the jacobian of the inverse tranformation, shape [batch_size]
        """
        x, inv_log_det_jac = self.base_flow.forward(y)
        return x, inv_log_det_jac
