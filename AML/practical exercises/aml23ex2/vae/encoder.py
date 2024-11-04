import torch
import torch.nn as nn

from typeguard import typechecked
from torchtyping import TensorType, patch_typeguard
from typing import Tuple

patch_typeguard()

class Encoder(nn.Module):
    
    def __init__(self, input_dim: int, latent_dim: int, hidden_dim: int = 100) -> None:
        """Initialize the encoder. The encoder is a 2-layer MLP that outputs the parameters of the variational distribution (see slides 102-104)

        Args:
            input_dim (int): Dimension of the inputs
            latent_dim (int): Dimension of the latent representations (i.e. outputs of the encoder)
            hidden_dim (int, optional): Dimension of the first hidden layer of the MLP. Defaults to 100.
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear21 = nn.Linear(hidden_dim, latent_dim)
        self.linear22 = nn.Linear(hidden_dim, latent_dim)
        
    
    @typechecked
    def forward(self, x: TensorType['batch_size', 'input_dim']) -> Tuple[TensorType['batch_size', 'latent_dim'], TensorType['batch_size', 'latent_dim']]:
        """Obtain the parameters of q(z) for a batch of data points.
        
        Args:
            x: Batch of data points, shape [batch_size, input_dim]
        
        Returns:
            mu: Means of q(z), shape [batch_size, latent_dim]
            logsigma: Log-sigmas of q(z), shape [batch_size, latent_dim]
        """
        ##########################################################
        # YOUR CODE HERE
        hidden = self.linear1(x)
        hidden = torch.relu(hidden)
        mu = self.linear21(hidden)
        logsigma = self.linear22(hidden)
        return mu, logsigma
        ##########################################################