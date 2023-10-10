import torch
import torch.nn as nn

from typeguard import typechecked
from torchtyping import TensorType, patch_typeguard

patch_typeguard()

class Decoder(nn.Module):
    
    def __init__(self, input_dim: int, latent_dim: int, hidden_dim: int = 100) -> None:
        """Initialize the decoder. The decoder is a 2-layer MLP that "inverts" the encoder mapping and thus mirrors its structure.

        Args:
            input_dim (int): Dimension of the inputs to the encoder
            latent_dim (int): Dimension of the latent representations (i.e. outputs of the encoder)
            hidden_dim (int, optional): Dimension of the first hidden layer of the MLP. Defaults to 100.
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.linear3 = nn.Linear(latent_dim, hidden_dim)
        self.linear4 = nn.Linear(hidden_dim, input_dim)
        
    
    @typechecked
    def forward(self, z: TensorType['batch_size', 'latent_dim']) -> TensorType['batch_size', 'input_dim']:
        """Convert sampled latent variables z into observations x.
        
        Args:
            z: Sampled latent variables, shape [batch_size, latent_dim]
        
        Returns:
            theta: Parameters of the conditional likelihood, shape [batch_size, input_dim]
        """
        ##########################################################
        # YOUR CODE HERE
        theta = torch.relu(self.linear3(z))
        theta = torch.sigmoid(self.linear4(theta))
        return theta
        ##########################################################