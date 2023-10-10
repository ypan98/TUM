from functools import partial

import einops as eo
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked

patch_typeguard()


# A batch of (noisy) images
ImageBatch = TensorType["batch_size", "channels", "height", "width", torch.float32]

# Integer noise level between 0 and N - 1
NoiseLevel = TensorType["batch_size", torch.long]

# Normalized noise level between 0 and 1
NormalizedNoiseLevel = TensorType["batch_size", torch.float32]


def batch_broadcast(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Make `a` broadcast along the batch dimension of `b`.

    We assume the batch dimension to be the first one.
    """

    assert a.ndim == 1
    return a.view(-1, *((1,) * (b.ndim - 1)))


class ResNet(nn.Module):
    """A minimal convolutional residual network."""

    def __init__(self, feature_dim: int, hidden_dim: int, n_layers: int):
        super().__init__()

        ConvLayer = partial(nn.Conv2d, kernel_size=3, padding=1)

        # Layers to map from data space to learned latent space and back
        self.embed = nn.Sequential(ConvLayer(feature_dim + 1, hidden_dim), nn.SiLU())
        self.out = ConvLayer(hidden_dim, feature_dim)

        self.layers = nn.ModuleList(
            [
                nn.Sequential(
                    ConvLayer(hidden_dim + 1, hidden_dim),
                    nn.SiLU(),
                    ConvLayer(hidden_dim, hidden_dim, kernel_size=3),
                )
                for i in range(n_layers)
            ]
        )

    @typechecked
    def forward(self, z_n: ImageBatch, n: NormalizedNoiseLevel) -> ImageBatch:
        # Align n with the feature dimension of 2D image tensors
        n = n[:, None, None, None].expand(n.shape[0], -1, *z_n.shape[2:])

        z_n = self.embed(torch.cat((z_n, n), dim=-3))

        for layer in self.layers:
            z_n = z_n + layer(torch.cat((z_n, n), dim=-3))

        return self.out(z_n)


class MiniUnet(nn.Module):
    """A minimal U-net implementation [1].

    [1] Olaf Ronneberger, Philipp Fischer, Thomas Brox: "U-Net: Convolutional Networks
        for Biomedical Image Segmentation". https://arxiv.org/abs/1505.04597
    """

    def __init__(self, feature_dim: int, hidden_dim: int, n_layers: int):
        super().__init__()

        assert n_layers <= 2, (
            "MNIST images can only be downsampled twice "
            "without taking care of padding issues"
        )

        self.n_layers = n_layers

        ConvLayer = partial(nn.Conv2d, kernel_size=3, padding=1)

        # Layers to map from data space to learned latent space and back
        self.embed = nn.Sequential(ConvLayer(feature_dim + 1, hidden_dim), nn.SiLU())
        self.out = ConvLayer(hidden_dim, feature_dim)

        # At each scale, we perform one nonlinear map with residual connection
        self.downscaling = nn.ModuleList(
            [
                nn.Sequential(
                    ConvLayer(4**i * hidden_dim + 1, 4**i * hidden_dim),
                    nn.SiLU(),
                    nn.Conv2d(4**i * hidden_dim, 4**i * hidden_dim, kernel_size=1),
                )
                for i in range(n_layers)
            ]
        )
        bottom_channels = 4**n_layers * hidden_dim
        self.bottom_map = nn.Sequential(
            ConvLayer(bottom_channels + 1, bottom_channels),
            nn.SiLU(),
            ConvLayer(bottom_channels, bottom_channels),
        )
        self.upscaling = nn.ModuleList(
            [
                nn.Sequential(
                    ConvLayer(2 * 4**i * hidden_dim + 1, 4**i * hidden_dim),
                    nn.SiLU(),
                    nn.Conv2d(4**i * hidden_dim, 4**i * hidden_dim, kernel_size=1),
                )
                for i in reversed(range(1, n_layers + 1))
            ]
        )

    @typechecked
    def forward(self, z_n: ImageBatch, n: NormalizedNoiseLevel) -> ImageBatch:
        # Align n with the feature dimension of 2D image tensors
        n = n[:, None, None, None]

        def cat_n(z_n, *tensors):
            return torch.cat((z_n, *tensors, n.expand(-1, -1, *z_n.shape[2:])), dim=-3)

        z_n = self.embed(cat_n(z_n))

        skip_connections = []
        for down_layer in self.downscaling:
            z_n = z_n + down_layer(cat_n(z_n))
            z_n = eo.rearrange(z_n, "b c (h h2) (w w2) -> b (c h2 w2) h w", h2=2, w2=2)
            skip_connections.append(z_n)

        z_n = self.bottom_map(cat_n(z_n))

        for up_layer in self.upscaling:
            z_n = z_n + up_layer(cat_n(z_n, skip_connections.pop()))
            z_n = eo.rearrange(z_n, "b (c h2 w2) h w -> b c (h h2) (w w2)", h2=2, w2=2)

        return self.out(z_n)


class DDPM(nn.Module):
    """A denoising diffusion model as described in [1].

    References:

    [1] "Denoising Diffusion Probabilistic Models", Ho et al., https://arxiv.org/abs/2006.11239
    """

    def __init__(self, N: int, type: str, hidden_dim: int, n_layers: int):
        """Initialize the diffusion model.

        Args:
            N: Number of diffusion steps
        """

        super().__init__()

        self.N = N
        self.type = type

        if type == "resnet":
            self.model = ResNet(feature_dim=1, hidden_dim=hidden_dim, n_layers=n_layers)
        elif type == "unet":
            self.model = MiniUnet(
                feature_dim=1, hidden_dim=hidden_dim, n_layers=n_layers
            )
        else:
            raise RuntimeError(f"Unknown model type {type}")

        # Compute a beta schedule and various derived variables as defined on the slides
        ##########################################################
        # YOUR CODE HERE
        start = 1e-4
        end = 2e-2
        beta = torch.linspace(start, end, N)
        alpha = 1 - beta
        alpha_bar = alpha.cumprod(dim=0)
        alpha_bar_prev = torch.cat((torch.zeros(1), alpha_bar[:-1]))
        beta_tilde = beta * (1-alpha_bar_prev)/(1-alpha_bar)
        
        ##########################################################

        self.register_buffer("alpha", alpha.float())
        self.register_buffer("beta", beta.float())
        self.register_buffer("alpha_bar", alpha_bar.float())
        self.register_buffer("beta_tilde", beta_tilde.float())

    @typechecked
    def simplified_loss(
        self, x0: ImageBatch, n: NoiseLevel, epsilon: ImageBatch
    ) -> torch.Tensor:
        """Compute the simplified ELBO loss.

        Args:
            x0: Raw image data to compute the loss for
            n: Noise level
            epsilon: Noise instance

        Returns:
            0-dimensional tensor of the fully-reduced loss
        """

        ##########################################################
        # YOUR CODE HERE
        n_ = n[:, None, None, None]
        z_n = torch.sqrt(self.alpha_bar[n_]) * x0 + torch.sqrt(1-self.alpha_bar[n_]) * epsilon
        n_normalized = n / (self.N - 1)
        pred_epsilon = self.model(z_n, n_normalized)
        # c_n = self.beta[n_] **2 / (2 * self.beta_tilde[n_] * self.alpha[n_] * (1 - self.alpha_bar[n_]))
        c_n = 1
        loss = torch.mean( c_n * (pred_epsilon - epsilon)**2)
        return loss
        ##########################################################

    def loss(self, x0: ImageBatch) -> torch.Tensor:
        batch_size = x0.shape[0]
        n = torch.randint(self.N, (batch_size,), device=x0.device)
        epsilon = torch.randn_like(x0)

        return self.simplified_loss(x0, n, epsilon)

    @typechecked
    def estimate_x0(
        self, z_n: ImageBatch, n: NoiseLevel, epsilon: ImageBatch
    ) -> ImageBatch:
        """Re-construct x_0 from z_n and epsilon.

        Args:
            z_n: Noise images
            n: Noise level
            epsilon: Noise that produced z_n

        Returns:
            The reconstructed x_0
        """

        ##########################################################
        # YOUR CODE HERE
        n_ = n[:, None, None, None]
        x0 = (z_n - torch.sqrt(1 - self.alpha_bar[n_]) * epsilon) / torch.sqrt(self.alpha_bar[n_])
        return x0
        ##########################################################

    @typechecked
    def sample_z_n_previous(
        self, x0: ImageBatch, z_n: ImageBatch, n: NoiseLevel
    ) -> ImageBatch:
        """Sample z_{n-1} given z_n and x_0.

        Args:
            x0: (Estimate of) images
            z_n: Noisy images
            n: Noise level

        Returns:
            A z_{n-1} sample
        """

        ##########################################################
        # YOUR CODE HERE
        n_ = n[:, None, None, None]
        alpha_bar_prev = torch.cat((torch.zeros(1), self.alpha_bar[:-1]))
        mu_tilde = (torch.sqrt(self.alpha[n_]) * (1 - alpha_bar_prev[n_]) / (1 - self.alpha_bar[n_])) * z_n + \
            (torch.sqrt(alpha_bar_prev[n_]) * self.beta[n_]/(1 - self.alpha_bar[n_])) * x0
        z_n_prev = torch.normal(mean=mu_tilde, std=self.beta_tilde[n_])
        return z_n_prev
        ##########################################################

    @torch.no_grad()
    def sample(self, batch_size: int, device: torch.device) -> ImageBatch:
        """Sample new images from scratch by iteratively denoising pure noise.

        Args:
            batch_size: Number of images to generate
            device: Device to generate them on

        Returns:
            Generated images
        """

        ##########################################################
        # YOUR CODE HERE
        z_n = torch.randn(batch_size, 1, 28, 28, device=device)
        for n in reversed(range(1, self.N + 1)):
            noise_level = ((n-1) * torch.ones(batch_size, device=device)).long()
            normalized_noise_level = noise_level / (self.N - 1)
            pred_noise = self.model(z_n, normalized_noise_level)
            x0 = self.estimate_x0(z_n, noise_level, pred_noise)
            if n > 1:
                z_n = self.sample_z_n_previous(x0, z_n, noise_level)
        return x0
        ##########################################################
