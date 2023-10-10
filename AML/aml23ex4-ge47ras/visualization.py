import math

import matplotlib.pyplot as pp
import numpy as np


def visualize_dataset_mnist(dataset, n: int = 10) -> pp.Figure:
    """Plot random images from the MNIST dataset.

    Args:
        dataset: MNIST Dataset instance from torchvision.
    """

    samples = [dataset[i] for i in np.random.choice(len(dataset), n, replace=False)]
    images = [img.squeeze().cpu().numpy() for img, _ in samples]
    labels = [f"Class {y}" for _, y in samples]

    return plot_image_grid(images, labels, ncols=5)


def visualize_mnist_samples(samples, ncols: int = 5) -> pp.Figure:
    """Plot randomly generated MNIST samples.

    Args:
        samples: Samples drawn from the model
    """

    labels = [f"Sample #{i}" for i in range(len(samples))]
    return plot_image_grid(
        samples.squeeze().cpu().detach().numpy(), labels, ncols=ncols
    )


def plot_image_grid(images: np.ndarray, labels: list[str], *, ncols: int) -> pp.Figure:
    nrows = int(math.ceil(len(images) / ncols))

    fig = pp.figure(figsize=(5, 1.5 * nrows))
    axes = fig.subplots(
        nrows,
        ncols,
        gridspec_kw=dict(left=0, right=1, top=1, bottom=0, hspace=0, wspace=0.1),
    )

    for i in range(len(images)):
        ax = axes[i // ncols, i % ncols]
        ax.grid(False)
        ax.set_yticks([])
        ax.set_xticks([])
        ax.imshow(images[i], cmap="gray")
        ax.set_title(labels[i])

    return fig
