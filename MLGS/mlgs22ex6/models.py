from math import ceil
from typing import Tuple

from scipy.stats import norm, binom_test
from statsmodels.stats.proportion import proportion_confint
import torch
from torch import nn


class ConvNN(nn.Module):
    """
    A simple convolutional neural network for image classification on MNIST.
    """

    def __init__(self):
        super(ConvNN, self).__init__()
        self.sequential = nn.Sequential(
                             nn.Conv2d(1, 5, 5),
                             nn.ReLU(),
                             nn.BatchNorm2d(5),
                             nn.MaxPool2d(2),
                             nn.Conv2d(5, 5, 5),
                             nn.ReLU(),
                             nn.MaxPool2d(2),
                             nn.Flatten(),
                             nn.Linear(80, 10),
                            )

    def forward(self, input):
        assert input.min() >= 0 and input.max() <= 1.
        return self.sequential(input)

    def device(self):
        """
        Convenience function returning the device the model is located on.
        """
        return next(self.parameters()).device


def lower_confidence_bound(num_class_A: int, num_samples: int, alpha: float) \
                                                                    -> float:
    """
    Computes a lower bound on the probability of the event occuring in a
    Bernoulli distribution.

    Parameters
    ----------
    num_class_A: int
        The number of times the event occured in the samples.
    num_samples: int
        The total number of samples from the bernoulli distribution.
    alpha: float
        The desired confidence level, e.g. 0.05.

    Returns
    -------
    lower_bound: float
        The lower bound on the probability of the event occuring in a Bernoulli
        distribution.

    """
    return proportion_confint(num_class_A, num_samples, alpha=2 * alpha,
                              method="beta")[0]


class SmoothClassifier(nn.Module):
    """
    Randomized smoothing classifier.
    """

    # to abstain, Smooth returns this int
    ABSTAIN = -1

    def __init__(self, base_classifier: nn.Module, num_classes: int,
                 sigma: float):
        """
        Constructor for SmoothClassifier.
        Parameters
        ----------
        base_classifier: nn.Module
            The base classifier (i.e. f(x)) that maps an input sample to a
            logit vector.
        num_classes: int
            The number of classes.
        sigma: float
            The variance used for the Gaussian perturbations.
        """
        super(SmoothClassifier, self).__init__()
        self.base_classifier = base_classifier
        self.num_classes = num_classes
        self.sigma = sigma

    def device(self):
        return self.base_classifier.device()

    def certify(self, inputs: torch.Tensor, n0: int, num_samples: int,
                alpha: float, batch_size: int) -> Tuple[int, float]:
        """
        Certify the input sample using randomized smoothing.

        Uses lower_confidence_bound to get a lower bound on p_A, the
        probability of the top class.

        Parameters
        ----------
        inputs: torch.Tensor of shape [1, C, N, N], where C is the number of
                channels and N is the image width/height.
            The input image to certify.
        n0: int
            Number of samples to determine the most likely class.
        num_samples: int
            Number of samples to use for the robustness certification.
        alpha: float
            The confidence level, e.g. 0.05 for an expected error rate of 5%.
        batch_size: int
           The batch size to use during the certification, i.e. how many noise
           samples to classify in parallel.

        Returns
        -------
        Tuple containing:
            * top_class: int. The predicted class g(x) of the input sample x.
                         Returns -1 in case the classifier abstains because the
                         desired confidence level could not be reached.
            * radius: float. The radius for which the prediction can be
                      certified. Is zero in case the classifier abstains.
        """
        self.base_classifier.eval()

        ##########################################################
        # YOUR CODE HERE
        class_counts_small_n = self._sample_noise_predictions(inputs, n0, batch_size);
        most_likely_class = torch.argmax(class_counts_small_n).item()
        class_counts_large_n = self._sample_noise_predictions(inputs, num_samples, batch_size);
        most_likely_class_count = class_counts_large_n[most_likely_class].item()
        p_A_lower_bound = lower_confidence_bound(most_likely_class_count, num_samples, alpha)
        ##########################################################

        if p_A_lower_bound < 0.5:
            return SmoothClassifier.ABSTAIN, 0.0
        else:
            ##########################################################
            # YOUR CODE HERE
            top_class = most_likely_class
            radius = self.sigma*norm.ppf(p_A_lower_bound)
            ##########################################################
            return top_class, radius

    def predict(self, inputs: torch.tensor, num_samples: int, alpha: float,
                batch_size: int) -> int:
        """
        Predict a label for the input sample via the smooth classifier g(x).

        Uses the test binom_test(count1, count1+count2, p=0.5) > alpha to
        determine whether the top class is the winning class with at least the
        confidence level alpha.

        Parameters
        ----------
        inputs: torch.Tensor of shape [1, C, N, N], where C is the number of
                channels and N is the image width/height.
            The input image to predict.
        num_samples: int
            The number of samples to draw in order to determine the most likely
            class.
        alpha: float
            The desired confidence level that the top class is indeed the most
            likely class. E.g. alpha=0.05 means that the expected error rate
            must not be larger than 5%.
        batch_size: int
            The batch size to use during the prediction, i.e. how many noise
            samples to classify in parallel.

        Returns
        -------
        int: the winning class or -1 in case the desired confidence level could
             not be reached.
        """
        self.base_classifier.eval()
        class_counts = self._sample_noise_predictions(inputs, num_samples,
                                                      batch_size).cpu()
        ##########################################################
        # YOUR CODE HERE
        sorted_class_counts, _ = torch.sort(class_counts)
        p_val = binom_test(sorted_class_counts[0], sorted_class_counts[0]+sorted_class_counts[1], p=0.5)
        largest_class = torch.argmax(class_counts)
        return largest_class.item() if p_val > alpha else SmoothClassifier.ABSTAIN
        ##########################################################

    def _sample_noise_predictions(self, inputs: torch.tensor, num_samples: int,
                                  batch_size: int) -> torch.Tensor:
        """
        Sample random noise perturbations for the input sample and count the
        predicted classes of the base classifier.

        Note: This function clamps the distorted samples in the valid range,
              i.e. [0,1].

        Parameters
        ----------
        inputs: torch.Tensor of shape [1, C, N, N], where C is the number of
                channels and N is the image width/height.
            The input image to predict.
        num_samples: int
            The number of samples to draw.
        batch_size: int
            The batch size to use during the prediction, i.e. how many noise
            samples to classify in parallel.

        Returns
        -------
        torch.Tensor of shape [K,], where K is the number of classes.
            Each entry of the tensor contains the number of times the base
            classifier predicted the corresponding class for the noise samples.
        """
        num_remaining = num_samples
        with torch.no_grad():
            classes = torch.arange(self.num_classes).to(self.device())
            class_counts = torch.zeros([self.num_classes], dtype=torch.long,
                                       device=self.device())
            for it in range(ceil(num_samples / batch_size)):
                this_batch_size = min(num_remaining, batch_size)
                ##########################################################
                # YOUR CODE HERE
                num_remaining -= this_batch_size
                batch_samples = inputs.repeat([this_batch_size, 1, 1, 1])
                batch_logits = self.forward(batch_samples.to(self.device()))
                batch_predictions = torch.argmax(batch_logits, dim=1)
                (unique, unique_counts) = torch.unique(batch_predictions, return_counts=True);
                for i,c in zip(unique, unique_counts):
                    class_counts[i.item()] += c
                ##########################################################
        return class_counts

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Make a single prediction for the input batch using the base classifier
        and random Gaussian noise.

        Note: This function clamps the distorted samples in the valid range,
              i.e. [0,1].

        Parameters
        ----------
        inputs: torch.Tensor of shape [B, C, N, N], where B is the batch size,
                C is the number of channels, and N is the image width/height.
            The input batch of images to predict.

        Returns
        -------
        torch.Tensor of shape [B, K]
            The logits for each input image.
        """
        noise = torch.randn_like(inputs) * self.sigma
        return self.base_classifier((inputs + noise).clamp(0, 1))
