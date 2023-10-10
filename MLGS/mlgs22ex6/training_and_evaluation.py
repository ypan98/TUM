from typing import Callable, Union, Tuple, List, Dict
import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import Dataset, DataLoader
from tqdm.autonotebook import tqdm
from models import SmoothClassifier


def train_model(model: nn.Module, dataset: Dataset, batch_size: int, 
                device: Union[torch.device, str], loss_function: Callable, 
                optimizer: Optimizer, epochs: int = 1, 
                loss_args: Union[dict, None] = None) -> Tuple[List, List]:
    """
    Train a model on the input dataset.

    Parameters
    ----------
    model: nn.Module
        The input model to be trained.
    dataset: torch.utils.data.Dataset
        The dataset to train on.
    batch_size: int
        The training batch size.
    device: torch.device or str
        The calculation device.
    loss_function: function with signature: (x, y, model, **kwargs) -> 
                   (loss, logits)
        The function used to compute the loss.
    optimizer: Optimizer
        The model's optimizer.
    epochs: int
        Number of epochs to train for. Default: 1.
    loss_args: dict or None
        Additional arguments to be passed to the loss function.

    Returns
    -------
    Tuple containing
        * losses: List[float]. The losses obtained at each step.
        * accuracies: List[float]. The accuracies obtained at each step.
    """
    model.train()
    if loss_args is None:
        loss_args = {}
    losses = []
    accuracies = []
    num_train_batches = torch.tensor(len(dataset) / batch_size)
    num_train_batches = int(torch.ceil(num_train_batches).item())
    for epoch in range(epochs):
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        for x,y in tqdm(train_loader, total=num_train_batches):
            ##########################################################
            # YOUR CODE HERE
            # gradient descent
            optimizer.zero_grad()
            with torch.enable_grad():
                loss, logits = loss_function(x.to(device), y, model, **loss_args)
                loss.backward()
                optimizer.step()
            with torch.no_grad():
                # save loss for every step
                losses.append(loss.item())
                # save accuracy for every step
                (_, class_pred) = logits.max(dim=1)
                matched = class_pred == y
                acc = (matched.sum()/matched.shape[0]).item()
                accuracies.append(acc)
            ##########################################################
    return losses, accuracies


def predict_model(model: nn.Module, dataset: Dataset, batch_size: int, 
                  device: Union[torch.device, str], 
                  attack_function: Union[Callable, None] = None,
                  attack_args: Union[Callable, None] = None) -> float:
    """
    Use the model to predict a label for each sample in the provided dataset. 
    
    Optionally performs an attack via the attack function first.
    
    Parameters
    ----------
    model: nn.Module
        The input model to be used.
    dataset: torch.utils.data.Dataset
        The dataset to predict for.
    batch_size: int
        The batch size.
    device: torch.device or str
        The calculation device.
    attack_function: function or None
        If not None, call the function to obtain a perturbed batch before 
        evaluating the prediction.
    attack_args: dict or None
        Additionall arguments to be passed to the attack function.

    Returns
    -------
    float: the accuracy on the provided dataset.
    """
    model.eval()
    if attack_args is None:
        attack_args = {}
    test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    num_batches = torch.tensor(len(dataset) / batch_size)
    num_batches = int(torch.ceil(num_batches).item())
    predictions = []
    targets = []
    for x, y in tqdm(test_loader, total=num_batches):
        ##########################################################
        # YOUR CODE HERE
        x, y = x.to(device), y.to(device)
        if not attack_function is None:
            x.requires_grad = True
            logits = model(x)
            x = attack_function(logits, x, y, **attack_args)
        with torch.no_grad():
            logits = model(x)
            (_, batch_pred) = logits.max(dim=1)
            predictions.append(batch_pred)
            targets.append(y)
        ##########################################################
    predictions = torch.cat(predictions)
    targets = torch.cat(targets)
    accuracy = (predictions == targets).float().mean().item()
    return accuracy


def evaluate_robustness_smoothing(base_classifier: nn.Module, 
                                  sigma: float, 
                                  dataset: Dataset, 
                                  device: Union[torch.device, str],
                                  num_samples_1: int = 1000, 
                                  num_samples_2: int = 10000, 
                                  alpha: float = 0.05, 
                                  certification_batch_size: float = 5000, 
                                  num_classes: int = 10) -> Dict:
    """
    Evaluate the robustness of a smooth classifier based on the input base 
    classifier via randomized smoothing.
    
    Parameters
    ----------
    base_classifier: nn.Module
        The input base classifier to use in the randomized smoothing process.
    sigma: float
        The variance to use for the Gaussian noise samples.
    dataset: Dataset
        The input dataset to predict on.
    device: torch.device or str
        The calculation device.
    num_samples_1: int
        The number of samples used to determine the most likely class.
    num_samples_2: int
        The number of samples used to perform the certification.
    alpha: float
        The desired confidence level that the top class is indeed the most 
        likely class. E.g. alpha=0.05 means that the expected error rate must 
        not be larger than 5%.
    certification_batch_size: int
        The batch size to use during the certification, i.e. how many noise 
        samples to classify in parallel.
    num_classes: int
        The number of classes.

    Returns
    -------
    Dict containing the following keys:
        * abstains: int. The number of times the smooth classifier abstained, 
            i.e. could not certify the input sample to the desired 
            confidence level.
        * false_predictions: int. The number of times the prediction could be 
            certified but was not correct.
        * correct_certified: int. The number of times the prediction could be 
            certified and was correct.
        * avg_radius: float. The average radius for which the predictions could 
            be certified.
    """
    model = SmoothClassifier(base_classifier=base_classifier, sigma=sigma, 
                             num_classes=num_classes)
    model.eval()
    test_loader = DataLoader(dataset, batch_size=1, shuffle=False)

    abstains = 0
    false_predictions = 0
    correct_certified = 0
    radii = []
    for x, y in tqdm(test_loader, total=len(dataset)):
        ##########################################################
        # YOUR CODE HERE
        x, y = x.to(device), y.to(device)
        top_class, radius = model.certify(x, n0=num_samples_1, num_samples=num_samples_2, alpha=alpha,
                                          batch_size=certification_batch_size)
        if top_class == model.ABSTAIN:
            abstains += 1
        else:
            if top_class == y:
                correct_certified += 1
            else:
                false_predictions += 1
            radii.append(radius)
        ##########################################################
    avg_radius = torch.tensor(radii).mean().item()
    return dict(abstains=abstains, false_predictions=false_predictions, 
                correct_certified=correct_certified, avg_radius=avg_radius)