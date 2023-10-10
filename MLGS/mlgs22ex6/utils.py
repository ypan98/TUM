import torch
import torchvision.datasets as datasets
import torchvision


def get_mnist_data(train=True):
    return datasets.MNIST(root='./data', train=train, download=True, 
                          transform=torchvision.transforms.Compose([
                              torchvision.transforms.ToTensor(),
                          ]))


def get_device():
    use_cuda = torch.cuda.is_available() 
    return torch.device("cuda" if use_cuda else "cpu")