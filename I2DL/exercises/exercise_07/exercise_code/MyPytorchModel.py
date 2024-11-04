import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision
import torchvision.transforms as transforms
import numpy as np
import math

class MyPytorchModel(pl.LightningModule):
    
    def __init__(self, hparams, input_size=3 * 32 * 32, num_classes=10):
        super().__init__()

        # set hyperparams
        self.hparams = hparams

        ########################################################################
        # TODO: Initialize your model!                                         #
        ########################################################################
        self.model = nn.Sequential(
            nn.Linear(input_size, self.hparams["n_hidden"]),
            nn.Tanh(),            
            nn.Linear(self.hparams["n_hidden"], self.hparams["n_hidden"]),
            nn.Tanh(),
            nn.Linear(self.hparams["n_hidden"], self.hparams["n_hidden"]),
            nn.Tanh(),
            nn.Linear(self.hparams["n_hidden"], num_classes),
        )

        for name, param in self.model.named_parameters():
            if name.endswith(".bias"):
                param.data.fill_(0)
            else:
                param.data.normal_(0, math.sqrt(6) / math.sqrt(param.shape[0]+param.shape[1]))

        pass

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################

    def forward(self, x):

        # x.shape = [batch_size, 3, 32, 32] -> flatten the image first
        x = x.view(x.shape[0], -1)

        # feed x into model!
        x = self.model(x)

        return x
    
    def general_step(self, batch, batch_idx, mode):
        images, targets = batch

        # forward pass
        out = self.forward(images)

        # loss
        loss = F.cross_entropy(out, targets)
        

        preds = out.argmax(axis=1)
        n_correct = (targets == preds).sum()
        return loss, n_correct
    
    def general_end(self, outputs, mode):
        # average over all batches aggregated during one epoch
        avg_loss = torch.stack([x[mode + '_loss'] for x in outputs]).mean()
        total_correct = torch.stack([x[mode + '_n_correct'] for x in outputs]).sum().cpu().numpy()
        acc = total_correct / len(self.sampler[mode])
        return avg_loss, acc

    def training_step(self, batch, batch_idx):
        loss, n_correct = self.general_step(batch, batch_idx, "train")
        #tensorboard_logs = {'loss': loss}
        self.log('loss',loss)
        return {'loss': loss, 'train_n_correct':n_correct}#, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        loss, n_correct = self.general_step(batch, batch_idx, "val")
        self.log('val_loss',loss)
        return {'val_loss': loss, 'val_n_correct':n_correct}
    
    def test_step(self, batch, batch_idx):
        loss, n_correct = self.general_step(batch, batch_idx, "test")
        return {'test_loss': loss, 'test_n_correct':n_correct}

    def validation_epoch_end(self, outputs):
        avg_loss, acc = self.general_end(outputs, "val")
        #print("Val-Acc={}".format(acc))
        #tensorboard_logs = {'val_loss': avg_loss}
        self.log('val_loss',avg_loss)
        self.log('val_acc',acc)
        return {'val_loss': avg_loss, 'val_acc': acc}#, 'log': tensorboard_logs}

    def prepare_data(self):

        # create dataset
        CIFAR_ROOT = "../datasets/cifar10"
        my_transform = None
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        ########################################################################
        # TODO: Define your transforms (convert to tensors, normalize).        #
        # If you want, you can also perform data augmentation!                 #
        ########################################################################
        my_transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(mean,std),
                                transforms.RandomHorizontalFlip(0.25),
                                transforms.RandomVerticalFlip(0.25),
                                transforms.RandomApply([transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5))], p=0.25)
                                ]) 
        pass

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################
        train_val_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
        cifar_complete_augmented = torchvision.datasets.ImageFolder(root=CIFAR_ROOT, transform=my_transform)
        cifar_complete_train_val = torchvision.datasets.ImageFolder(root=CIFAR_ROOT, transform=train_val_transform)

        N = len(cifar_complete_augmented)        
        num_train, num_val = int(N*0.6), int(N*0.2)
        np.random.seed(0)
        indices = np.random.permutation(N)
        train_idx, val_idx, test_idx = indices[:num_train], indices[num_train:num_train+num_val], indices[num_train+num_val:]
        train_sampler = SubsetRandomSampler(train_idx)
        val_sampler = SubsetRandomSampler(val_idx)
        test_sampler= SubsetRandomSampler(test_idx)
        self.sampler = {"train": train_sampler, "val": val_sampler, "test": test_sampler}


        # assign to use in dataloaders
        self.dataset = {}
        self.dataset["train"], self.dataset["val"], self.dataset["test"] = cifar_complete_augmented, cifar_complete_train_val, cifar_complete_train_val

    def train_dataloader(self):
        return DataLoader(self.dataset["train"], batch_size=self.hparams["batch_size"], sampler=self.sampler["train"])

    def val_dataloader(self):
        return DataLoader(self.dataset["val"], batch_size=self.hparams["batch_size"], sampler=self.sampler["val"])
    
    def test_dataloader(self):
        return DataLoader(self.dataset["test"], batch_size=self.hparams["batch_size"], sampler=self.sampler["test"])

    def configure_optimizers(self):

        optim = None
        ########################################################################
        # TODO: Define your optimizer.                                         #
        ########################################################################
        optim = torch.optim.SGD(self.model.parameters(), self.hparams["learning_rate"], momentum=0.9)

        pass

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################
        return optim

    def getTestAcc(self, loader = None):
        self.model.eval()
        self.model = self.model.to(self.device)

        if not loader: loader = self.test_dataloader()

        scores = []
        labels = []

        for batch in loader:
            X, y = batch
            X = X.to(self.device)
            score = self.forward(X)
            scores.append(score.detach().cpu().numpy())
            labels.append(y.detach().cpu().numpy())

        scores = np.concatenate(scores, axis=0)
        labels = np.concatenate(labels, axis=0)

        preds = scores.argmax(axis=1)
        acc = (labels == preds).mean()
        return preds, acc
