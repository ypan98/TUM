"""Models for facial keypoint detection"""

import torch
import torch.nn as nn
import pytorch_lightning as pl

import os
import torch.nn.functional as F
from exercise_code.data.facial_keypoints_dataset import FacialKeypointsDataset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

class KeypointModel(pl.LightningModule):
    """Facial keypoint detection model"""
    def __init__(self, hparams):
        """
        Initialize your model from a given dict containing all your hparams
        Warning: Don't change the method declaration (i.e. by adding more
            arguments), otherwise it might not work on the submission server
        """
        super(KeypointModel, self).__init__()
        self.hparams = hparams
        ########################################################################
        # TODO: Define all the layers of your CNN, the only requirements are:  #
        # 1. The network takes in a batch of images of shape (Nx1x96x96)       #
        # 2. It ends with a linear layer that represents the keypoints.        #
        # Thus, the output layer needs to have shape (Nx30),                   #
        # with 2 values representing each of the 15 keypoint (x, y) pairs      #
        #                                                                      #
        # Some layers you might consider including:                            #
        # maxpooling layers, multiple conv layers, fully-connected layers,     #
        # and other layers (such as dropout or batch normalization) to avoid   #
        # overfitting.                                                         #
        ########################################################################
        layer_list = [
            nn.Conv2d(in_channels=1, out_channels=hparams["conv_layers_c"][0], kernel_size=hparams["conv_layers_k"][0]),
            nn.BatchNorm2d(hparams["conv_layers_c"][0]),
            nn.ReLU(),
            nn.MaxPool2d(hparams["max_poolings_k"]),
            nn.Dropout2d(hparams["dropout_p"]),
            ]
        for i in range(1, len(hparams["conv_layers_c"])):
            curr_list = [
                nn.Conv2d(in_channels=hparams["conv_layers_c"][i-1], out_channels=hparams["conv_layers_c"][i], kernel_size=hparams["conv_layers_k"][i]),
                nn.BatchNorm2d(hparams["conv_layers_c"][i]),
                nn.ReLU(),
                nn.MaxPool2d(hparams["max_poolings_k"]),
                nn.Dropout2d(hparams["dropout_p"]),
            ]
            layer_list = [*layer_list, *curr_list]

        layer_list = [
            *layer_list,
            nn.Flatten(),
            nn.Linear(6400, 30) # 256 * (5,5)
        ]

        self.layers = nn.Sequential(*layer_list)
        pass

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################

    def forward(self, x):
        # check dimensions to use show_keypoint_predictions later
        if x.dim() == 3:
            x = torch.unsqueeze(x, 0)
        ########################################################################
        # TODO: Define the forward pass behavior of your model                 #
        # for an input image x, forward(x) should return the                   #
        # corresponding predicted keypoints                                    #
        ########################################################################
        x = self.layers.forward(x)

        pass

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################
        return x

    def prepare_data(self):

        download_url = 'https://vision.in.tum.de/webshare/g/i2dl/facial_keypoints.zip'
        i2dl_exercises_path = os.path.dirname(os.path.abspath(os.getcwd()))
        data_root = os.path.join(i2dl_exercises_path, "datasets", "facial_keypoints")
        my_transform = transforms.Compose([
                            transforms.ToTensor(),
                            # transforms.Normalize(0.5, 0.5),
                        ]) 

        train_dataset = FacialKeypointsDataset(
            train=True,
            transform=my_transform,
            root=data_root,
            download_url=download_url
        )
        val_dataset = FacialKeypointsDataset(
            train=False,
            transform=transforms.ToTensor(),
            root=data_root,
        )
        self.data = {
            'train': train_dataset,
            'val': val_dataset
        }
    
    def train_dataloader(self):
        return DataLoader(self.data['train'], shuffle=True, batch_size=self.hparams['batch_size'])

    def val_dataloader(self):
        return DataLoader(self.data['val'], batch_size=self.hparams['batch_size'])

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), self.hparams["lr"], eps=1e-08, weight_decay=0, amsgrad=False)
    
    def general_step(self, batch, batch_idx, mode):
        images = batch["image"]
        targets = batch["keypoints"]
        out = self.forward(images)
        out_reshaped = out.view(-1, 15,2) # (30) -> (15, 2) to match targets shape 
        loss_fn = nn.MSELoss()
        loss = loss_fn(out_reshaped, targets)
        score = 1.0 / (2 * loss)
        return loss, score
    
    def general_end(self, outputs, mode):
        loss_map = {
            'train': "loss",
            "val": "val_loss"
        }
        avg_loss = torch.stack([x[loss_map[mode]] for x in outputs]).mean()
        score = 1/(2*avg_loss)
        return avg_loss, score

    def training_step(self, batch, batch_idx):
        loss, score = self.general_step(batch, batch_idx, "train")
        self.log('train_score', score)
        return {'loss': loss, 'train_score':score}

    def validation_step(self, batch, batch_idx):
        loss, score = self.general_step(batch, batch_idx, "val")
        self.log('val_score', score)
        return {'val_loss': loss, 'val_score':score}

    def training_epoch_end(self, outputs):
        avg_loss, score = self.general_end(outputs, "train")
        self.log('loss_train_epoch', avg_loss)
        return None
        return {'train_loss_epoch': avg_loss, 'train_score_epoch': score}

    def validation_epoch_end(self, outputs):
        avg_loss, score = self.general_end(outputs, "val")
        self.log('loss_val_epoch', avg_loss)
        return {'val_loss_epoch': avg_loss, 'val_score_epoch': score}



















class DummyKeypointModel(pl.LightningModule):
    """Dummy model always predicting the keypoints of the first train sample"""
    def __init__(self):
        super().__init__()
        self.prediction = torch.tensor([[
            0.4685, -0.2319,
            -0.4253, -0.1953,
            0.2908, -0.2214,
            0.5992, -0.2214,
            -0.2685, -0.2109,
            -0.5873, -0.1900,
            0.1967, -0.3827,
            0.7656, -0.4295,
            -0.2035, -0.3758,
            -0.7389, -0.3573,
            0.0086, 0.2333,
            0.4163, 0.6620,
            -0.3521, 0.6985,
            0.0138, 0.6045,
            0.0190, 0.9076,
        ]])

    def forward(self, x):
        return self.prediction.repeat(x.size()[0], 1, 1, 1)
