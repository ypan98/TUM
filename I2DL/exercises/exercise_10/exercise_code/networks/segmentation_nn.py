"""SegmentationNN"""
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchvision import models

class SegmentationNN(pl.LightningModule):

    def __init__(self, num_classes=23, hparams=None):
        super().__init__()
        self.hparams = hparams
        #######################################################################
        #                             YOUR CODE                               #
        #######################################################################
        resnet = models.resnet50(pretrained=True)
        resnet.fc = nn.Identity()
        for params in resnet.parameters():
            params.requires_grad = False
        self.enconder = resnet

        # 2048x1x1 => 23x240x240
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.ConvTranspose2d(2048, 1024, (2,2)),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.ConvTranspose2d(1024, 512, (2, 2)),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.ConvTranspose2d(512, 256, (2, 2)), # out 256x15x15
            nn.ReLU(),

            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.ConvTranspose2d(256, 128, (1,1)),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.ConvTranspose2d(128, 64, (1, 1)),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.ConvTranspose2d(64, 32, (1, 1)),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.ConvTranspose2d(32, 23, (1, 1)), # out 23x240x240
        )
        # models.segmentation.fcn_resnet50(pretrained=True, num_classes=23)
        pass

        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################

    def forward(self, x):
        """
        Forward pass of the convolutional neural network. Should not be called
        manually but by calling a model instance directly.

        Inputs:
        - x: PyTorch input Variable
        """
        #######################################################################
        #                             YOUR CODE                               #
        #######################################################################
        latent_x = self.enconder(x)
        x = self.decoder(latent_x.unsqueeze(2).unsqueeze(3))
        pass

        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################

        return x

class SegmentationNN2(pl.LightningModule):

    def __init__(self, num_classes=23, hparams=None):
        super().__init__()
        self.hparams = hparams
        #######################################################################
        #                             YOUR CODE                              #
        #######################################################################
        self.encoder = models.mobilenet_v2(pretrained=True).features;
        self.decoder = nn.Sequential(
            nn.Conv2d(1280, 400, 2, 2),
            nn.ReLU(),
            nn.Upsample(size=(60, 60)),
            nn.Conv2d(400, 100, 2, 2),
            nn.ReLU(),
            nn.Upsample(size=(240, 240)),
            nn.Conv2d(100, num_classes, 1, 1)
        )

        # models.segmentation.fcn_resnet50(pretrained=True, num_classes=23)
        pass

        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################

    def forward(self, x):
        """
        Forward pass of the convolutional neural network. Should not be called
        manually but by calling a model instance directly.

        Inputs:
        - x: PyTorch input Variable
        """
        #######################################################################
        #                             YOUR CODE                               #
        #######################################################################
        x = self.encoder(x)
        x = self.decoder(x)
        pass
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################

        return x

    @property
    def is_cuda(self):
        """
        Check if model parameters are allocated on the GPU.
        """
        return next(self.parameters()).is_cuda

    def save(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with "*.model".

        Inputs:
        - path: path string
        """
        print('Saving model... %s' % path)
        torch.save(self, path)

        
class DummySegmentationModel(pl.LightningModule):

    def __init__(self, target_image):
        super().__init__()
        def _to_one_hot(y, num_classes):
            scatter_dim = len(y.size())
            y_tensor = y.view(*y.size(), -1)
            zeros = torch.zeros(*y.size(), num_classes, dtype=y.dtype)

            return zeros.scatter(scatter_dim, y_tensor, 1)

        target_image[target_image == -1] = 1

        self.prediction = _to_one_hot(target_image, 23).permute(2, 0, 1).unsqueeze(0)

    def forward(self, x):
        return self.prediction.float()
