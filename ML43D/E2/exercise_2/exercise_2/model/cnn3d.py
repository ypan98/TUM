"""3D CNN network implementation"""
from torch import nn
import torch


class MLPConv(nn.Module):
    """MLP Conv layer as described in Section 4.2 / Fig. 3 of https://arxiv.org/pdf/1604.03265.pdf"""
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        """
        :param in_channels: number of input channels to the first conv layer
        :param out_channels: number of output channels for conv layers
        :param kernel_size: kernel_size of first conv layer
        :param stride: stride of first conv layer
        """
        super().__init__()
        # TODO: Define MLPConv model as nn.Sequential as described in the paper (Conv3d, ReLU, Conv3D, ReLU, Conv3D, ReLU)
        # The first conv has kernel_size and stride provided as the parameters, rest of the convs have 1x1x1 filters, with default stride
        self.model = nn.Sequential(
            nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride),
            nn.ReLU(),
            nn.Conv3d(in_channels=out_channels, out_channels=out_channels, kernel_size=(1,1,1)),
            nn.ReLU(),
            nn.Conv3d(in_channels=out_channels, out_channels=out_channels, kernel_size=(1,1,1)),
            nn.ReLU()
        )

    def forward(self, x):
        """
        :param x: tensor of shape [B, C, D, H, W], where B = batch size, C = num feature channels, D = depth of volume, H and W = height and width of volume
        """
        return self.model(x)


class ThreeDeeCNN(nn.Module):
    """
    3DCNN Network as described in Section 4.2 / Fig. 3 of https://arxiv.org/pdf/1604.03265.pdf.
    However, inputs here are 32x32x32 in contrast to original figure. Note that architecture does not change inspite of this.
    """

    def __init__(self, n_classes):
        """
        :param n_classes: Number of classes to classified, e.g. for our shapenet experiments we have a 13 class classification problem
        """
        super().__init__()

        # TODO: Define backbone as sequence of 3 MLPConvs as per the paper
        self.backbone = nn.Sequential(
            MLPConv(1, 48, 6, 2),
            MLPConv(48, 160, 5, 2),
            MLPConv(160, 512, 3, 2)
        )

        self.feature_cube_side = 2  # side of resulting volume after last MLPConv layer

        # predictors for partial objects, i.e. for each of the elements of the 2x2x2 volume from the backbone
        self.partial_predictors = nn.ModuleList()
        for i in range(8):
            self.partial_predictors.append(
                # TODO: partial predictor linear layers as per the paper
                nn.Linear(in_features=512, out_features=n_classes),
            )

        # TODO: add predictor for full 2x2x2 feature volume
        self.full_predictor = nn.Sequential(
            nn.Linear(in_features=2 * 2 * 2 * 512, out_features=2048),
            nn.Linear(in_features=2048, out_features=2048),
            nn.Linear(in_features=2048, out_features=n_classes)
        )

    def forward(self, x):
        """
            :param x: tensor of shape [B, C, D, H, W], where B = batch size, C = 1, D = 32, H = 32 and W = 32
            :return: a tensor of shape [B, 9, n_classes], i.e. for each shape in the batch, the class scores for the whole object (1) and class scores for partial object (8)
        """
        batch_size = x.shape[0]

        # TODO: Get backbone features
        backbone_features = self.backbone(x)

        predictions_partial = []
        # get prediction for each of the partial objects
        for d in range(backbone_features.shape[2]):
            for h in range(backbone_features.shape[3]):
                for w in range(backbone_features.shape[4]):
                    partial_predictor = self.partial_predictors[d * self.feature_cube_side ** 2 + h * self.feature_cube_side + w]

                    # TODO: get prediction for object for backbone feature at d, h, w
                    current_part_features = backbone_features[:, :, d, h, w]
                    partial_object_prediction = partial_predictor(current_part_features.view(batch_size, -1))

                    predictions_partial.append(partial_object_prediction)

        # TODO: Get prediction for whole object
        full_prediction = self.full_predictor(backbone_features.view(batch_size, -1))

        return torch.stack([full_prediction] + predictions_partial, dim=1)
