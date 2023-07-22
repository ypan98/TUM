import torch
import torch.nn as nn


class ThreeDEPN(nn.Module):
    def __init__(self):
        super().__init__()

        self.num_features = 80

        # TODO: 4 Encoder layers
        self.enconder_l1 = nn.Sequential(
            nn.Conv3d(in_channels=2, out_channels=self.num_features, kernel_size=(4,4,4), stride=(2,2,2), padding=1),
            nn.LeakyReLU(0.2)
        )
        self.enconder_l2 = nn.Sequential(
            nn.Conv3d(in_channels=self.num_features, out_channels=self.num_features*2, kernel_size=(4,4,4), stride=(2,2,2), padding=1),
            nn.BatchNorm3d(self.num_features*2),
            nn.LeakyReLU(0.2)
        )
        self.enconder_l3 = nn.Sequential(
            nn.Conv3d(in_channels=self.num_features*2, out_channels=self.num_features * 4, kernel_size=(4,4,4), stride=(2,2,2), padding=1),
            nn.BatchNorm3d(self.num_features * 4),
            nn.LeakyReLU(0.2)
        )
        self.enconder_l4 = nn.Sequential(
            nn.Conv3d(in_channels=self.num_features * 4, out_channels=self.num_features * 8, kernel_size=(4,4,4)),
            nn.BatchNorm3d(self.num_features * 8),
            nn.LeakyReLU(0.2)
        )

        # TODO: 2 Bottleneck layers
        self.bottleneck = nn.Sequential(
            nn.Linear(in_features=self.num_features * 8, out_features=self.num_features * 8),
            nn.ReLU(),
            nn.Linear(in_features=self.num_features * 8, out_features=self.num_features * 8),
            nn.ReLU()
        )

        # TODO: 4 Decoder layers
        self.decoder_l1 = nn.Sequential(
            nn.ConvTranspose3d(in_channels=self.num_features * 8 * 2, out_channels=self.num_features * 4, kernel_size=(4,4,4)),
            nn.BatchNorm3d(self.num_features * 4),
            nn.ReLU()
        )
        self.decoder_l2 = nn.Sequential(
            nn.ConvTranspose3d(in_channels=self.num_features * 4 * 2, out_channels=self.num_features * 2, kernel_size=(4,4,4), stride=(2,2,2), padding=(1,1,1)),
            nn.BatchNorm3d(self.num_features * 2),
            nn.ReLU()
        )
        self.decoder_l3 = nn.Sequential(
            nn.ConvTranspose3d(in_channels=self.num_features * 2 * 2, out_channels=self.num_features, kernel_size=(4,4,4), stride=(2,2,2), padding=(1,1,1)),
            nn.BatchNorm3d(self.num_features),
            nn.ReLU()
        )
        self.decoder_l4 = nn.ConvTranspose3d(in_channels=self.num_features * 2, out_channels=1, kernel_size=(4, 4, 4), stride=(2, 2, 2), padding=(1, 1, 1))


    def forward(self, x):
        b = x.shape[0]
        # Encode
        # TODO: Pass x though encoder while keeping the intermediate outputs for the skip connections
        x_e1 = self.enconder_l1(x)
        x_e2 = self.enconder_l2(x_e1)
        x_e3 = self.enconder_l3(x_e2)
        x_e4 = self.enconder_l4(x_e3)

        # Reshape and apply bottleneck layers
        x = x_e4.view(b, -1)
        x = self.bottleneck(x)
        x = x.view(x.shape[0], x.shape[1], 1, 1, 1)
        # Decode
        # TODO: Pass x through the decoder, applying the skip connections in the process
        x = self.decoder_l1(torch.cat((x, x_e4), dim=1))
        x = self.decoder_l2(torch.cat((x, x_e3), dim=1))
        x = self.decoder_l3(torch.cat((x, x_e2), dim=1))
        x = self.decoder_l4(torch.cat((x, x_e1), dim=1))
        x = torch.squeeze(x, dim=1)
        # TODO: Log scaling
        x = torch.abs(x)
        x = torch.log1p(x)
        return x
