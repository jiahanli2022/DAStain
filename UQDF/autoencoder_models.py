from torch import nn as nn
import torch

import torch
import torch.nn as nn


class LargeAutoEncoderUNet(nn.Module):
    def __init__(self):
        super().__init__()

        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=2, padding=1),
            nn.ReLU(True)
        )  # 128x128

        self.enc2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(True)
        )  # 64x64

        self.enc3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.ReLU(True)
        )  # 32x32

        self.enc4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, stride=2, padding=1),
            nn.ReLU(True)
        )  # 16x16

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(True),
        )

        # Decoder
        self.dec4 = nn.Sequential(
            nn.ConvTranspose2d(512+512, 256, 4, stride=2, padding=1),  # concat skip connection
            nn.ReLU(True)
        )  # 32x32

        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(256+256, 128, 4, stride=2, padding=1),
            nn.ReLU(True)
        )  # 64x64

        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(128+128, 64, 4, stride=2, padding=1),
            nn.ReLU(True)
        )  # 128x128

        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(64+64, 3, 4, stride=2, padding=1),
            nn.Sigmoid()
        )  # 256x256

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)

        # Bottleneck
        b = self.bottleneck(e4)

        # Decoder with skip connections
        d4 = self.dec4(torch.cat([b, e4], dim=1))
        d3 = self.dec3(torch.cat([d4, e3], dim=1))
        d2 = self.dec2(torch.cat([d3, e2], dim=1))
        d1 = self.dec1(torch.cat([d2, e1], dim=1))

        return d1

