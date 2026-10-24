# SRResNet (Ledig et al., CVPR 2017): 16 residual blocks, 64 channels, PixelShuffle x2 per octave.
# Trained from scratch with L1 by training/train_srresnet_l1.py (no official checkpoint is used).
# Input/Output: [B, 3, H, W] in [0, 1] (output not clamped).

import math

import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.bn1(residual)
        residual = self.prelu(residual)
        residual = self.conv2(residual)
        residual = self.bn2(residual)
        return x + residual


class SRResNet(nn.Module):
    def __init__(self, upscale_factor=4, num_res_blocks=16):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=9, padding=4, bias=True)
        self.prelu = nn.PReLU()

        self.res_blocks = nn.Sequential(*[ResidualBlock(64) for _ in range(num_res_blocks)])

        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)

        upsample_layers = []
        for _ in range(int(math.log(upscale_factor, 2))):
            upsample_layers.append(nn.Conv2d(64, 256, kernel_size=3, padding=1))
            upsample_layers.append(nn.PixelShuffle(2))
            upsample_layers.append(nn.PReLU())
        self.upsample = nn.Sequential(*upsample_layers)

        self.conv3 = nn.Conv2d(64, 3, kernel_size=9, padding=4)

    def forward(self, x):
        out1 = self.prelu(self.conv1(x))
        out = self.res_blocks(out1)
        out2 = self.bn2(self.conv2(out))
        out = out1 + out2
        out = self.upsample(out)
        return self.conv3(out)
