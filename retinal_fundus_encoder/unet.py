# Adapted from https://github.com/milesial/Pytorch-UNet

import torch
import torch.nn as nn
import torch.nn.functional as F

from retinal_fundus_encoder.resnet import RESNET_CONSTRUCTORS


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels_1, in_channels_2, out_channels, bilinear=True):
        super().__init__()

        if bilinear:
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.Conv2d(in_channels_1, in_channels_1 // 2, kernel_size=1)
            )
        else:
            self.up = nn.ConvTranspose2d(in_channels_1, in_channels_1 // 2, kernel_size=2, stride=2)

        self.conv = DoubleConv(in_channels_1 // 2 + in_channels_2, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, n_channels, bilinear=True, encoder_backbone='default', imagenet_init=True):
        """
        Args:
            n_channels (int): Number of channels in input images.
            bilinear (bool): Choose whether to use bilinear interpolation as upsampling method.
            encoder_backbone (str): Name of the encoder backbone.
            imagenet_init (bool): If available, choose whether to initialize the encoder with weights
                                  pre-trained on ImageNet.
        """
        super().__init__()

        self.encoder = None
        self.decoder = None
        if encoder_backbone == 'default':
            self.encoder = nn.ModuleDict({
                'down0': DoubleConv(n_channels, 64),
                'down1': Down(64, 128),
                'down2': Down(128, 256),
                'down3': Down(256, 512),
                'down4': Down(512, 1024)
            })
            self.decoder = nn.ModuleDict({
                'up1': Up(1024, 512, 512, bilinear),
                'up2': Up(512, 256, 256, bilinear),
                'up3': Up(256, 128, 128, bilinear),
                'up4': Up(128, 64, 64, bilinear),
            })
        elif encoder_backbone in RESNET_CONSTRUCTORS:
            try:
                assert n_channels == 3
            except AssertionError:
                raise AssertionError("ResNet backbone currently only support 3 input channels.")

            resnet = RESNET_CONSTRUCTORS[encoder_backbone]
            modules = list(resnet(pretrained=imagenet_init).children())
            self.encoder = nn.ModuleDict({
                'down0': nn.Identity(),
                'down1': nn.Sequential(*modules[:3]),
                'down2': nn.Sequential(*modules[3:5]),
                'down3': modules[5],
                'down4': modules[6],
                'down5': modules[7]
            })
            if encoder_backbone in ['resnet-18', 'resnet-34']:
                self.decoder = nn.ModuleDict({
                    'up1': Up(512, 256, 256, bilinear),
                    'up2': Up(256, 128, 128, bilinear),
                    'up3': Up(128, 64, 64, bilinear),
                    'up4': Up(64, 64, 32, bilinear),
                    'up5': Up(32, n_channels, 16, bilinear)
                })
            else:
                self.decoder = nn.ModuleDict({
                    'up1': Up(2048, 1024, 1024, bilinear),
                    'up2': Up(1024, 512, 512, bilinear),
                    'up3': Up(512, 256, 256, bilinear),
                    'up4': Up(256, 64, 128),
                    'up5': Up(128, n_channels, 64, bilinear)
                })
        else:
            raise ValueError(f"Unsupported backbone '{encoder_backbone}'.")
