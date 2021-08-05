# Adapted from https://github.com/pytorch/vision and https://github.com/VainF/DeepLabV3Plus-Pytorch

import torch
import torch.nn as nn
import torch.nn.functional as F

from retinal_fundus_encoder.resnet import get_resnet_encoder, RESNET_OUT_CHANNELS


class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        super(ASPPConv, self).__init__(*modules)


class ASPPPooling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False)
        )
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        size = x.shape[-2:]
        x = self.pool(x)
        if x.shape[0] > 1:
            x = self.batch_norm(x)
        x = self.relu(x)

        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)


class ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates):
        super(ASPP, self).__init__()
        out_channels = 256
        modules = []
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)))

        rate1, rate2, rate3 = tuple(atrous_rates)
        modules.append(ASPPConv(in_channels, out_channels, rate1))
        modules.append(ASPPConv(in_channels, out_channels, rate2))
        modules.append(ASPPConv(in_channels, out_channels, rate3))
        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(5 * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
        )

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)


class DeepLabHeadV3Plus(nn.Module):
    def __init__(self, in_channels, low_level_channels, aspp_dilate):
        super(DeepLabHeadV3Plus, self).__init__()
        self.project = nn.Sequential(
            nn.Conv2d(low_level_channels, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
        )

        self.aspp = ASPP(in_channels, aspp_dilate)

        self.conv = nn.Sequential(
            nn.Conv2d(304, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self._init_weight()

    def forward(self, features):
        low_level_feature = self.project(features[0])
        output_feature = self.aspp(features[1])
        output_feature = F.interpolate(
            output_feature,
            size=low_level_feature.shape[2:],
            mode='bilinear',
            align_corners=False
        )

        return self.conv(torch.cat([low_level_feature, output_feature], dim=1))

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class DeepLabv3Plus(nn.Module):
    def __init__(self, n_channels, encoder_backbone='resnet-50', imagenet_init=True):
        """
        Args:
            n_channels (int): Number of channels in input images.
            backbone (str): Name of the encoder backbone.
            imagenet_init (bool): If available, choose whether to initialize the encoder with weights
                                  pre-trained on ImageNet.
            use_dilation (bool): Whether to replace strides in the last two blocks with dilation.
        """
        super().__init__()

        if n_channels != 3:
            raise ValueError("`n_channels` for DeepLabv3+ must be equal to 3.")

        self.encoder = get_resnet_encoder(encoder_backbone, imagenet_init=imagenet_init, use_dilation=True)
        self.decoder = DeepLabHeadV3Plus(
            RESNET_OUT_CHANNELS[encoder_backbone],
            RESNET_OUT_CHANNELS[encoder_backbone] // 8,
            [12, 24, 36]
        )
