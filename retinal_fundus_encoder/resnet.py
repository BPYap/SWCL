from collections import OrderedDict

import torch
import torch.nn as nn
import torchvision
from torchvision.models._utils import IntermediateLayerGetter

from big_transfer.bit_pytorch.models import ResNetV2, tf2th

RESNET_CONSTRUCTORS = {
    'resnet-18': lambda **kw: _add_dilation(torchvision.models.resnet.resnet18, **kw),
    'resnet-34': lambda **kw: _add_dilation(torchvision.models.resnet.resnet34, **kw),
    'resnet-50': torchvision.models.resnet.resnet50,
    'resnet-101': torchvision.models.resnet.resnet101,
    'resnet-152': torchvision.models.resnet.resnet152
}

RESNETV2_CONSTRUCTORS = {
    'resnetv2-50x1': lambda **kw: _ResNetV2([3, 4, 6, 3], 1, **kw)
}

RESNET_OUT_CHANNELS = {
    'resnet-18': 512,
    'resnet-34': 512,
    'resnet-50': 2048,
    'resnet-101': 2048,
    'resnet-152': 2048,
    'resnetv2-50x1': 2048
}


def _add_dilation(model_fn, **kwargs):
    """Modify BasicBlock in ResNet to support dilation."""
    replace_stride_with_dilation = None
    if "replace_stride_with_dilation" in kwargs:
        replace_stride_with_dilation = kwargs["replace_stride_with_dilation"]
        del kwargs["replace_stride_with_dilation"]
    model = model_fn(**kwargs)
    if replace_stride_with_dilation is not None:
        dilation = prev_dilation = 1
        for dilate, layer in zip(replace_stride_with_dilation, [model.layer2, model.layer3, model.layer4]):
            if dilate:
                dilation *= 2
                layer[0].downsample[0].stride = 1
                layer[0].downsample[0].dilation = (prev_dilation, prev_dilation)
                for block in layer:
                    block.conv1.stride = 1
                    block.conv1.dilation = (prev_dilation, prev_dilation)
                    block.conv1.padding = (prev_dilation, prev_dilation)
                    block.conv2.stride = 1
                    block.conv2.dilation = (dilation, dilation)
                    block.conv2.padding = (dilation, dilation)
                    prev_dilation = dilation

    return model


class _ResNetV2(ResNetV2):
    """ResNet v2 with dilation support and classification layer removed."""

    def __init__(self, block_units, width_factor, replace_stride_with_dilation=None):
        super().__init__(block_units, width_factor, head_size=1)

        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        else:
            dilation = 1
            for dilate, block in zip(replace_stride_with_dilation,
                                     [self.body.block2, self.body.block3, self.body.block4]):
                if dilate:
                    block.unit01.conv2.stride = 1
                    block.unit01.conv2.dilation = (dilation, dilation)
                    block.unit01.conv2.padding = (dilation, dilation)
                    block.unit01.downsample.stride = 1
                    dilation *= 2
                    for unit in block[1:]:
                        unit.conv2.stride = 1
                        unit.conv2.dilation = (dilation, dilation)
                        unit.conv2.padding = (dilation, dilation)

        self.head = nn.Sequential(OrderedDict([
            ('gn', nn.GroupNorm(32, 2048 * width_factor)),
            ('relu', nn.ReLU(inplace=True)),
        ]))

    def forward(self, x):
        out = OrderedDict()
        x = self.root(x)
        for i, block in enumerate(self.body):
            x = block(x)
            if i == 0:
                out['low_level'] = x
        out['out'] = self.head(x)

        return out

    def load_from(self, weights, prefix='resnet/'):
        with torch.no_grad():
            self.root.conv.weight.copy_(
                tf2th(weights[f'{prefix}root_block/standardized_conv2d/kernel']))

            for bname, block in self.body.named_children():
                for uname, unit in block.named_children():
                    unit.load_from(weights, prefix=f'{prefix}{bname}/{uname}/')


def get_resnet_encoder(backbone, imagenet_init, use_dilation):
    """
    Args:
        backbone (str): Name of the encoder backbone.
        imagenet_init (bool): If available, choose whether to initialize the encoder with weights
                              pre-trained on ImageNet.
        use_dilation (bool): Whether to replace strides in the last two blocks with dilation.
    """
    if use_dilation:
        replace_stride_with_dilation = [False, True, True]
    else:
        replace_stride_with_dilation = None

    if backbone in RESNET_CONSTRUCTORS:
        resnet = RESNET_CONSTRUCTORS[backbone]
        backbone = resnet(pretrained=imagenet_init, replace_stride_with_dilation=replace_stride_with_dilation)
        return_layers = {'layer1': 'low_level', 'layer4': 'out'}
        encoder = IntermediateLayerGetter(backbone, return_layers=return_layers)
    elif backbone in RESNETV2_CONSTRUCTORS:
        resnetv2 = RESNETV2_CONSTRUCTORS[backbone]
        encoder = resnetv2(replace_stride_with_dilation=replace_stride_with_dilation)
    else:
        raise ValueError(f"Unsupported backbone '{backbone}'.")

    return encoder
