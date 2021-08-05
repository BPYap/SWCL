import math

import torch
import torch.distributed as dist
import torch.nn as nn


class GatherLayer(torch.autograd.Function):
    """Gather tensors from all process, supporting backward propagation.

    adapted from https://github.com/open-mmlab/OpenSelfSup/blob/master/openselfsup/models/utils/gather_layer.py
    """

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        output = [torch.zeros_like(input) for _ in range(dist.get_world_size())]
        dist.all_gather(output, input)

        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        input, = ctx.saved_tensors
        grad_out = torch.zeros_like(input)
        grad_out[:] = grads[dist.get_rank()]

        return grad_out


def get_cams(feature_maps, classification_layer, targets, upsample_size):
    num_feat_maps = feature_maps.shape[1]
    # extract weights associated with the target class
    weights = classification_layer.weight[targets, :].reshape((-1, num_feat_maps, 1, 1))
    # each cam is the weighted sum of the feature maps
    cams = (weights * feature_maps).sum(1).unsqueeze(1)
    # upsample each cam to match the target size
    up_sample = nn.Upsample(size=upsample_size, mode='bilinear', align_corners=True)
    heat_maps = up_sample(cams)

    return heat_maps


def forward_patches(model, inputs, patch_size=384, dict_keys=None):
    height = inputs.shape[-2]
    width = inputs.shape[-1]
    vertical_stride = math.floor(height / math.ceil(height / patch_size))
    horizontal_stride = math.floor(width / math.ceil(width / patch_size))

    if dict_keys:
        logits = {key: torch.zeros(inputs.shape[0], 2, height, width).to(inputs.device) for key in dict_keys}
    else:
        logits = torch.zeros(inputs.shape[0], 2, height, width).to(inputs.device)

    for row in list(range(0, height - patch_size, vertical_stride)) + [height - patch_size]:
        for col in list(range(0, width - patch_size, horizontal_stride)) + [width - patch_size]:
            patches = inputs[:, :, row:row + patch_size, col:col + patch_size].to(inputs.device)
            output = model(patches)
            if dict_keys:
                for key in dict_keys:
                    logits[key][:, :, row:row + patch_size, col:col + patch_size] += output[key]
            else:
                logits[:, :, row:row + patch_size, col:col + patch_size] += output
            del patches

    return logits
