# Adapted from https://github.com/pytorch/vision/blob/master/torchvision/transforms/functional_tensor.py

import torch
import torchvision.transforms.functional as F
from torch.nn.functional import conv2d, pad as torch_pad


def _cast_squeeze_in(img, req_dtypes):
    need_squeeze = False
    # make image NCHW
    if img.ndim < 4:
        img = img.unsqueeze(dim=0)
        need_squeeze = True

    out_dtype = img.dtype
    need_cast = False
    if out_dtype not in req_dtypes:
        need_cast = True
        req_dtype = req_dtypes[0]
        img = img.to(req_dtype)
    return img, need_cast, need_squeeze, out_dtype


def _cast_squeeze_out(img, need_cast, need_squeeze, out_dtype):
    if need_squeeze:
        img = img.squeeze(dim=0)

    if need_cast:
        if out_dtype in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
            # it is better to round before cast
            img = torch.round(img)
        img = img.to(out_dtype)

    return img


def _get_gaussian_kernel1d(kernel_size, sigma):
    ksize_half = (kernel_size - 1) * 0.5

    x = torch.linspace(-ksize_half, ksize_half, steps=kernel_size)
    pdf = torch.exp(-0.5 * (x / sigma).pow(2))
    kernel1d = pdf / pdf.sum()

    return kernel1d


def _get_gaussian_kernel2d(kernel_size, sigma, dtype, device):
    kernel1d_x = _get_gaussian_kernel1d(kernel_size[0], sigma[0]).to(device, dtype=dtype)
    kernel1d_y = _get_gaussian_kernel1d(kernel_size[1], sigma[1]).to(device, dtype=dtype)
    kernel2d = torch.mm(kernel1d_y[:, None], kernel1d_x[None, :])

    return kernel2d


def gaussian_blur(img, kernel_size, sigma):
    if not (isinstance(img, torch.Tensor)):
        raise TypeError('img should be Tensor. Got {}'.format(type(img)))

    dtype = img.dtype if torch.is_floating_point(img) else torch.float32
    kernel = _get_gaussian_kernel2d(kernel_size, sigma, dtype=dtype, device=img.device)
    kernel = kernel.expand(img.shape[-3], 1, kernel.shape[0], kernel.shape[1])

    img, need_cast, need_squeeze, out_dtype = _cast_squeeze_in(img, [kernel.dtype, ])

    # padding = (left, right, top, bottom)
    padding = [kernel_size[0] // 2, kernel_size[0] // 2, kernel_size[1] // 2, kernel_size[1] // 2]
    img = torch_pad(img, padding, mode="reflect")
    img = conv2d(img, kernel, groups=img.shape[-3])

    img = _cast_squeeze_out(img, need_cast, need_squeeze, out_dtype)

    return img


class GaussianBlur(torch.nn.Module):
    def __init__(self, kernel_size, sigma=(0.1, 2.0)):
        super().__init__()

        self.kernel_size = (int(kernel_size), int(kernel_size))
        self.sigma = sigma

    @staticmethod
    def get_params(sigma_min, sigma_max):
        return torch.empty(1).uniform_(sigma_min, sigma_max).item()

    def forward(self, img):
        sigma = self.get_params(self.sigma[0], self.sigma[1])

        t_img = img
        if not isinstance(img, torch.Tensor):
            t_img = F.to_tensor(img)

        output = gaussian_blur(t_img, self.kernel_size, [sigma, sigma])

        if not isinstance(img, torch.Tensor):
            output = F.to_pil_image(output)

        return output


def invert(img):
    bound = torch.tensor(1 if img.is_floating_point() else 255, dtype=img.dtype, device=img.device)

    return bound - img


def solarize(img, threshold):
    if not (isinstance(img, torch.Tensor)):
        raise TypeError('img should be Tensor. Got {}'.format(type(img)))

    inverted_img = invert(img)

    return torch.where(img >= threshold, inverted_img, img)


class Solarize(torch.nn.Module):
    def __init__(self, threshold):
        super().__init__()

        self.threshold = threshold

    def forward(self, img):
        t_img = img
        if not isinstance(img, torch.Tensor):
            t_img = F.to_tensor(img)

        output = solarize(t_img, self.threshold)

        if not isinstance(img, torch.Tensor):
            output = F.to_pil_image(output)

        return output
