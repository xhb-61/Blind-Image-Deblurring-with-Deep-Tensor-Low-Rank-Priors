import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import argparse
import math
import random

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from PIL import Image


def ensure_reproducibility(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def crop_image(img, d=32):
    """Make dimensions divisible by `d`"""
    imgsize = img.shape

    new_size = (imgsize[0] - imgsize[0] % d, imgsize[1] - imgsize[1] % d)

    bbox = [
        int((imgsize[0] - new_size[0]) / 2),
        int((imgsize[1] - new_size[1]) / 2),
        int((imgsize[0] + new_size[0]) / 2),
        int((imgsize[1] + new_size[1]) / 2),
    ]

    img_cropped = img[0: new_size[0], 0: new_size[1], :]
    return img_cropped


def get_params(opt_over, net, net_input, downsampler=None):
    """Returns parameters that we want to optimize over.

    Args:
        opt_over: comma separated list, e.g. "net,input" or "net"
        net: network
        net_input: torch.Tensor that stores input `z`
    """
    opt_over_list = opt_over.split(",")
    params = []

    for opt in opt_over_list:

        if opt == "net":
            params += [x for x in net.parameters()]
        elif opt == "down":
            assert downsampler is not None
            params += [x for x in downsampler.parameters()]
        elif opt == "input":
            net_input.requires_grad = True
            params += [net_input]
        else:
            assert False, "what is it?"

    return params


def get_image_grid(images_np, nrow=8):
    """Creates a grid from a list of images by concatenating them."""
    images_torch = [torch.from_numpy(x) for x in images_np]
    torch_grid = torchvision.utils.make_grid(images_torch, nrow)

    return torch_grid.numpy()


def plot_image_grid(images_np, nrow=8, factor=1, interpolation="lanczos"):
    """Draws images in a grid

    Args:
        images_np: list of images, each image is np.array of size 3xHxW of 1xHxW
        nrow: how many images will be in one row
        factor: size if the plt.figure
        interpolation: interpolation used in plt.imshow
    """
    n_channels = max(x.shape[0] for x in images_np)
    assert (n_channels == 3) or (n_channels == 1), "images should have 1 or 3 channels"

    images_np = [
        x if (x.shape[0] == n_channels) else np.concatenate([x, x, x], axis=0)
        for x in images_np
    ]

    grid = get_image_grid(images_np, nrow)

    plt.figure(figsize=(len(images_np) + factor, 12 + factor))

    if images_np[0].shape[0] == 1:
        plt.imshow(grid[0], cmap="gray", interpolation=interpolation)
    else:
        plt.imshow(grid.transpose(1, 2, 0), interpolation=interpolation)

    plt.show()

    return grid


def load(path):
    """Load PIL image."""
    img = Image.open(path)

    return img


def get_image(path, imsize=-1):
    """Load an image and resize to a cpecific size.

    Args:
        path: path to image
        imsize: tuple or scalar with dimensions; -1 for `no resize`
    """
    img = load(path)

    if isinstance(imsize, int):
        imsize = (imsize, imsize)

    if imsize[0] != -1 and img.size != imsize:
        if imsize[0] > img.size[0]:
            img = img.resize(imsize, Image.BICUBIC)
        else:
            img = img.resize(imsize, Image.ANTIALIAS)

    img_np = pil_to_np(img)

    return img, img_np


def get_color_image(path, imsize=-1):
    """Load an image and resize to a cpecific size.

    Args:
        path: path to image
        imsize: tuple or scalar with dimensions; -1 for `no resize`
    """
    img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    y = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2YCrCb)
    y, _, _ = cv2.split(y)

    # img = img[:512, :512, :]

    img = img
    img = img.transpose(2, 0, 1)

    return img.astype(np.float32) / 255.0, y.astype(np.float32) / 255.0


def fill_noise(x, noise_type):
    """Fills tensor `x` with noise of type `noise_type`."""
    # torch.manual_seed(0)
    if noise_type == "u":
        x.uniform_()
    elif noise_type == "n":
        x.normal_()
    else:
        assert False


def get_noise(input_depth, method, spatial_size, noise_type="u", var=1.0 / 10):
    """Returns a pytorch.Tensor of size (1 x `input_depth` x `spatial_size[0]` x `spatial_size[1]`)
    initialized in a specific way.
    Args:
        input_depth: number of channels in the tensor
        method: `noise` for fillting tensor with noise; `meshgrid` for np.meshgrid
        spatial_size: spatial size of the tensor to initialize
        noise_type: 'u' for uniform; 'n' for normal
        var: a factor, a noise will be multiplicated by. Basically it is standard deviation scaler.
    """
    if isinstance(spatial_size, int):
        spatial_size = (spatial_size, spatial_size)
    if method == "noise":
        shape = [1, input_depth, spatial_size[0], spatial_size[1]]
        net_input = torch.zeros(shape)

        fill_noise(net_input, noise_type)
        net_input *= var
    elif method == "meshgrid":
        assert input_depth == 2
        X, Y = np.meshgrid(
            np.arange(0, spatial_size[1]) / float(spatial_size[1] - 1),
            np.arange(0, spatial_size[0]) / float(spatial_size[0] - 1),
        )
        meshgrid = np.concatenate([X[None, :], Y[None, :]])
        net_input = np_to_torch(meshgrid)
    else:
        assert False

    return net_input


def pil_to_np(img_PIL):
    """Converts image in PIL format to np.array.

    From W x H x C [0...255] to C x W x H [0..1]
    """
    ar = np.array(img_PIL)

    if len(ar.shape) == 3:
        ar = ar.transpose(2, 0, 1)
    else:
        ar = ar[None, ...]

    return ar.astype(np.float32) / 255.0


def np_to_pil(img_np):
    """Converts image in np.array format to PIL image.

    From C x W x H [0..1] to  W x H x C [0...255]
    """
    ar = np.clip(img_np * 255, 0, 255).astype(np.uint8)

    if len(img_np.shape) == 3:
        if img_np.shape[2] == 1:
            ar = ar[2]
        else:
            ar = ar  # .transpose(1, 2, 0)

    return Image.fromarray(ar)


def np_to_torch(img_np):
    """Converts image in numpy.array to torch.Tensor.

    From C x W x H [0..1] to  C x W x H [0..1]
    """
    return torch.from_numpy(img_np)[None, :]


def torch_to_np(img_var):
    """Converts an image in torch.Tensor format to np.array.

    From 1 x C x W x H [0..1] to  C x W x H [0..1]
    """
    return img_var.detach().cpu().numpy()[0]


def optimize(optimizer_type, parameters, closure, LR, num_iter):
    """Runs optimization loop.

    Args:
        optimizer_type: 'LBFGS' of 'adam'
        parameters: list of Tensors to optimize over
        closure: function, that returns loss variable
        LR: learning rate
        num_iter: number of iterations
    """
    if optimizer_type == "LBFGS":
        # Do several steps with adam first
        optimizer = torch.optim.Adam(parameters, lr=0.001)
        for j in range(100):
            optimizer.zero_grad()
            closure()
            optimizer.step()

        print("Starting optimization with LBFGS")

        def closure2():
            optimizer.zero_grad()
            return closure()

        optimizer = torch.optim.LBFGS(
            parameters, max_iter=num_iter, lr=LR, tolerance_grad=-1, tolerance_change=-1
        )
        optimizer.step(closure2)

    elif optimizer_type == "adam":
        print("Starting optimization with ADAM")
        optimizer = torch.optim.Adam(parameters, lr=LR)
        from torch.optim.lr_scheduler import MultiStepLR

        scheduler = MultiStepLR(
            optimizer, milestones=[5000, 10000, 15000], gamma=0.1
        )  # learning rates
        for j in range(num_iter):
            scheduler.step(j)
            optimizer.zero_grad()
            closure()
            optimizer.step()
    else:
        assert False


def pixelshuffle(image, scale):
    """
    Discription: Given an image, return a reversible sub-sampling
    [Input]: Image ndarray float
    [Return]: A mosic image of shuffled pixels
    """
    if scale == 1:
        return image
    w, h, c = image.shape
    mosaic = np.array([])
    for ws in range(scale):
        band = np.array([])
        for hs in range(scale):
            temp = image[ws::scale, hs::scale, :]  # get the sub-sampled image
            band = np.concatenate((band, temp), axis=1) if band.size else temp
        mosaic = np.concatenate((mosaic, band), axis=0) if mosaic.size else band
    return mosaic


def reverse_pixelshuffle(image, scale, fill=0, fill_image=0, ind=[0, 0]):
    """
    Discription: Given a mosaic image of subsampling, recombine it to a full image
    [Input]: Image
    [Return]: Recombine it using different portions of pixels
    """
    w, h, c = image.shape
    real = np.zeros((w, h, c))  # real image
    wf = 0
    hf = 0
    for ws in range(scale):
        hf = 0
        for hs in range(scale):
            temp = real[ws::scale, hs::scale, :]
            wc, hc, cc = temp.shape  # get the shpae of the current images
            if fill == 1 and ws == ind[0] and hs == ind[1]:
                real[ws::scale, hs::scale, :] = fill_image[
                                                wf: wf + wc, hf: hf + hc, :
                                                ]
            else:
                real[ws::scale, hs::scale, :] = image[wf: wf + wc, hf: hf + hc, :]
            hf = hf + hc
        wf = wf + wc
    return real


def readimg(path_to_image):
    img = cv2.imread(path_to_image)
    x = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(x)

    return img, y, cb, cr


def blurring(cleanimg, kernel, kernelsize):
    num_pad = (kernelsize - 1) // 2

    clean_img_pad = F.pad(cleanimg, (num_pad,) * 4, mode="reflect")
    out = F.conv3d(
        clean_img_pad.unsqueeze(0), kernel.unsqueeze(1), groups=cleanimg.shape[0]
    )  # 1 x B x C x H x W
    return out.squeeze(0)


def save_img_np(save_dir, img_np):
    img_pil = np_to_pil(img_np)
    img_pil.save(save_dir)


class ImageSpliterTh:
    def __init__(self, im, pch_size, stride, sf=1, extra_bs=1, weight_type="Gaussian"):
        """
        Input:
            im: n x c x h x w, torch tensor, float, low-resolution image in SR
            pch_size, stride: patch setting
            sf: scale factor in image super-resolution 超分辨率中的缩放因子（scale factor），默认值为1
            pch_bs: aggregate pchs to processing, only used when inputing single image
            extra_bs: 用于处理单个图像时，将多个小块聚合在一起处理的批次大小，默认值为1
        """
        assert weight_type in ["Gaussian", "ones"]
        self.weight_type = weight_type
        assert stride <= pch_size
        self.stride = stride
        self.pch_size = pch_size
        self.sf = sf
        self.extra_bs = extra_bs

        bs, chn, height, width = im.shape
        self.true_bs = bs

        self.height_starts_list = self.extract_starts(height)
        self.width_starts_list = self.extract_starts(width)
        self.starts_list = []
        for ii in self.height_starts_list:
            for jj in self.width_starts_list:
                self.starts_list.append([ii, jj])

        self.length = self.__len__()
        self.count_pchs = 0

        self.im_ori = im
        self.dtype = torch.float64
        self.im_res = torch.zeros(
            [bs, chn, height * sf, width * sf], dtype=self.dtype, device=im.device
        )
        self.pixel_count = torch.zeros(
            [bs, chn, height * sf, width * sf], dtype=self.dtype, device=im.device
        )

    def extract_starts(self, length):
        if length <= self.pch_size:
            starts = [
                0,
            ]
        else:
            starts = list(range(0, length, self.stride))
            for ii in range(len(starts)):
                if starts[ii] + self.pch_size > length:
                    starts[ii] = length - self.pch_size
            starts = sorted(set(starts), key=starts.index)
        return starts

    def __len__(self):
        return len(self.height_starts_list) * len(self.width_starts_list)

    def __iter__(self):
        return self

    def __next__(self):
        if self.count_pchs < self.length:
            index_infos = []
            current_starts_list = self.starts_list[
                                  self.count_pchs: self.count_pchs + self.extra_bs
                                  ]
            for ii, (h_start, w_start) in enumerate(current_starts_list):
                w_end = w_start + self.pch_size
                h_end = h_start + self.pch_size
                current_pch = self.im_ori[:, :, h_start:h_end, w_start:w_end]
                if ii == 0:
                    pch = current_pch
                else:
                    pch = torch.cat([pch, current_pch], dim=0)

                h_start *= self.sf
                h_end *= self.sf
                w_start *= self.sf
                w_end *= self.sf
                index_infos.append([h_start, h_end, w_start, w_end])

            self.count_pchs += len(current_starts_list)
        else:
            raise StopIteration()

        return pch, index_infos

    def update(self, pch_res, index_infos):
        """
        Input:
            pch_res: (n*extra_bs) x c x pch_size x pch_size, float
            index_infos: [(h_start, h_end, w_start, w_end),]
        """
        assert pch_res.shape[0] % self.true_bs == 0
        pch_list = torch.split(pch_res, self.true_bs, dim=0)
        assert len(pch_list) == len(index_infos)
        for ii, (h_start, h_end, w_start, w_end) in enumerate(index_infos):
            current_pch = pch_list[ii].type(self.dtype)
            current_weight = self.get_weight(
                current_pch.shape[-2], current_pch.shape[-1]
            )
            self.im_res[:, :, h_start:h_end, w_start:w_end] += (
                    current_pch * current_weight
            )
            self.pixel_count[:, :, h_start:h_end, w_start:w_end] += current_weight

    @staticmethod
    def generate_kernel_1d(ksize):
        sigma = 0.3 * ((ksize - 1) * 0.5 - 1) + 0.8  # opencv default setting
        if ksize % 2 == 0:
            kernel = cv2.getGaussianKernel(
                ksize=ksize + 1, sigma=sigma, ktype=cv2.CV_64F
            )
            kernel = kernel[1:, ]
        else:
            kernel = cv2.getGaussianKernel(ksize=ksize, sigma=sigma, ktype=cv2.CV_64F)

        return kernel

    def get_weight(self, height, width):
        if self.weight_type == "ones":
            kernel = torch.ones(1, 1, height, width)
        elif self.weight_type == "Gaussian":
            kernel_h = self.generate_kernel_1d(height).reshape(-1, 1)
            kernel_w = self.generate_kernel_1d(width).reshape(1, -1)
            kernel = np.matmul(kernel_h, kernel_w)
            kernel = (
                torch.from_numpy(kernel).unsqueeze(0).unsqueeze(0)
            )  # 1 x 1 x pch_size x pch_size
        else:
            raise ValueError(f"Unsupported weight type: {self.weight_type}")

        return kernel.to(dtype=self.dtype, device=self.im_ori.device)

    def gather(self):
        assert torch.all(self.pixel_count != 0)
        return self.im_res.div(self.pixel_count)


def correct_boundary(fe, grid_size):
    # fe: [50, 256, 3, 3]
    cand = []
    for i in range(1, grid_size[1]):
        cand.append((i, i + 10))
    for i in range((grid_size[0] - 1) * grid_size[1] + 1, grid_size[0] * grid_size[1]):
        cand.append((i, i - 10))
    for i in range(grid_size[0]):
        cand.append((i * (grid_size[1]), i * (grid_size[1]) + 1))
    for i in range(grid_size[0]):
        cand.append(((i + 1) * (grid_size[1]) - 1, (i + 1) * (grid_size[1]) - 2))

    with torch.no_grad():
        fe_data = fe.data
        for i, j in cand:
            fe_data[i:i + 1].copy_(fe_data[j:j + 1])


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm') != -1:
        # nn.init.uniform(m.weight.data, 1.0, 0.02)
        m.weight.data.normal_(mean=0, std=math.sqrt(2. / 9. / 64.)).clamp_(-0.025, 0.025)
        nn.init.constant_(m.bias.data, 0.0)
