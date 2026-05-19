import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import math

import numpy as np
import torch
import torch.nn.functional as F
from scipy import signal

dtype = torch.cuda.FloatTensor


class EFF:
    """
    EFF Class implements the Efficient Filter Flow approximation for non-uniform blurring
    The main idea is to devide the image by overlapping patches
    """

    def __init__(self, img_size, blur_size=[31, 31], grid_size=[6, 8]):
        self.img_size = img_size
        self.blur_size = blur_size
        self.grid_size = grid_size
        self.patch_size = None
        self.patch_inner_size = None
        self.padded_size = None
        self.W = None
        self.Wsum = None
        self.pad = (
            None  # note that the top, bottom, left and right/ left,right,top,bottom
        )
        self.t_patch = None
        self.b_patch = None
        self.l_patch = None
        self.r_patch = None
        self.x_patch_centers = None
        self.y_patch_centers = None

    def makeEFF(self):
        """
        Compute the window function and related parameters.
        """
        h_psf, w_psf = self.blur_size
        h_sharp, w_sharp = self.img_size

        if (h_sharp - w_sharp) * (self.grid_size[0] - self.grid_size[1]) < 0:
            self.grid_size = self.grid_size[::-1]

        h_grid, w_grid = self.grid_size

        if h_grid > 1:
            h_patch_inner = math.ceil(h_sharp / (h_grid - 1))
            h_patch_inner = int(h_patch_inner - h_patch_inner % 2)
        else:
            h_patch_inner = h_sharp

        if w_grid > 1:
            w_patch_inner = math.ceil(w_sharp / (w_grid - 1))
            w_patch_inner = int(w_patch_inner - w_patch_inner % 2)
        else:
            w_patch_inner = w_sharp

        self.patch_inner_size = [h_patch_inner, w_patch_inner]
        w_patch = w_patch_inner + w_psf
        h_patch = h_patch_inner + h_psf
        self.patch_size = [h_patch, w_patch]

        y_patch_centers = np.arange(h_grid) * h_patch_inner + h_patch
        x_patch_centers = np.arange(w_grid) * w_patch_inner + w_patch
        self.y_patch_centers = y_patch_centers
        self.x_patch_centers = x_patch_centers

        h_padded = y_patch_centers[-1] + h_patch
        w_padded = x_patch_centers[-1] + w_patch
        self.padded_size = [h_padded, w_padded]

        pad_zero_t = int(math.floor((h_padded - h_sharp) / 2))
        pad_zero_b = int(math.ceil((h_padded - h_sharp) / 2))
        pad_zero_l = int(math.floor((w_padded - w_sharp) / 2))
        pad_zero_r = int(math.ceil((w_padded - w_sharp) / 2))
        self.pad = [pad_zero_t, pad_zero_b, pad_zero_l, pad_zero_r]

        l_patch = [x_patch_centers[gx] - w_patch for gx in range(w_grid)]
        r_patch = [x_patch_centers[gx] + w_patch for gx in range(w_grid)]
        t_patch = [y_patch_centers[gy] - h_patch for gy in range(h_grid)]
        b_patch = [y_patch_centers[gy] + h_patch for gy in range(h_grid)]
        self.l_patch, self.r_patch, self.t_patch, self.b_patch = (
            l_patch,
            r_patch,
            t_patch,
            b_patch,
        )

        b_r = signal.windows.barthann(2 * h_patch_inner + 1).reshape(-1, 1)
        b_c = signal.windows.barthann(2 * w_patch_inner + 1).reshape(1, -1)
        W = b_r * b_c
        W = W[:-1, :-1]
        W = torch.from_numpy(W)
        W = self.padimg(
            W,
            [
                h_patch - h_patch_inner,
                h_patch - h_patch_inner,
                w_patch - w_patch_inner,
                w_patch - w_patch_inner,
            ],
        )

        Wsum = torch.zeros((h_padded, w_padded))
        for gx in range(w_grid):
            for gy in range(h_grid):
                Wsum[t_patch[gy] : b_patch[gy], l_patch[gx] : r_patch[gx]] += W
        self.W = W[None, None, ...]
        self.Wsum = Wsum[None, None, ...]

    def extractImageStack(self, img, which_patch, mode="sharp"):
        """
        从填充后的图像中提取patch。

        参数：
            img: 输入图像
            which_patch: 指定要提取的patch索引，可以是整数、列表或range对象
            mode: 提取模式，默认为"sharp"
        """
        assert img.shape[2:] == tuple(self.padded_size), "图像尺寸不匹配"

        # 处理which_patch的类型
        if isinstance(which_patch, range):
            which_patch = list(which_patch)  # 将range转换为列表
        elif not isinstance(which_patch, list):
            which_patch = [which_patch]  # 将单个整数包装成列表

        n_grid = len(which_patch)
        channels = img.shape[1]
        h_patch, w_patch = self.patch_size
        img_stack = torch.zeros(
            (n_grid, channels, 2 * h_patch, 2 * w_patch), device=img.device
        )

        for i, patch_idx in enumerate(which_patch):
            gy, gx = self.ind2sub(self.grid_size, patch_idx)  # patch_idx现在是整数
            t, b, l, r = (
                self.t_patch[gy],
                self.b_patch[gy],
                self.l_patch[gx],
                self.r_patch[gx],
            )
            img_stack[i] = img[:, :, t:b, l:r]
            if mode != "blurry":
                img_stack[i] *= self.W.squeeze(0)

        return img_stack

    def combineStackImage(self, img, img_stack, which_patch):
        """
        Combine the stack image into the full image.
        """
        assert img.shape[2:] == tuple(self.padded_size), "Size does not match"
        # 处理which_patch的类型
        if isinstance(which_patch, range):
            which_patch = list(which_patch)  # 将range转换为列表
        elif not isinstance(which_patch, list):
            which_patch = [which_patch]  # 将单个整数包装成列表
        n_grid = len(which_patch) if isinstance(which_patch, list) else 1
        which_patch = (
            [which_patch] if not isinstance(which_patch, list) else which_patch
        )
        for i, patch_idx in enumerate(which_patch):
            gy, gx = self.ind2sub(self.grid_size, patch_idx)
            t, b, l, r = (
                self.t_patch[gy],
                self.b_patch[gy],
                self.l_patch[gx],
                self.r_patch[gx],
            )
            img[0, :, t:b, l:r] += img_stack[i]

    def cuda(self):
        self.W = self.W.cuda().type(dtype)
        self.Wsum = self.Wsum.cuda().type(dtype)

    def cpu(self):
        self.W = self.W.cpu()
        self.Wsum = self.Wsum.cpu()

    def forward(self, img, kernel):
        # 假设 grid_size 和 patch_size 已定义
        n_grid = self.grid_size[0] * self.grid_size[1]
        channels = img.shape[1]  # 例如 3
        h_patch, w_patch = self.patch_size

        # 提取 patch 堆栈
        img_stack = self.extractImageStack(img, list(range(n_grid)), "sharp")
        # img_stack 形状: (n_grid, channels, 2*h_patch, 2*w_patch)

        # 确保 kernel 形状正确
        if kernel.shape[1] == 1:
            kernel = kernel.repeat(1, n_grid, 1, 1)
        # kernel 形状: (1, n_grid, kernel_size_x, kernel_size_y)

        kernel_size_x, kernel_size_y = kernel.shape[2], kernel.shape[3]
        start_x = (kernel_size_x - 1) // 2
        start_y = (kernel_size_y - 1) // 2

        # 初始化输出张量
        IMG_S = torch.zeros(
            (n_grid, channels, 2 * h_patch, 2 * w_patch), device=img.device
        )

        # 对每个 patch 执行分组卷积
        for i in range(n_grid):
            # 输入形状: (1, channels, 2*h_patch, 2*w_patch)
            input_patch = img_stack[i : i + 1]
            # 卷积核形状: (1, 1, kernel_size_x, kernel_size_y) -> (channels, 1, kernel_size_x, kernel_size_y)
            kernel_patch = kernel[:, i : i + 1].repeat(channels, 1, 1, 1)

            # 分组卷积
            conv_result = F.conv2d(
                input_patch,  # (1, channels, 2*h_patch, 2*w_patch)
                kernel_patch,  # (channels, 1, kernel_size_x, kernel_size_y)
                groups=channels,  # 每个通道独立卷积
                padding=0,
            )  # 输出形状: (1, channels, 2*h_patch - kernel_size_x + 1, 2*w_patch - kernel_size_y + 1)

            h_out, w_out = conv_result.shape[2], conv_result.shape[3]
            IMG_S[i, :, start_x : start_x + h_out, start_y : start_y + w_out] = (
                conv_result[0]
            )

        # 组合结果
        img_placeholder = torch.zeros_like(img)
        self.combineStackImage(img_placeholder, IMG_S, range(n_grid))
        img_blurred = self.padimg(img_placeholder, self.pad, "depad")
        return img_blurred

    def extractImageStack_y(self, img, kernel_size, which_patch, mode="blurry"):
        """
        extract the padded img with the patch index
        """
        if not isinstance(which_patch, list):
            which_patch = [which_patch]
        n_grid = len(which_patch)
        channels = img.shape[1]  # 3 for color image
        kernel_size_x, kernel_size_y = kernel_size[0], kernel_size[1]
        start_x = int((kernel_size_x - 1) / 2.0)
        start_y = int((kernel_size_y - 1) / 2.0)

        img_stack = torch.zeros(
            (
                n_grid,
                channels,
                2 * self.patch_size[0] - 2 * start_x,
                2 * self.patch_size[1] - 2 * start_y,
            ),
            device=img.device,
        )

        assert img.shape[2:] == tuple(self.padded_size), "the size does not match"
        t_patch = self.t_patch
        b_patch = self.b_patch
        l_patch = self.l_patch
        r_patch = self.r_patch
        h_patch = self.patch_size[0]
        w_patch = self.patch_size[1]

        try:
            n_grid = len(which_patch)
        except:
            n_grid = 1
            which_patch = [which_patch]
        assert (
            np.max(np.array(which_patch)) < self.grid_size[0] * self.grid_size[1]
        ), "out of the number of patches"
        channels = img.shape[1]  # if img.ndim == 4 else 1

        weights = self.W

        if channels > 1:
            weights = weights.repeat(1, channels, 1, 1)
        for i in range(n_grid):
            gy, gx = self.ind2sub(self.grid_size, which_patch[i])
            img_stack_s = img[
                ...,
                t_patch[gy] + start_x : b_patch[gy] - start_x,
                l_patch[gx] + start_y : r_patch[gx] - start_y,
            ]
            if mode != "blurry":
                img_stack[i, ...] = (img_stack_s * weights)[0, ...]
            else:
                img_stack[i, ...] = img_stack_s[0, ...]
        return img_stack

    def padimg(self, img, padsize, mode="pad"):
        """
        Pad or depad the image with required size.
        """
        assert len(padsize) == 4, "Four sizes needed to pad"
        t_pad, b_pad, l_pad, r_pad = padsize
        if mode == "pad":
            img_padded = F.pad(img, (l_pad, r_pad, t_pad, b_pad), "constant")
        else:
            img_padded = img[..., t_pad : -b_pad or None, l_pad : -r_pad or None]
        return img_padded

    def ind2sub(self, array_shape, ind):
        rows = int(ind) // array_shape[1]
        cols = ind % array_shape[1]
        return rows, cols
