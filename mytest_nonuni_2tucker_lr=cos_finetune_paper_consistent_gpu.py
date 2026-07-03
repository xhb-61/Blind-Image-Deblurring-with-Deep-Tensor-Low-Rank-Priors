from __future__ import print_function
import os
import argparse

_gpu_parser = argparse.ArgumentParser(add_help=False)
_gpu_parser.add_argument(
    "--gpu",
    type=str,
    default=os.environ.get("CUDA_VISIBLE_DEVICES", "1"),
    help="physical GPU id to expose to this process",
)
_gpu_args, _ = _gpu_parser.parse_known_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(_gpu_args.gpu)

import glob
import os
import random
import warnings

import numpy as np
import torch
import torchvision.transforms.functional as TVTF
from torch import nn
from tqdm import tqdm
import tensorly as tl
from tensorly.decomposition import tucker
from SSIM import SSIM
from eff import EFF
from networks.knet import Generator, ResNet18
from networks.skip import skip
from utils.common_utils import (
    correct_boundary,
    get_color_image,
    get_noise,
    np_to_torch,
    save_img_np,
    torch_to_np,
)

os.environ["CUDA_VISIBLE_DEVICES"] = str(_gpu_args.gpu)

# torch.backends.cudnn.benchmark = True
# torch.backends.cudnn.enabled = False
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
parser = argparse.ArgumentParser()

parser.add_argument(
    "--num_iter", type=int, default=5000, help="number of epochs of training"
)
parser.add_argument(
    "--img_size", type=int, default=[256, 256], help="size of each image dimension"
)
parser.add_argument(
    "--kernel_size", type=int, default=[55, 55], help="size of blur kernel"
)
parser.add_argument(
    "--grid_size", type=int, default=[5, 10], help="size of grid for kernel"
)
parser.add_argument(
    "--data_path",
    type=str,
    default="./datasets/lai/nonuniform",
    # default="./datasets/real_dataset",
    help="path to blurry image",
)
parser.add_argument(
    "--save_path",
    type=str,
    default="./results_2tucker_newname",
    help="path to save results",
)
parser.add_argument(
    "--save_frequency", type=int, default=100, help="lfrequency to save results"
)
parser.add_argument("--seed", type=int, default=0, help="random seed")
parser.add_argument(
    "--gpu",
    type=str,
    default=str(_gpu_args.gpu),
    help="physical GPU id to expose to this process",
)
opt = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.gpu)
print(opt)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)

def getPSNR(I1, I2):
    s1 = I1-I2 #|I1 - I2|
    s1 = np.float32(s1)     # cannot make a square on 8 bits
    s1 = s1 ** 2            # |I1 - I2|^2
    sse = s1.sum()          # sum elements per channel
    if sse <= 0:        # sum channels
        return 0            # for small values return zero
    else:
        shape = I1.shape
        # print(shape)
        mse = 1.0 * sse / (shape[0] * shape[1] * shape[2])
        psnr = 10.0 * np.log10((1.0) / mse)
        return psnr

def mcp_threshold(x, lam, gamma):
    abs_x = torch.abs(x)
    scaled = (gamma / (gamma - 1.0)) * (abs_x - lam)
    shrink = torch.minimum(abs_x, torch.clamp(scaled, min=0.0))
    return torch.sign(x) * shrink


def dtccp_tucker_prox(x, dual, ranks, eta, alpha, mcp_lambda, mcp_gamma):
    core, factors = tucker(x - dual, rank=ranks, init="svd")
    prox_lambda = eta * mcp_lambda / max(2.0 * alpha, 1e-12)
    core_threshold = mcp_threshold(core, prox_lambda, mcp_gamma)
    return tl.tucker_to_tensor((core_threshold, factors))



dtype = torch.cuda.FloatTensor
warnings.filterwarnings("ignore")
files_source = glob.glob(os.path.join(opt.data_path, "*.png"))
files_source.sort()
save_path = opt.save_path
os.makedirs(save_path, exist_ok=True)

for f in files_source:
    INPUT = "noise"
    pad = "reflection"
    lr_img = 1e-2
    lr_kernel = 5e-4
    num_iter = opt.num_iter
    reg_noise_std = 3e-2
    patch_tv_loss_weight = 0
    tv_loss_weight = 0
    mcp_lambda = 0.01
    mcp_gamma = 15.0
    eta_k = 0.05
    eta_x = 0.01
    beta_k = 8.0
    beta_x = 5.0
    alpha_k = 0.1
    alpha_x = 0.1
    ckir_noise_std = 0.3
    ckir_num_candidates = 40

    path_to_image = f
    imgname = os.path.basename(f)
    imgname = os.path.splitext(imgname)[0]


    new_path = os.path.join(opt.save_path, "%s_work" % imgname)
    os.makedirs(new_path, exist_ok=True)
    imgs, y = get_color_image(path_to_image, -1)  # load image and convert to np.
    

    img_blur = np_to_torch(imgs).type(dtype)
    parts = imgname.split("_")
    gt_name = "_".join(parts[:2])

    img_size = imgs.shape

    # load groudtruth
    real_img_name = f"./datasets/lai/ground_truth/{imgname[:-8]}.png"
    
    img_real, y1 = get_color_image(real_img_name, -1)
    img_real = np_to_torch(img_real).type(dtype).squeeze()
    img_real_np = torch_to_np(img_real.permute(1, 2, 0).unsqueeze(0))

    netG_path = "models/lai/netG_nonuniform.pth"
    netE_path = "models/lai/netE_nonuniform.pth"

    # make the EFF
    eff = EFF(
        [img_size[1], img_size[2]],
        [opt.kernel_size[0], opt.kernel_size[1]],
        [opt.grid_size[0], opt.grid_size[1]],
    )
    eff.makeEFF()
    eff.cuda()
    n_grid = eff.grid_size[0] * eff.grid_size[1]
    # ######################################################################

    # extract y pactches
    Y2 = eff.padimg(img_blur, eff.pad)
    Y = eff.extractImageStack_y(Y2, opt.kernel_size, list(range(n_grid)))
    Y = Y.detach()

    input_depth = 8

    net_input = get_noise(
        input_depth, INPUT, (eff.padded_size[0], eff.padded_size[1])
    ).type(dtype)

    net = skip(
        input_depth,
        3,
        num_channels_down=[128, 128, 128, 128, 128],
        num_channels_up=[128, 128, 128, 128, 128],
        num_channels_skip=[16, 16, 16, 16, 16],
        upsample_mode="bilinear",
        need_sigmoid=True,
        need_bias=True,
        pad=pad,
        act_fun="LeakyReLU",
    )
    net = net.type(dtype)

    # Losses
    mse = nn.MSELoss().type(dtype)
    ssim_loss = SSIM().type(dtype)

    net_kernel_decoder = Generator(opt.kernel_size[0]).cuda()
    net_kernel_decoder = nn.DataParallel(net_kernel_decoder)
    net_kernel_decoder.load_state_dict(torch.load(netG_path))
    net_kernel_decoder.cuda()
    for p in net_kernel_decoder.parameters():
        p.requires_grad = False
    net_kernel_decoder.eval()

    net_kernel_encoder = ResNet18().cuda()
    net_kernel_encoder = nn.DataParallel(net_kernel_encoder)
    net_kernel_encoder.load_state_dict(torch.load(netE_path))
    for p in net_kernel_encoder.parameters():
        p.requires_grad = False
    net_kernel_encoder.eval()
    y = TVTF.to_tensor(y).unsqueeze(0).cuda()
    z = net_kernel_encoder(y)
    
    candidate_fes = [
        net_kernel_decoder.module.g1(
            (z + torch.randn_like(z) * ckir_noise_std).repeat(n_grid, 1, 1, 1)
        )
        for _ in range(ckir_num_candidates)
    ]
    best_fe = max(candidate_fes, key=lambda x: ssim_loss(eff.forward(net(net_input), net_kernel_decoder.module.Gk(x)
                                                                     .view(1, opt.grid_size[0]*opt.grid_size[1], opt.kernel_size[0], opt.kernel_size[1])), img_blur))
    fe = best_fe.requires_grad_(True)
 
    optimizer_img = torch.optim.Adam(net.parameters(), lr=lr_img)
    optimizer_kernel = torch.optim.Adam([{"params": [fe], "lr": lr_kernel}])
    scheduler_img = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_img, T_max=500)
    scheduler_kernel = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_kernel, T_max=500)

    net_input_saved = net_input.detach().clone()
    out_avg = None
    ux = None
    u = None

    tl.set_backend('pytorch')
    ### start SelfDeblur
    for step in tqdm(range(1, num_iter + 1)):
        # input regularization
        noise = reg_noise_std * torch.randn_like(net_input_saved)
        net_input = net_input_saved + noise

        ## kernel initialize
        if step == 1:
            if isinstance(net_kernel_decoder, nn.DataParallel):
                out_k = net_kernel_decoder.module.Gk(fe)
            else:
                out_k = net_kernel_decoder.Gk(fe)
            out_k = out_k.view(1, out_k.shape[0], opt.kernel_size[0], opt.kernel_size[1])
            save_path_ker = os.path.join(new_path, "initial_k.png")
            out_k_np_full = np.zeros(
                (
                    opt.kernel_size[0] * opt.grid_size[0] + 2 * (opt.grid_size[0] - 1),
                    opt.kernel_size[1] * opt.grid_size[1] + 2 * (opt.grid_size[1] - 1),
                )
            )
            for ii in range(opt.grid_size[0]):
                for jj in range(opt.grid_size[1]):
                    out_k_np = torch_to_np(out_k)
                    out_k_np = out_k_np.squeeze()
                    h = ii * opt.kernel_size[0] + (ii) * 2
                    w = jj * opt.kernel_size[1] + (jj) * 2
                    idx = ii * opt.grid_size[1] + jj
                    out_k_np[idx: idx + 1, :, :] /= np.max(
                        out_k_np[idx: idx + 1, :, :]
                    )
                    out_k_np_full[
                    h: h + opt.kernel_size[0], w: w + opt.kernel_size[1]
                    ] = out_k_np[idx: idx + 1, :, :]
            save_img_np(save_path_ker, out_k_np_full)

        ## update x with DTCCP/ADMM
        if isinstance(net_kernel_decoder, nn.DataParallel):
            out_k_c0 = net_kernel_decoder.module.Gk(fe).detach()
        else:
            out_k_c0 = net_kernel_decoder.Gk(fe).detach()
        out_k_c0 = out_k_c0.view(1, opt.grid_size[0]*opt.grid_size[1], opt.kernel_size[0], opt.kernel_size[1])

        out_x0 = net(net_input).detach()
        if ux is None:
            ux = torch.zeros_like(out_x0)
        ranks = [1, 3, eff.padded_size[0], eff.padded_size[1]]
        x_estimate = dtccp_tucker_prox(
            out_x0, ux, ranks, eta_x, alpha_x, mcp_lambda, mcp_gamma
        ).detach()

        optimizer_img.zero_grad()
        out_x = net(net_input)
        out_img = eff.forward(out_x, out_k_c0)
        if step < 500:
            fidelity_loss = 0.5 * mse(out_img, img_blur)
        else:
            fidelity_loss = 1 - ssim_loss(out_img, img_blur)
        loss_reg = 0.5 * beta_x * mse(out_x, x_estimate + ux)
        total_loss = fidelity_loss + loss_reg
        total_loss.backward()
        optimizer_img.step()
        scheduler_img.step()
        ux = (ux + x_estimate - out_x.detach()).detach()

        ## update k
        out_x0 = net(net_input).detach()
        if isinstance(net_kernel_decoder, nn.DataParallel):
            out_k_c0 = net_kernel_decoder.module.Gk(fe).detach().clone()
        else:
            out_k_c0 = net_kernel_decoder.Gk(fe).detach().clone()
        out_k_c0 = out_k_c0.view(1, opt.grid_size[0]*opt.grid_size[1], opt.kernel_size[0], opt.kernel_size[1])
        if u is None:
            u = torch.zeros_like(out_k_c0)
        ranks = [1, opt.grid_size[0]*opt.grid_size[1], opt.kernel_size[0], opt.kernel_size[1]]
        k_estimate = dtccp_tucker_prox(
            out_k_c0, u, ranks, eta_k, alpha_k, mcp_lambda, mcp_gamma
        ).detach()

        optimizer_kernel.zero_grad()
        if isinstance(net_kernel_decoder, nn.DataParallel):
            out_k_c = net_kernel_decoder.module.Gk(fe)
        else:
            out_k_c = net_kernel_decoder.Gk(fe)
        out_k_c = out_k_c.view(1, opt.grid_size[0]*opt.grid_size[1], opt.kernel_size[0], opt.kernel_size[1])
        out_img = eff.forward(out_x0, out_k_c)
        if step < 500:
            fidelity_loss = 0.5 * mse(out_img, img_blur)
        else:
            fidelity_loss = 1 - ssim_loss(out_img, img_blur)
        loss_reg = 0.5 * beta_k * mse(out_k_c, k_estimate + u)
        total_loss = fidelity_loss + loss_reg
        total_loss.backward()
        optimizer_kernel.step()
        scheduler_kernel.step()
        u = (u + k_estimate - out_k_c.detach()).detach()


        if (step) % 1000 == 0:
            correct_boundary(fe, opt.grid_size)

        if step % opt.save_frequency == 0:

            out_x_depad = eff.padimg(out_x, eff.pad, "depad")
            out_x_depad = torch.clamp(out_x_depad, 0, 1)
            out_x_np = torch_to_np(out_x_depad)
            out_x_np = out_x_np.squeeze()
            out_x_np = out_x_np.transpose(1, 2, 0)

            # psnr_step = getPSNR(out_x_np,img_real_np)
            # print(psnr_step)

            save_path_img = os.path.join(new_path, f"{step}_x.png")
            save_img_np(save_path_img, out_x_np)
            save_img_np(os.path.join(opt.save_path, f"{imgname}_x.png"), out_x_np)

            save_path_ker = os.path.join(new_path, f"{step}_k.png")
            out_k_np_full = np.zeros(
                (
                    opt.kernel_size[0] * opt.grid_size[0] + 2 * (opt.grid_size[0] - 1),
                    opt.kernel_size[1] * opt.grid_size[1] + 2 * (opt.grid_size[1] - 1),
                )
            )
            for ii in range(opt.grid_size[0]):
                for jj in range(opt.grid_size[1]):
                    out_k_np = torch_to_np(out_k_c)
                    out_k_np = out_k_np.squeeze()
                    h = ii * opt.kernel_size[0] + (ii) * 2
                    w = jj * opt.kernel_size[1] + (jj) * 2
                    idx = ii * opt.grid_size[1] + jj
                    out_k_np[idx: idx + 1, :, :] /= np.max(
                        out_k_np[idx: idx + 1, :, :]
                    )
                    out_k_np_full[
                    h: h + opt.kernel_size[0], w: w + opt.kernel_size[1]
                    ] = out_k_np[idx: idx + 1, :, :]
            save_img_np(save_path_ker, out_k_np_full)
