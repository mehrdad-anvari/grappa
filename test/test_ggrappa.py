from utils.mri_data import SliceDataset
from utils.transforms import to_tensor, complex_center_crop, tensor_to_complex_np
from utils.fftc import ifft2c_new
from utils.math import complex_abs
from utils.coil_combine import rss
from utils.calib import get_calib

import matplotlib.pyplot as plt

from ggrappa.grappaND import GRAPPA_Recon
from grappa.grappa import grappa

import torch
import numpy as np

center_fractions = 0.08


def uniform_mask(shape, accel, center_fraction=0.08, seed=None):
    """
    Create a uniform undersampling mask for kspace.

    Args:
        kspace (torch.Tensor): (num_coils, ny, nx)
        accel (int): Acceleration factor (R).
        center_fraction (float): Fraction of low-freq lines to always keep.
        seed (int, optional): Random seed for reproducibility.

    Returns:
        torch.Tensor: Binary mask of shape (num_coils, ny, nx).
    """
    num_coils, nx, ny = shape
    rng = torch.Generator()
    if seed is not None:
        rng.manual_seed(seed)

    # number of low-freq lines
    num_low_freqs = int(round(ny * center_fraction))
    # uniform line positions
    mask = torch.zeros(ny, dtype=torch.float32)

    # center lines
    pad = (ny - num_low_freqs + 1) // 2
    mask[pad : pad + num_low_freqs] = 1

    # remaining lines: pick every accel-th line
    mask[::accel] = 1
    # expand to full kspace size
    mask = mask.view(1, 1, ny)  # (1, ny, 1)
    mask = mask.expand(num_coils, nx, ny)  # (num_coils, ny, nx)

    return mask


def transform(kspace, mask, target, attrs, name, dataslice):
    shape = kspace.shape
    kspace = to_tensor(kspace)
    calib = get_calib(torch.view_as_complex_copy(kspace), center_fractions)
    mask = uniform_mask(shape, 4, center_fractions).unsqueeze(-1)
    masked_kspace = kspace * mask

    plt.figure(dpi=1200)
    plt.axis(False)
    plt.title("undersampled k-space")
    crop_size = (masked_kspace.shape[-2], masked_kspace.shape[-2])
    cropped_mask = complex_center_crop(masked_kspace, crop_size)
    plt.imshow(
        np.log(np.abs(tensor_to_complex_np(cropped_mask)[0]) + 1e-6), cmap="gray"
    )
    plt.savefig("test/results/test_ggrappa_mask.png")
    return masked_kspace, kspace, calib, mask


dataset = SliceDataset(
    root="test/data",
    challenge="multicoil",
    transform=transform,
    # num_coils=[4],
    regex_ex="FLA",
)

for masked_kspace, original_kspace, calib, mask in dataset:
   

    # ggrappa
    masked_kspace: torch.Tensor
    calib: torch.Tensor

    sig_2d = torch.view_as_complex_copy(masked_kspace)  # (nc, kx, ky)
    acs_2d = calib  # (nc, acsky, acskx)

    # Insert dummy kz at axis=2 → (nc, ky, 1, kx)
    sig_4d = torch.permute(sig_2d.unsqueeze(2), (0, 3, 2, 1))
    acs_4d = torch.permute(acs_2d.unsqueeze(2), (0, 3, 2, 1))

    kspace1, t_weights, t_estimation = GRAPPA_Recon(
        sig_4d,  # (nc, ky, 1, kx)
        acs_4d,  # (nc, acsky, 1, acskx)
        af=[4, 1],  # [afy, afz] for 2D
        kernel_size=(5, 1, 5),  # (ky, kz, kx) with kz=1 in 2D
        lambda_=1e-2,
        cuda=True,
        cuda_mode="all",
        quiet=True,
    )

    print(f"Kernel weight calc:   {t_weights:.4f} s")
    print(f"Sample estimation:    {t_estimation:.4f} s")
    total_ggrappa = t_weights + t_estimation

    # Back to 2D shape for your downstream IFFT
    kspace1 = torch.permute(kspace1, (0, 3, 2, 1))
    kspace1 = kspace1.squeeze(2)  # (nc, kx, ky)
    kspace1 = torch.view_as_real(kspace1)

    kspace1 = kspace1.cpu()
    image = ifft2c_new(kspace1)
    crop_size = (image.shape[-2], image.shape[-2])
    image = complex_center_crop(image, crop_size)
    image = complex_abs(image)
    image = rss(image)

    plt.figure(dpi=1200)
    plt.axis(False)
    plt.title("ggrappa")
    plt.imshow(image, cmap="gray")
    plt.savefig("test/results/test_ggrappa_ggrappa.png")

    # proposed
    kspace2, (t_unique, t_weights, t_estimation) = grappa(
        torch.view_as_complex_copy(masked_kspace).cuda(),
        calib.cuda(),
        coil_axis=0,
        undersampling_pattern="2D",
    )

    print(f"Unique pattern search: {t_unique:.4f} s")
    print(f"Kernel weight calc:   {t_weights:.4f} s")
    print(f"Sample estimation:    {t_estimation:.4f} s")
    total_proposed = t_weights + t_estimation

    kspace2 = to_tensor(kspace2.cpu())
    image = ifft2c_new(kspace2)
    crop_size = (image.shape[-2], image.shape[-2])
    image = complex_center_crop(image, crop_size)
    image = complex_abs(image)
    image = rss(image)

    plt.figure(dpi=1200)
    plt.axis(False)
    plt.title("proposed")
    plt.imshow(image, cmap="gray")
    plt.savefig("test/results/test_ggrappa_proposed.png")

    print("speed up factor (T_ggrappa/T_proposed) = ", total_ggrappa / total_proposed)
    break
