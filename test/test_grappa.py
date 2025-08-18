from utils.mri_data import SliceDataset
from utils.transforms import to_tensor, complex_center_crop, tensor_to_complex_np
from utils.fftc import ifft2c_new
from utils.math import complex_abs
from utils.coil_combine import rss
from utils.subsample import create_mask_for_mask_type
from utils.evaluate import ssim, psnr, nmse
from utils.calib import get_calib

import matplotlib.pyplot as plt

from grappa.grappa import grappa
from utils.pygrappa import grappa as pygrappa

import torch
import numpy as np

center_fractions = 0.08
mask_func = create_mask_for_mask_type(
    mask_type_str="one_sided", center_fractions=[center_fractions], accelerations=[4]
)


def transform(kspace, mask, target, attrs, name, dataslice):
    kspace = to_tensor(kspace)
    calib = get_calib(torch.view_as_complex_copy(kspace), center_fractions)
    mask, _ = mask_func(kspace.shape)
    masked_kspace = kspace * mask - 0.0
    plt.figure(dpi=1200)
    plt.axis(False)
    plt.title("undersampled k-space")
    crop_size = (masked_kspace.shape[-2], masked_kspace.shape[-2])
    cropped_mask = complex_center_crop(masked_kspace, crop_size)
    plt.imshow(
        np.log(np.abs(tensor_to_complex_np(cropped_mask)[0]) + 1e-6), cmap="gray"
    )
    plt.savefig("test/results/test_grappa_mask.png")
    return masked_kspace, kspace, calib, mask


dataset = SliceDataset(
    root="test/data",
    challenge="multicoil",
    transform=transform,
    num_coils=[4],
    regex_ex="T2",
)

for masked_kspace, original_kspace, calib, mask in dataset:
    # ground truth image
    original_kspace: torch.Tensor
    kspace0 = original_kspace.detach().clone()
    image_gt = ifft2c_new(kspace0)
    crop_size = (image_gt.shape[-2], image_gt.shape[-2])
    image_gt = complex_center_crop(image_gt, crop_size)
    image_gt = complex_abs(image_gt)
    image_gt = rss(image_gt)

    # proposed grappa
    masked_kspace: torch.Tensor
    calib: torch.Tensor
    kspace1, (t_unique, t_weights, t_estimation) = grappa(
        torch.view_as_complex_copy(masked_kspace).cuda(),
        calib.cuda(),
        coil_axis=0,
        undersampling_pattern="2D",
    )

    print(f"Unique pattern search: {t_unique:.4f} s")
    print(f"Kernel weight calc:   {t_weights:.4f} s")
    print(f"Sample estimation:    {t_estimation:.4f} s")
    total_proposed = t_unique + t_weights + t_estimation

    kspace1 = to_tensor(kspace1.cpu())
    image = ifft2c_new(kspace1)
    crop_size = (image.shape[-2], image.shape[-2])
    image = complex_center_crop(image, crop_size)
    image = complex_abs(image)
    image = rss(image)

    print("SSIM: ", ssim(image_gt.numpy(), image.numpy()))
    print("PSNR: ", psnr(image_gt.numpy(), image.numpy()))
    print("NMSE: ", nmse(image_gt.numpy(), image.numpy()))

    plt.figure(dpi=1200)
    plt.axis(False)
    plt.title("proposed")
    plt.imshow(image, cmap="gray")
    plt.savefig("test/results/proposed_grappa.png")

    # pygrappa
    kspace = torch.view_as_complex_copy(masked_kspace).numpy()
    calib = calib.numpy()
    kspace2, (t_unique, t_weights, t_estimation) = pygrappa(kspace, calib, coil_axis=0)

    print(f"Unique pattern search: {t_unique:.4f} s")
    print(f"Kernel weight calc:   {t_weights:.4f} s")
    print(f"Sample estimation:    {t_estimation:.4f} s")
    total_pygrappa = t_unique + t_weights + t_estimation

    kspace2 = to_tensor(kspace2)
    image = ifft2c_new(kspace2)
    crop_size = (image.shape[-2], image.shape[-2])
    image = complex_center_crop(image, crop_size)
    image = complex_abs(image)
    image = rss(image)

    print("SSIM: ", ssim(image_gt.numpy(), image.numpy()))
    print("PSNR: ", psnr(image_gt.numpy(), image.numpy()))
    print("NMSE: ", nmse(image_gt.numpy(), image.numpy()))

    plt.figure(dpi=1200)
    plt.axis(False)
    plt.title("pygrappa")
    plt.imshow(image, cmap="gray")
    plt.savefig("test/results/pygrappa_grappa.png")

    print("speed up factor (T_pygrappa/T_proposed) = ", total_pygrappa / total_proposed)
    break
