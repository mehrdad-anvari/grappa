from utils.mri_data import SliceDataset
from utils.transforms import to_tensor, complex_center_crop
from utils.fftc import ifft2c_new
from utils.math import complex_abs
from utils.coil_combine import rss
from utils.subsample import create_mask_for_mask_type
from utils.evaluate import ssim, psnr, nmse

from grappa.grappa import grappa
from utils.pygrappa import grappa as pygrappa

import pandas as pd
import torch

center_fractions = 0.08
R = 4
mask_name = "one_sided"
N_coils = 4
contrast = "T2"
N = 50  # number of slices to process
kernel_size = (5, 5)
lamda_list = [0.3, 0.2, 0.1, 0.01, 0.001, 0.0001]


def get_calib(
    kspace: torch.Tensor, center_fraction: float = center_fractions
) -> torch.Tensor:
    """
    Extracts calibration region from k-space by keeping full sx
    and only a fraction of the center along sy.

    Parameters
    ----------
    kspace : torch.Tensor
        Input k-space of shape (coils, sx, sy).
    center_fraction : float
        Fraction of sy dimension to keep in the center (0 < f <= 1).

    Returns
    -------
    calib : torch.Tensor
        Calibration region of shape (coils, sx, sy_calib).
    """
    if kspace.ndim != 3:
        raise ValueError("Expected kspace shape (coils, sx, sy)")

    _, _, sy = kspace.shape

    # Compute size along phase-encode (sy)
    sy_calib = int(round(sy * center_fraction))
    sy_calib = max(sy_calib, 1)

    # Center index
    cy = sy // 2
    y_start, y_end = cy - sy_calib // 2, cy + (sy_calib + 1) // 2

    calib = kspace[:, :, y_start:y_end]

    return calib


def transform(kspace, mask, target, attrs, name, dataslice):
    kspace = to_tensor(kspace)
    calib = get_calib(torch.view_as_complex_copy(kspace), center_fractions)
    mask, _ = mask_func(kspace.shape)
    masked_kspace = kspace * mask - 0.0
    return masked_kspace, kspace, calib, mask


for lamda in lamda_list:
    csv_file_name = f"{contrast}_{mask_name}_{N_coils}_{R}_{kernel_size[0]}x{kernel_size[1]}_{lamda}"
    print(csv_file_name)

    mask_func = create_mask_for_mask_type(
        mask_type_str="one_sided",
        center_fractions=[center_fractions],
        accelerations=[R],
    )

    dataset = SliceDataset(
        root="test/data",
        challenge="multicoil",
        transform=transform,
        num_coils=[N_coils],
        regex_ex=contrast,
    )

    print("number of available slices in the dataset", len(dataset))
    records_data = []
    for i, (masked_kspace, original_kspace, calib, mask) in enumerate(dataset):
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
        kspace1, (proposed_t_unique, proposed_t_weights, proposed_t_estimation) = (
            grappa(
                torch.view_as_complex_copy(masked_kspace).cuda(),
                calib.cuda(),
                coil_axis=0,
                undersampling_pattern="2D",
                kernel_size=kernel_size,
                lamda=lamda,
            )
        )

        kspace1 = to_tensor(kspace1.cpu())
        image = ifft2c_new(kspace1)
        crop_size = (image.shape[-2], image.shape[-2])
        image = complex_center_crop(image, crop_size)
        image = complex_abs(image)
        image = rss(image)

        proposed_ssim = ssim(image_gt.numpy(), image.numpy())
        proposed_psnr = psnr(image_gt.numpy(), image.numpy())
        proposed_nmse = nmse(image_gt.numpy(), image.numpy())

        # pygrappa
        kspace = torch.view_as_complex_copy(masked_kspace).numpy()
        calib = calib.numpy()
        kspace2, (pygrappa_t_unique, pygrappa_t_weights, pygrappa_t_estimation) = (
            pygrappa(kspace, calib, coil_axis=0, kernel_size=kernel_size, lamda=lamda)
        )

        kspace2 = to_tensor(kspace2)
        image = ifft2c_new(kspace2)
        crop_size = (image.shape[-2], image.shape[-2])
        image = complex_center_crop(image, crop_size)
        image = complex_abs(image)
        image = rss(image)

        pygrappa_ssim = ssim(image_gt.numpy(), image.numpy())
        pygrappa_psnr = psnr(image_gt.numpy(), image.numpy())
        pygrappa_nmse = nmse(image_gt.numpy(), image.numpy())

        record = {
            "t_unique_proposed": proposed_t_unique,
            "t_weights_proposed": proposed_t_weights,
            "t_estimation_proposed": proposed_t_estimation,
            "t_unique_pygrappa": pygrappa_t_unique,
            "t_weights_pygrappa": pygrappa_t_weights,
            "t_estimation_pygrappa": pygrappa_t_estimation,
            "ssim_proposed": proposed_ssim,
            "psnr_proposed": proposed_psnr,
            "nmse_proposed": proposed_nmse,
            "ssim_pygrappa": pygrappa_ssim,
            "psnr_pygrappa": pygrappa_psnr,
            "nmse_pygrappa": pygrappa_nmse,
        }

        print(i, "/", N, ":\n", record)
        records_data.append(record)

        if i == N or i == len(dataset) - 1:
            break

    df = pd.DataFrame(records_data)
    df.to_csv(f"test/results/{csv_file_name}.csv", index=False)
