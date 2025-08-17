from utils.mri_data import SliceDataset
from utils.transforms import to_tensor, complex_center_crop
from utils.fftc import ifft2c_new
from utils.math import complex_abs
from utils.coil_combine import rss
from utils.subsample import MagicMaskFractionFunc
import matplotlib.pyplot as plt

from grappa.grappa import grappa
from utils.pygrappa import grappa as pygrappa

import torch

center_fractions = 0.08
maskClass = MagicMaskFractionFunc(
    center_fractions=[center_fractions], accelerations=[4]
)


def get_calib(kspace: torch.Tensor, center_fraction: float = 0.08) -> torch.Tensor:
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

    coils, sx, sy = kspace.shape

    # Compute size along phase-encode (sy)
    sy_calib = int(round(sy * center_fraction))
    sy_calib = max(sy_calib, 1)

    # Center index
    cy = sy // 2
    y_start, y_end = cy - sy_calib // 2, cy + (sy_calib + 1) // 2

    calib = kspace[:, :, y_start:y_end]

    return calib


def transform(kspace, mask, target, attrs, name, dataslice):
    kspace = torch.tensor(kspace)
    calib = get_calib(kspace, center_fractions)
    mask = maskClass(shape=kspace.shape)[0]
    masked_kspace = kspace * mask
    return masked_kspace, calib, mask


dataset = SliceDataset(
    root="test/data",
    challenge="multicoil",
    transform=transform,
    num_coils=[4],
    regex_ex="T2",
)

for kspace, calib, mask in dataset:
    kspace: torch.Tensor
    calib: torch.Tensor
    kspace1, (t_unique, t_weights, t_estimation) = grappa(
        kspace.detach().clone().cuda(), calib.cuda(), coil_axis=0
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

    plt.imshow(image, cmap="gray")
    plt.savefig("test/results/proposed_grappa.png")

    kspace = kspace.detach().clone().numpy()
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

    plt.imshow(image, cmap="gray")
    plt.savefig("test/results/pygrappa_grappa.png")

    print("speed up factor (T_pygrappa/T_proposed) = ", total_pygrappa/total_proposed)
    break
