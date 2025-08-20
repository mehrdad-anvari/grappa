
import os
# os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.mri_data import SliceDataset
from utils.transforms import to_tensor, complex_center_crop
from utils.fftc import ifft2c_new
from utils.calib import get_calib
from utils.math import complex_abs
from utils.coil_combine import rss
from utils.evaluate import ssim, psnr, nmse
from ggrappa.grappaND import GRAPPA_Recon
from grappa.grappa import grappa
from utils.pygrappa import grappa as pygrappa
import torch
import torch.nn.functional as F
import pandas as pd
import argparse
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "Times New Roman",   # or just "Times"
    "font.size": 5
})
def benchmark(
    contrast: str = "T2",
    mask_name: str = "standard",
    N_coils: int = 4,
    N: int = 50,
    R: int = 4,
    center_fractions: float = 0.08,
    kernel_size: tuple = (5, 5),
    lamda: float = 0.1,
    prefix: str = "",
):
    def uniform_mask(shape, accel, center_fraction=center_fractions, seed=None):
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
        mask = uniform_mask(shape, R, center_fractions).unsqueeze(-1)
        masked_kspace = kspace.detach().clone() * mask

        return masked_kspace, kspace, calib, mask

    dataset = SliceDataset(
        root="../test/data",
        challenge="multicoil",
        transform=transform,
        num_coils=[N_coils],
        regex_ex=contrast,
    )
    csv_file_name = f"{prefix}_{contrast}_{mask_name}_{N_coils}_{R}_{kernel_size[0]}x{kernel_size[1]}_{lamda}"
    records_data = []
    for i, (masked_kspace, original_kspace, calib, mask) in enumerate(dataset):
        original_kspace: torch.Tensor
        image_gt = ifft2c_new(original_kspace)
        crop_size = (image_gt.shape[-2], image_gt.shape[-2])
        image_gt = complex_center_crop(image_gt, crop_size)
        image_gt = complex_abs(image_gt)
        image_gt = rss(image_gt)

        # ggrappa
        masked_kspace: torch.Tensor
        calib: torch.Tensor

        sig_2d = torch.view_as_complex_copy(masked_kspace)  # (nc, kx, ky)
        acs_2d = calib.detach().clone()  # (nc, acsky, acskx)

        # Insert dummy kz at axis=2 → (nc, ky, 1, kx)
        sig_4d = torch.permute(sig_2d.unsqueeze(2), (0, 3, 2, 1))
        acs_4d = torch.permute(acs_2d.unsqueeze(2), (0, 3, 2, 1))

        kspace1, ggrappa_t_weights, ggrappa_t_estimation = GRAPPA_Recon(
            sig_4d,  # (nc, ky, 1, kx)
            acs_4d,  # (nc, acsky, 1, acskx)
            af=[R, 1],  # [afy, afz] for 2D
            kernel_size=(kernel_size[0], 1, kernel_size[1]),  # (ky, kz, kx) with kz=1 in 2D
            lambda_=lamda,
            cuda=True,
            cuda_mode="all",
            quiet=True,
        )

        # Back to 2D shape for your downstream IFFT
        kspace1 = torch.permute(kspace1, (0, 3, 2, 1))
        kspace1 = kspace1.squeeze(2)  # (nc, kx, ky)
        kspace1 = torch.view_as_real(kspace1)

        kspace1 = kspace1.cpu()

        # Padding
        pad_ky = (masked_kspace.shape[1] - kspace1.shape[1], 0)
        pad_kx = (masked_kspace.shape[2] - kspace1.shape[2], 0)

        pad = (0, 0, *pad_kx, *pad_ky)

        kspace1 = F.pad(kspace1, pad)

        image = ifft2c_new(kspace1)
        crop_size = (image.shape[-2], image.shape[-2])
        
        image = complex_center_crop(image, crop_size)
        image = complex_abs(image)
        image_ggrappa = rss(image)

        # proposed
        kspace2, (t_unique, proposed_t_weights, proposed_t_estimation) = grappa(
            torch.view_as_complex_copy(masked_kspace).cuda(),
            calib.cuda(),
            coil_axis=0,
            lamda=lamda,
            kernel_size=kernel_size,
            undersampling_pattern="2D",
        )

        kspace2 = to_tensor(kspace2.cpu())
        image = ifft2c_new(kspace2)

        crop_size = (image.shape[-2], image.shape[-2])
        image = complex_center_crop(image, crop_size)
        image = complex_abs(image)
        image_proposed = rss(image)


        # undersampled
        kspace = torch.view_as_complex_copy(masked_kspace).numpy()
        # calib = calib.numpy()
        kspace2 = kspace.copy()

        kspace2 = to_tensor(kspace2)
        image = ifft2c_new(kspace2)
        crop_size = (image.shape[-2], image.shape[-2])
        image = complex_center_crop(image, crop_size)
        image = complex_abs(image)
        image_undersampled = rss(image)


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
        image_pygrappa = rss(image)



        image_gt_minus_proposed = image_gt - image_proposed
        image_gt_minus_pygrappa = image_gt - image_pygrappa
        image_gt_minus_ggrappa = image_gt - image_ggrappa

        images = [
            (image_gt, "Ground Truth"),
            (image_undersampled, "Undersampled"),
            (image_proposed, "Proposed"),
            (image_gt_minus_proposed, "Difference (Proposed)"),
            (image_pygrappa, "PyGRAPPA"),
            (image_gt_minus_pygrappa, "Difference (PyGRAPPA)"),
            (image_ggrappa, "GGRAPPA"),
            (image_gt_minus_ggrappa, "Difference (GGRAPPA)")
        ]

        fig, axes = plt.subplots(4, 2, figsize=(2.8, 6))  # 4 rows, 2 cols

        for ax, (img, title) in zip(axes.ravel(), images):
            ax.imshow(img, cmap="gray")    # or cmap="viridis" etc
            ax.set_title(title)
            ax.axis("off")                 # Hide axes

        plt.tight_layout()
        plt.savefig("../test/results/sample_plot.png", dpi=300)   # Save as PNG, 600 DPI
        plt.show()
        break

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark GRAPPA implementations")

    parser.add_argument(
        "--contrast", type=str, default="T2", help="Contrast type (e.g., T1, T2, FLAIR)"
    )
    parser.add_argument(
        "--mask_name",
        type=str,
        default="one_sided",
        help="Mask type (e.g., one_sided, random)",
    )
    parser.add_argument("--N_coils", type=int, default=4, help="Number of coils")
    parser.add_argument("--N", type=int, default=50, help="Number of slices to process")
    parser.add_argument("--R", type=int, default=4, help="Acceleration factor")
    parser.add_argument(
        "--center_fractions",
        type=float,
        default=0.08,
        help="Fraction of low-frequencies to keep in mask",
    )
    parser.add_argument(
        "--kernel_size",
        type=int,
        nargs=2,
        default=(5, 5),
        help="Kernel size for GRAPPA (two integers)",
    )
    parser.add_argument(
        "--lamda", type=float, default=0.1, help="Regularization parameter"
    )
    parser.add_argument(
        "--prefix", type=str, default="", help="Prefix for output CSV file"
    )

    args = parser.parse_args()

    benchmark(
        contrast=args.contrast,
        mask_name=args.mask_name,
        N_coils=args.N_coils,
        N=args.N,
        R=args.R,
        center_fractions=args.center_fractions,
        kernel_size=tuple(args.kernel_size),
        lamda=args.lamda,
        prefix=args.prefix,
    )