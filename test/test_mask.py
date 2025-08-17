from utils.mri_data import SliceDataset
from utils.transforms import to_tensor, complex_center_crop
from utils.fftc import ifft2c_new
from utils.math import complex_abs
from utils.coil_combine import rss
from utils.subsample import create_mask_for_mask_type
import torch
import matplotlib.pyplot as plt

maskClass = create_mask_for_mask_type(
    mask_type_str="one_sided", center_fractions=[0.08], accelerations=[4]
)


def transform(kspace, mask, target, attrs, name, dataslice):
    kspace = to_tensor(kspace)
    mask = maskClass(shape=kspace.shape)[0]
    masked_kspace = kspace * mask
    image = ifft2c_new(masked_kspace)
    crop_size = (image.shape[-2], image.shape[-2])
    image = complex_center_crop(image, crop_size)
    image = complex_abs(image)
    image = rss(image)
    return image, mask


dataset = SliceDataset(
    root="test/data",
    challenge="multicoil",
    transform=transform,
    num_coils=[4],
    regex_ex="T2",
)

for image, mask in dataset:
    plt.figure(figsize=(16, 8))
    plt.plot(mask.flatten(), marker=".", linestyle="None")
    plt.savefig("mask.png")
    acquired_cols = int(torch.sum(mask.flatten()).item())
    all_cols = mask.flatten().shape[0]
    print(
        acquired_cols,
        all_cols,
        f"{acquired_cols / all_cols:.4f}",
    )
    plt.axis(False)
    plt.imshow(image, cmap="gray")
    plt.savefig("test2.png")
    break
