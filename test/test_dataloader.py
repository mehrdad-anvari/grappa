from utils.mri_data import SliceDataset
from utils.transforms import to_tensor, complex_center_crop
from utils.fftc import ifft2c_new
from utils.math import complex_abs
from utils.coil_combine import rss

import matplotlib.pyplot as plt

dataset = SliceDataset(
    root="test/data", challenge="multicoil", num_coils=[4], regex_ex="T2"
)

for kspace, mask, target, attrs, name, dataslice in dataset:
    print(attrs)  # Should print the shape of the kspace tensor
    masked_kspace = to_tensor(kspace)
    image = ifft2c_new(masked_kspace)
    crop_size = (image.shape[-2], image.shape[-2])
    image = complex_center_crop(image, crop_size)
    image = complex_abs(image)
    image = rss(image)
    plt.axis(False)
    plt.imshow(image, cmap="gray")
    plt.savefig("test.png")
    break
