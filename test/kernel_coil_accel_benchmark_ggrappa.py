from benchmark_ggrappa import benchmark

def sweep_kernels_coils_accel():
    configs = [
        {"kernel_size": (3, 3), "lamda": 0.1},
        {"kernel_size": (5, 5), "lamda": 0.01},
    ]
    coil_list = [4, 12, 16, 20]
    accel_configs = [
        {"R": 4, "center_fraction": 0.08},
        {"R": 8, "center_fraction": 0.04},
    ]

    for cfg in configs:
        for n_coils in coil_list:
            for accel in accel_configs:
                print(
                    f"\nRunning benchmark with kernel_size={cfg['kernel_size']}, "
                    f"lamda={cfg['lamda']}, coils={n_coils}, "
                    f"R={accel['R']}, center_fraction={accel['center_fraction']}"
                )

                benchmark(
                    contrast="T2",
                    mask_name="one_sided",
                    N_coils=n_coils,
                    N=50,
                    R=accel["R"],
                    center_fractions=accel["center_fraction"],
                    kernel_size=cfg["kernel_size"],
                    lamda=cfg["lamda"],
                    prefix="ggrappa_kernel_coil_accel",
                )

if __name__ == "__main__":
    sweep_kernels_coils_accel()
