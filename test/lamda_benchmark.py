from benchmark import benchmark

def sweep_lamda():
    lamda_list = [0.3, 0.2, 0.1, 0.01, 0.001, 0.0001]
    kernel_sizes = [(3, 3), (5, 5)]

    for lamda in lamda_list:
        for kernel in kernel_sizes:
            print(f"\nRunning benchmark with lamda={lamda}, kernel_size={kernel}")

            benchmark(
                contrast="T2",
                mask_name="one_sided",
                N_coils=4,
                N=50,
                R=4,
                center_fractions=0.08,
                kernel_size=kernel,
                lamda=lamda,
                prefix="lamda_sweep",
            )

if __name__ == "__main__":
    sweep_lamda()