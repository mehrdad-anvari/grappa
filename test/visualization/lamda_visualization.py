import pandas as pd
import matplotlib.pyplot as plt
import os

# File mapping
files = {
    "3x3": {
        0.3:   "lamda_sweep_T2_one_sided_4_4_3x3_0.3.csv",
        0.2:   "lamda_sweep_T2_one_sided_4_4_3x3_0.2.csv",
        0.1:   "lamda_sweep_T2_one_sided_4_4_3x3_0.1.csv",
        0.01:  "lamda_sweep_T2_one_sided_4_4_3x3_0.01.csv",
        0.001: "lamda_sweep_T2_one_sided_4_4_3x3_0.001.csv",
        0.0001:"lamda_sweep_T2_one_sided_4_4_3x3_0.0001.csv"
    },
    "5x5": {
        0.3:   "lamda_sweep_T2_one_sided_4_4_5x5_0.3.csv",
        0.2:   "lamda_sweep_T2_one_sided_4_4_5x5_0.2.csv",
        0.1:   "lamda_sweep_T2_one_sided_4_4_5x5_0.1.csv",
        0.01:  "lamda_sweep_T2_one_sided_4_4_5x5_0.01.csv",
        0.001: "lamda_sweep_T2_one_sided_4_4_5x5_0.001.csv",
        0.0001:"lamda_sweep_T2_one_sided_4_4_5x5_0.0001.csv"
    }
}

results = {"3x3": [], "5x5": []}

for kernel, lam_dict in files.items():
    for lam, file in lam_dict.items():
        df = pd.read_csv(os.path.join('../test/results/lamda_sweep',file))
        ssim = df["ssim_proposed"]
        results[kernel].append((lam, ssim.mean()))

# Sort by lambda
for k in results:
    results[k] = sorted(results[k], key=lambda x: x[0], reverse=True)

# Plot styling
plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 10,
    "figure.figsize": (6,4)
})

fig, ax = plt.subplots()

# Kernel 3x3: solid line, filled circles
lams = [r[0] for r in results["3x3"]]
means = [r[1] for r in results["3x3"]]
ax.plot(lams, means, "-o", color="black", markerfacecolor="black", label="Kernel 3×3")

# Kernel 5x5: dashed line, open circles
lams = [r[0] for r in results["5x5"]]
means = [r[1] for r in results["5x5"]]
ax.plot(lams, means, "--o", color="black", markerfacecolor="white", label="Kernel 5×5")

# Axes setup
ax.set_xscale("log")
ax.set_xlabel("Regularization $\\lambda$")
ax.set_ylabel("SSIM (Proposed)")
ax.set_ylim([0.7, 0.82])
ax.legend(frameon=False)
ax.grid(True, which="both", ls="--", lw=0.5, color="gray", alpha=0.6)

plt.tight_layout()
plt.savefig("../test/results/lamda_plot.png", dpi=600)   # Save as PNG, 600 DPI
plt.show()