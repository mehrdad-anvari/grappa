import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.ticker import LogLocator

import numpy as np

# Parameters
N_coils_list = [4, 12, 16, 20]
R_list = [4, 8]
kernel_sizes = ["3x3", "5x5"]
lamda_dict = {"3x3": 0.1, "5x5": 0.01}

# File path template
file_template = "ggrappa_kernel_coil_accel_T2_one_sided_{N_coils}_{R}_{kernel}_{lamda}.csv"

# Initialize results
results_proposed = {k: {r: [] for r in R_list} for k in kernel_sizes}
results_ggrappa = {k: {r: [] for r in R_list} for k in kernel_sizes}
results_ratio = {k: {r: [] for r in R_list} for k in kernel_sizes}

# Load data and compute sums
for kernel in kernel_sizes:
    lamda = lamda_dict[kernel]
    for R in R_list:
        for N_coils in N_coils_list:
            file_name = file_template.format(N_coils=N_coils, R=R, kernel=kernel, lamda=lamda)
            print(file_name)
            if not os.path.exists(os.path.join('../test/results/ggrappa_kernel_coil_accel/',file_name)):
                print(f"File not found: {file_name}")
                continue
            df = pd.read_csv(os.path.join('../test/results/ggrappa_kernel_coil_accel/',file_name))
            total_ggrappa =df["t_weights_ggrappa"].sum() + df["t_estimation_ggrappa"].sum()
            
            total_proposed =  df["t_weights_proposed"].sum() + df["t_estimation_proposed"].sum()
            

            results_proposed[kernel][R].append(total_proposed)
            results_ggrappa[kernel][R].append(total_ggrappa)
            results_ratio[kernel][R].append(total_ggrappa / total_proposed)

# --- Plot styling ---
plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 10,
    "figure.figsize": (7,5)
})

fig, ax = plt.subplots()
x = range(len(N_coils_list))  # Equal spacing for N_coils

# Define line styles for PyGRAPPA/Proposed
line_styles_pygrappa = {
    ("3x3", 4): ("-o", "black", "black"),
    ("3x3", 8): ("--o", "black", "white"),
    ("5x5", 4): ("-s", "black", "black"),
    ("5x5", 8): ("--s", "black", "white")
}

# Define line styles for GGRAPPA/Proposed (light blue, diff markers)
line_styles_ggrappa = {
    ("3x3", 4): ("-^", "deepskyblue", "deepskyblue"),
    ("3x3", 8): ("--^", "deepskyblue", "white"),
    ("5x5", 4): ("-D", "deepskyblue", "deepskyblue"),
    ("5x5", 8): ("--D", "deepskyblue", "white")
}

# --- Plot GGRAPPA/Proposed ---
for kernel in kernel_sizes:
    for R in R_list:
        style, color, mface = line_styles_ggrappa[(kernel, R)]
        yvals = np.array(results_ratio[kernel][R])
        ax.plot(x, yvals, style, color=color, markerfacecolor=mface,
                label=f"Kernel {kernel}, R={R}", markersize=6) 



# File path template
file_template = "kernel_coil_accel_T2_one_sided_{N_coils}_{R}_{kernel}_{lamda}.csv"

# Initialize results
results_proposed = {k: {r: [] for r in R_list} for k in kernel_sizes}
results_pygrappa = {k: {r: [] for r in R_list} for k in kernel_sizes}
results_ratio = {k: {r: [] for r in R_list} for k in kernel_sizes}


# Load data and compute sums
for kernel in kernel_sizes:
    lamda = lamda_dict[kernel]
    for R in R_list:
        for N_coils in N_coils_list:
            file_name = file_template.format(N_coils=N_coils, R=R, kernel=kernel, lamda=lamda)
            print(file_name)
            if not os.path.exists(os.path.join('../test/results/kernel_coil_accel/',file_name)):
                print(f"File not found: {file_name}")
                continue
            df = pd.read_csv(os.path.join('../test/results/kernel_coil_accel/',file_name))
            total_proposed = df["t_unique_proposed"].sum() + df["t_weights_proposed"].sum() + df["t_estimation_proposed"].sum()
            
            total_pygrappa = df["t_unique_pygrappa"].sum() + df["t_weights_pygrappa"].sum() + df["t_estimation_pygrappa"].sum()

            results_proposed[kernel][R].append(total_proposed)
            results_pygrappa[kernel][R].append(total_pygrappa)
            results_ratio[kernel][R].append(total_pygrappa / total_proposed)





# --- Plot PyGRAPPA/Proposed ---
for kernel in kernel_sizes:
    for R in R_list:
        style, color, mface = line_styles_pygrappa[(kernel, R)]
        yvals = np.array(results_ratio[kernel][R])
        ax.plot(x, yvals, style, color=color, markerfacecolor=mface,
                label=f"Kernel {kernel}, R={R}", markersize=6)
        
# --- Axes labels ---
ax.set_xticks(x)
ax.set_xticklabels(N_coils_list)
ax.set_xlabel("Number of Coils")

# Custom Y-axis: show original ratios, not log
def y_formatter(val, pos):
    return f"{np.exp(val) - 10:.1f}"  # invert transform

ax.yaxis.set_major_formatter(mticker.FuncFormatter(y_formatter))
ax.set_ylabel("Time Ratio")
ax.set_yscale("log")

ax.set_ylim(1, 140)   # Force y-axis range


# Force major ticks explicitly
ax.yaxis.set_major_locator(LogLocator(base=10.0, subs=np.arange(1.0, 10.0), numticks=20))

# Optional: force labels exactly where you want them
ax.set_yticks([1,2,4,6,8,10,20,40,60,80,100])
ax.get_yaxis().set_major_formatter(mticker.ScalarFormatter())
ax.ticklabel_format(style='plain', axis='y')


# Grid
ax.grid(True, which="both", ls="--", lw=0.5, color="gray", alpha=0.6)


# Legend option
legend_outside = True
if legend_outside:
    ax.legend(frameon=False, bbox_to_anchor=(1.05, 1), loc="upper left")
else:
    ax.legend(frameon=False, loc="best")

plt.tight_layout()
plt.savefig("../test/results/time_ratio_plot.png", dpi=600)   # Save as PNG, 600 DPI
plt.show()













plt.show()