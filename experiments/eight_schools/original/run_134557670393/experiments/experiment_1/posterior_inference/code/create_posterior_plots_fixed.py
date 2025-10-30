"""
Create posterior visualization plots
"""

import sys
sys.path.insert(0, '/tmp/agent-home/.local/lib/python3.13/site-packages')

import arviz as az
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.size'] = 10

# Paths
OUTPUT_DIR = Path("/workspace/experiments/experiment_1/posterior_inference")
DIAG_DIR = OUTPUT_DIR / "diagnostics"
PLOT_DIR = OUTPUT_DIR / "plots"
IDATA_PATH = DIAG_DIR / "posterior_inference.netcdf"
DATA_PATH = "/workspace/data/data.csv"

print("Loading data and InferenceData...")
idata = az.from_netcdf(IDATA_PATH)
df = pd.read_csv(DATA_PATH)

# Extract posterior samples
mu_samples = idata.posterior['mu'].values.flatten()
tau_samples = idata.posterior['tau'].values.flatten()
theta_samples = idata.posterior['theta'].values  # shape: (chains, draws, studies)

print("\n6. Creating posterior predictive check plot...")
# y_rep is in posterior group for PyMC
if 'y_rep' in idata.posterior.data_vars:
    y_rep = idata.posterior['y_rep'].values  # shape: (chains, draws, studies)

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()

    for j in range(len(df)):
        y_rep_j = y_rep[:, :, j].flatten()
        ax = axes[j]

        # Posterior predictive distribution
        ax.hist(y_rep_j, bins=40, density=True, alpha=0.6, color='lightblue', edgecolor='black', label='Posterior predictive')

        # Observed value
        ax.axvline(df['y'].values[j], color='red', linestyle='--', linewidth=2, label=f'Observed: {df["y"].values[j]}')

        # 95% predictive interval
        y_rep_lower = np.percentile(y_rep_j, 2.5)
        y_rep_upper = np.percentile(y_rep_j, 97.5)
        ax.axvline(y_rep_lower, color='orange', linestyle=':', linewidth=1.5)
        ax.axvline(y_rep_upper, color='orange', linestyle=':', linewidth=1.5)

        ax.set_xlabel('Effect Size')
        ax.set_ylabel('Density')
        ax.set_title(f'Study {j+1}: PPC\nObserved in 95% PI: {y_rep_lower <= df["y"].values[j] <= y_rep_upper}')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(PLOT_DIR / "posterior_predictive_check.png", dpi=150, bbox_inches='tight')
    print(f"   Saved: posterior_predictive_check.png")
    plt.close()
else:
    print("   Skipping PPC plot: y_rep not found")

print("\nPPC plot created successfully!")
