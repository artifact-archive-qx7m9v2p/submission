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

print("\n1. Creating posterior distributions for mu and tau...")
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Mu posterior
axes[0].hist(mu_samples, bins=50, density=True, alpha=0.7, color='steelblue', edgecolor='black')
axes[0].axvline(np.mean(mu_samples), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(mu_samples):.2f}')
axes[0].axvline(np.percentile(mu_samples, 2.5), color='orange', linestyle=':', linewidth=1.5, label='95% CI')
axes[0].axvline(np.percentile(mu_samples, 97.5), color='orange', linestyle=':', linewidth=1.5)
axes[0].set_xlabel('Overall effect (mu)')
axes[0].set_ylabel('Density')
axes[0].set_title(f'Posterior: mu\nMean: {np.mean(mu_samples):.2f}, 95% CI: [{np.percentile(mu_samples, 2.5):.2f}, {np.percentile(mu_samples, 97.5):.2f}]')
axes[0].legend()
axes[0].grid(alpha=0.3)

# Tau posterior
axes[1].hist(tau_samples, bins=50, density=True, alpha=0.7, color='forestgreen', edgecolor='black')
axes[1].axvline(np.median(tau_samples), color='red', linestyle='--', linewidth=2, label=f'Median: {np.median(tau_samples):.2f}')
axes[1].axvline(np.percentile(tau_samples, 2.5), color='orange', linestyle=':', linewidth=1.5, label='95% CI')
axes[1].axvline(np.percentile(tau_samples, 97.5), color='orange', linestyle=':', linewidth=1.5)
axes[1].set_xlabel('Between-study heterogeneity (tau)')
axes[1].set_ylabel('Density')
axes[1].set_title(f'Posterior: tau\nMedian: {np.median(tau_samples):.2f}, 95% CI: [{np.percentile(tau_samples, 2.5):.2f}, {np.percentile(tau_samples, 97.5):.2f}]')
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(PLOT_DIR / "posterior_mu_tau.png", dpi=150, bbox_inches='tight')
print(f"   Saved: posterior_mu_tau.png")
plt.close()

print("\n2. Creating joint posterior (mu, tau)...")
fig, ax = plt.subplots(figsize=(8, 7))
h = ax.hexbin(mu_samples, tau_samples, gridsize=40, cmap='YlGnBu', mincnt=1)
ax.axvline(np.mean(mu_samples), color='red', linestyle='--', alpha=0.7, label=f'Mean mu: {np.mean(mu_samples):.2f}')
ax.axhline(np.median(tau_samples), color='orange', linestyle='--', alpha=0.7, label=f'Median tau: {np.median(tau_samples):.2f}')
ax.set_xlabel('Overall effect (mu)')
ax.set_ylabel('Between-study heterogeneity (tau)')
ax.set_title('Joint Posterior Distribution: (mu, tau)')
ax.legend()
ax.grid(alpha=0.3)
plt.colorbar(h, ax=ax, label='Density')
plt.tight_layout()
plt.savefig(PLOT_DIR / "joint_posterior_mu_tau.png", dpi=150, bbox_inches='tight')
print(f"   Saved: joint_posterior_mu_tau.png")
plt.close()

print("\n3. Creating forest plot with shrinkage...")
# Compute study-specific posterior means and CIs
theta_means = []
theta_lowers = []
theta_uppers = []
for j in range(len(df)):
    theta_j = theta_samples[:, :, j].flatten()
    theta_means.append(np.mean(theta_j))
    theta_lowers.append(np.percentile(theta_j, 2.5))
    theta_uppers.append(np.percentile(theta_j, 97.5))

mu_mean = np.mean(mu_samples)
mu_lower = np.percentile(mu_samples, 2.5)
mu_upper = np.percentile(mu_samples, 97.5)

fig, ax = plt.subplots(figsize=(10, 8))

# Plot observed effects
y_pos = np.arange(len(df)) * 2
ax.errorbar(df['y'].values, y_pos, xerr=1.96*df['sigma'].values,
            fmt='s', markersize=8, color='gray', alpha=0.5,
            capsize=5, capthick=2, label='Observed (y_i ± 1.96*sigma_i)')

# Plot posterior effects
ax.errorbar(theta_means, y_pos + 0.5,
            xerr=[np.array(theta_means) - np.array(theta_lowers),
                  np.array(theta_uppers) - np.array(theta_means)],
            fmt='o', markersize=8, color='steelblue',
            capsize=5, capthick=2, label='Posterior (theta_i, 95% CI)')

# Plot overall effect
ax.axvline(mu_mean, color='red', linestyle='--', linewidth=2, alpha=0.7)
ax.axvspan(mu_lower, mu_upper, alpha=0.2, color='red', label=f'Overall effect (mu, 95% CI)')

# Add arrows showing shrinkage
for j in range(len(df)):
    ax.annotate('', xy=(theta_means[j], y_pos[j] + 0.5), xytext=(df['y'].values[j], y_pos[j]),
                arrowprops=dict(arrowstyle='->', color='black', alpha=0.3, lw=1))

ax.set_yticks(y_pos + 0.25)
ax.set_yticklabels([f'Study {i+1}' for i in range(len(df))])
ax.set_xlabel('Effect Size')
ax.set_title('Forest Plot: Posterior Shrinkage\n(Arrows show shrinkage from observed to posterior mean)')
ax.legend(loc='best')
ax.grid(alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig(PLOT_DIR / "forest_plot_shrinkage.png", dpi=150, bbox_inches='tight')
print(f"   Saved: forest_plot_shrinkage.png")
plt.close()

print("\n4. Creating study-specific posterior distributions...")
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.flatten()

for j in range(len(df)):
    theta_j = theta_samples[:, :, j].flatten()
    ax = axes[j]

    ax.hist(theta_j, bins=40, density=True, alpha=0.7, color='steelblue', edgecolor='black')
    ax.axvline(df['y'].values[j], color='gray', linestyle='--', linewidth=2, label=f'Observed: {df["y"].values[j]:.1f}')
    ax.axvline(np.mean(theta_j), color='red', linestyle='-', linewidth=2, label=f'Posterior: {np.mean(theta_j):.1f}')
    ax.axvline(mu_mean, color='orange', linestyle=':', linewidth=2, label=f'Overall: {mu_mean:.1f}')

    ax.set_xlabel('Effect Size')
    ax.set_ylabel('Density')
    ax.set_title(f'Study {j+1}: y={df["y"].values[j]}, σ={df["sigma"].values[j]}')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(PLOT_DIR / "study_specific_posteriors.png", dpi=150, bbox_inches='tight')
print(f"   Saved: study_specific_posteriors.png")
plt.close()

print("\n5. Creating probability statements plot...")
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# P(mu > x)
x_vals = np.linspace(-10, 20, 100)
prob_mu_gt = [np.mean(mu_samples > x) for x in x_vals]
axes[0, 0].plot(x_vals, prob_mu_gt, linewidth=2, color='steelblue')
axes[0, 0].axhline(0.5, color='red', linestyle='--', alpha=0.5)
axes[0, 0].axvline(0, color='orange', linestyle=':', alpha=0.5, label=f'P(mu > 0) = {np.mean(mu_samples > 0):.3f}')
axes[0, 0].axvline(5, color='green', linestyle=':', alpha=0.5, label=f'P(mu > 5) = {np.mean(mu_samples > 5):.3f}')
axes[0, 0].set_xlabel('x')
axes[0, 0].set_ylabel('P(mu > x)')
axes[0, 0].set_title('Probability mu Exceeds Threshold')
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

# P(tau < x)
x_vals_tau = np.linspace(0, 15, 100)
prob_tau_lt = [np.mean(tau_samples < x) for x in x_vals_tau]
axes[0, 1].plot(x_vals_tau, prob_tau_lt, linewidth=2, color='forestgreen')
axes[0, 1].axhline(0.5, color='red', linestyle='--', alpha=0.5)
axes[0, 1].axvline(1, color='orange', linestyle=':', alpha=0.5, label=f'P(tau < 1) = {np.mean(tau_samples < 1):.3f}')
axes[0, 1].axvline(5, color='purple', linestyle=':', alpha=0.5, label=f'P(tau < 5) = {np.mean(tau_samples < 5):.3f}')
axes[0, 1].set_xlabel('x')
axes[0, 1].set_ylabel('P(tau < x)')
axes[0, 1].set_title('Probability tau Below Threshold')
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.3)

# Distribution of shrinkage
shrinkage_df = pd.read_csv(DIAG_DIR / "shrinkage_stats.csv")
axes[1, 0].bar(shrinkage_df['study'], shrinkage_df['shrinkage'], color='steelblue', edgecolor='black')
axes[1, 0].axhline(0, color='black', linewidth=1)
axes[1, 0].set_xlabel('Study')
axes[1, 0].set_ylabel('Shrinkage Factor')
axes[1, 0].set_title('Shrinkage by Study\n(Proportion of distance from y_i to mu that theta_i moved)')
axes[1, 0].set_xticks(shrinkage_df['study'])
axes[1, 0].grid(alpha=0.3, axis='y')

# Posterior precision (1/tau^2)
precision_samples = 1 / (tau_samples ** 2)
axes[1, 1].hist(precision_samples[precision_samples < 100], bins=50, density=True, alpha=0.7, color='purple', edgecolor='black')
axes[1, 1].axvline(np.median(precision_samples), color='red', linestyle='--', linewidth=2, label=f'Median: {np.median(precision_samples):.2f}')
axes[1, 1].set_xlabel('Precision (1/tau^2)')
axes[1, 1].set_ylabel('Density')
axes[1, 1].set_title('Posterior: Between-study Precision')
axes[1, 1].legend()
axes[1, 1].grid(alpha=0.3)
axes[1, 1].set_xlim(0, 100)

plt.tight_layout()
plt.savefig(PLOT_DIR / "probability_statements.png", dpi=150, bbox_inches='tight')
print(f"   Saved: probability_statements.png")
plt.close()

print("\n6. Creating posterior predictive check plot...")
y_rep = idata.posterior_predictive['y_rep'].values  # shape: (chains, draws, studies)

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

print("\nAll posterior plots created successfully!")
