"""
Create a single summary plot highlighting the critical prior issues.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats

# Set random seed
np.random.seed(42)

# Paths
OUTPUT_DIR = Path("/workspace/experiments/experiment_2/prior_predictive_check")
PLOTS_DIR = OUTPUT_DIR / "plots"

# Set style
sns.set_style("whitegrid")

# ============================================================================
# Create 2x2 summary figure
# ============================================================================

fig = plt.figure(figsize=(16, 12))
gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

# ----------------------------------------------------------------------------
# Panel 1: Current vs Proposed Phi Prior
# ----------------------------------------------------------------------------
ax1 = fig.add_subplot(gs[0, 0])

# Current prior: Uniform(-0.95, 0.95)
phi_current = np.random.uniform(-0.95, 0.95, 10000)

# Proposed prior: Beta(20, 2) rescaled to (0, 0.95)
phi_proposed = 0.95 * np.random.beta(20, 2, 10000)

# Observed ACF
obs_acf = 0.961

ax1.hist(phi_current, bins=50, alpha=0.5, color='red', label='Current: U(-0.95, 0.95)', density=True)
ax1.hist(phi_proposed, bins=50, alpha=0.5, color='green', label='Proposed: Beta(20,2) scaled', density=True)
ax1.axvline(obs_acf, color='blue', linestyle='--', linewidth=3, label=f'Observed ACF = {obs_acf}')
ax1.set_xlabel('phi (AR coefficient)', fontsize=13, fontweight='bold')
ax1.set_ylabel('Density', fontsize=13, fontweight='bold')
ax1.set_title('ISSUE 1: Phi Prior Mismatch\nCurrent prior doesn\'t favor high autocorrelation',
              fontsize=14, fontweight='bold', color='darkred')
ax1.legend(fontsize=11, loc='upper left')
ax1.grid(True, alpha=0.3)
ax1.set_xlim(-1, 1)

# Add text annotations
ax1.text(0.02, 0.98, 'Current median: -0.03\nProposed median: 0.90',
         transform=ax1.transAxes, fontsize=11, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# ----------------------------------------------------------------------------
# Panel 2: Current vs Proposed Sigma Prior
# ----------------------------------------------------------------------------
ax2 = fig.add_subplot(gs[0, 1])

# Current prior: HalfNormal(0, 1)
sigma_current = np.abs(np.random.normal(0, 1, 10000))

# Proposed prior: HalfNormal(0, 0.5)
sigma_proposed = np.abs(np.random.normal(0, 0.5, 10000))

ax2.hist(sigma_current, bins=50, alpha=0.5, color='red', label='Current: HalfNormal(0, 1)',
         density=True, range=(0, 3))
ax2.hist(sigma_proposed, bins=50, alpha=0.5, color='green', label='Proposed: HalfNormal(0, 0.5)',
         density=True, range=(0, 3))
ax2.axvline(0.5, color='orange', linestyle='--', linewidth=2, label='Expected residual SD ≈ 0.3-0.5')
ax2.set_xlabel('sigma (residual SD on log-scale)', fontsize=13, fontweight='bold')
ax2.set_ylabel('Density', fontsize=13, fontweight='bold')
ax2.set_title('ISSUE 2: Sigma Prior Too Wide\nAllows extreme predictions',
              fontsize=14, fontweight='bold', color='darkred')
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3)
ax2.set_xlim(0, 3)

# Add text annotations
pct_current_large = np.mean(sigma_current > 1) * 100
pct_proposed_large = np.mean(sigma_proposed > 1) * 100
ax2.text(0.98, 0.98, f'Current: {pct_current_large:.1f}% > 1.0\nProposed: {pct_proposed_large:.1f}% > 1.0',
         transform=ax2.transAxes, fontsize=11, verticalalignment='top', horizontalalignment='right',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# ----------------------------------------------------------------------------
# Panel 3: Impact on Prior Predictive Distribution
# ----------------------------------------------------------------------------
ax3 = fig.add_subplot(gs[1, :])

# Simulate quick comparison (simplified, not full AR)
np.random.seed(123)
year_grid = np.linspace(-1.67, 1.67, 40)

# Current priors - sample a few extreme cases
n_sim = 200
colors_current = ['red'] * n_sim
alpha_current = 0.1

for i in range(n_sim):
    alpha_draw = np.random.normal(4.3, 0.5)
    beta1_draw = np.random.normal(0.86, 0.2)
    sigma_draw = np.abs(np.random.normal(0, 1))  # Current

    mu = alpha_draw + beta1_draw * year_grid
    log_C = np.random.normal(mu, sigma_draw)
    C = np.exp(log_C)

    if np.max(C) < 1000:  # Only plot reasonable ones
        ax3.plot(year_grid, C, color='red', alpha=0.1, linewidth=0.5)

# Proposed priors
colors_proposed = ['green'] * n_sim
alpha_proposed = 0.1

for i in range(n_sim):
    alpha_draw = np.random.normal(4.3, 0.5)
    beta1_draw = np.random.normal(0.86, 0.15)  # Tighter
    sigma_draw = np.abs(np.random.normal(0, 0.5))  # Proposed

    mu = alpha_draw + beta1_draw * year_grid
    log_C = np.random.normal(mu, sigma_draw)
    C = np.exp(log_C)

    ax3.plot(year_grid, C, color='green', alpha=0.1, linewidth=0.5)

# Observed data
data = pd.read_csv("/workspace/data/data.csv")
ax3.scatter(data['year'], data['C'], color='blue', s=80, zorder=10,
           label='Observed Data', edgecolors='black', linewidth=1)

ax3.set_xlabel('Year (standardized)', fontsize=13, fontweight='bold')
ax3.set_ylabel('Count (C)', fontsize=13, fontweight='bold')
ax3.set_title('IMPACT: Prior Predictive Distributions (Simplified)\nRed = Current priors (wide), Green = Proposed priors (tighter)',
              fontsize=14, fontweight='bold')
ax3.legend(fontsize=12, loc='upper left')
ax3.grid(True, alpha=0.3)
ax3.set_ylim(0, 600)

# Add text box
textstr = 'Current priors: Many extreme trajectories (>1000 not shown)\nProposed priors: More concentrated around observed data'
ax3.text(0.02, 0.98, textstr, transform=ax3.transAxes, fontsize=12,
         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

# ----------------------------------------------------------------------------
# Overall title
# ----------------------------------------------------------------------------
fig.suptitle('PRIOR PREDICTIVE CHECK SUMMARY: Critical Issues Identified',
             fontsize=18, fontweight='bold', y=0.995)

plt.savefig(PLOTS_DIR / "summary_critical_issues.png", dpi=150, bbox_inches='tight')
print("Summary plot created: summary_critical_issues.png")
