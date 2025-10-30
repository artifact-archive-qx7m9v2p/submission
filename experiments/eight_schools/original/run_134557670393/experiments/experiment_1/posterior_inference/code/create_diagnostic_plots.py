"""
Create diagnostic plots for convergence assessment
"""

import sys
sys.path.insert(0, '/tmp/agent-home/.local/lib/python3.13/site-packages')

import arviz as az
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
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

print("Loading InferenceData...")
idata = az.from_netcdf(IDATA_PATH)

print("\n1. Creating trace plots...")
# Trace plots for main parameters
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
az.plot_trace(idata, var_names=['mu', 'tau'], axes=axes)
plt.tight_layout()
plt.savefig(PLOT_DIR / "trace_main_parameters.png", dpi=150, bbox_inches='tight')
print(f"   Saved: trace_main_parameters.png")
plt.close()

# Trace plots for theta (study-specific effects) - just create the figure without specifying axes
fig = plt.figure(figsize=(14, 12))
az.plot_trace(idata, var_names=['theta'])
plt.tight_layout()
plt.savefig(PLOT_DIR / "trace_theta_parameters.png", dpi=150, bbox_inches='tight')
print(f"   Saved: trace_theta_parameters.png")
plt.close()

print("\n2. Creating rank plots (for chain mixing)...")
# Rank plots for convergence assessment
fig = plt.figure(figsize=(12, 6))
az.plot_rank(idata, var_names=['mu', 'tau'])
plt.tight_layout()
plt.savefig(PLOT_DIR / "rank_plots_main.png", dpi=150, bbox_inches='tight')
print(f"   Saved: rank_plots_main.png")
plt.close()

print("\n3. Creating energy diagnostic plot...")
fig = plt.figure(figsize=(10, 5))
az.plot_energy(idata)
plt.tight_layout()
plt.savefig(PLOT_DIR / "energy_diagnostic.png", dpi=150, bbox_inches='tight')
print(f"   Saved: energy_diagnostic.png")
plt.close()

print("\n4. Creating autocorrelation plots...")
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
az.plot_autocorr(idata, var_names=['mu'], ax=axes[0])
axes[0].set_title('Autocorrelation: mu')
az.plot_autocorr(idata, var_names=['tau'], ax=axes[1])
axes[1].set_title('Autocorrelation: tau')
plt.tight_layout()
plt.savefig(PLOT_DIR / "autocorrelation.png", dpi=150, bbox_inches='tight')
print(f"   Saved: autocorrelation.png")
plt.close()

print("\n5. Creating pair plot (mu, tau)...")
fig = az.plot_pair(
    idata,
    var_names=['mu', 'tau'],
    kind='kde',
    marginals=True,
    figsize=(8, 8)
)
plt.savefig(PLOT_DIR / "pair_plot_mu_tau.png", dpi=150, bbox_inches='tight')
print(f"   Saved: pair_plot_mu_tau.png")
plt.close()

print("\n6. Creating forest plot (ESS and R-hat summary)...")
fig = plt.figure(figsize=(10, 8))
az.plot_forest(
    idata,
    var_names=['mu', 'tau', 'theta'],
    combined=True,
    figsize=(10, 8)
)
plt.tight_layout()
plt.savefig(PLOT_DIR / "forest_plot_all_parameters.png", dpi=150, bbox_inches='tight')
print(f"   Saved: forest_plot_all_parameters.png")
plt.close()

print("\n7. Creating convergence overview dashboard...")
# Comprehensive convergence dashboard
fig = plt.figure(figsize=(16, 10))

# Trace plots for mu and tau
ax1 = plt.subplot(3, 3, 1)
for chain in range(4):
    mu_chain = idata.posterior['mu'].sel(chain=chain).values
    ax1.plot(mu_chain, alpha=0.6, label=f'Chain {chain+1}')
ax1.set_title('Trace: mu')
ax1.set_xlabel('Iteration')
ax1.legend(fontsize=8)

ax2 = plt.subplot(3, 3, 2)
for chain in range(4):
    tau_chain = idata.posterior['tau'].sel(chain=chain).values
    ax2.plot(tau_chain, alpha=0.6, label=f'Chain {chain+1}')
ax2.set_title('Trace: tau')
ax2.set_xlabel('Iteration')
ax2.legend(fontsize=8)

# Posterior distributions
ax3 = plt.subplot(3, 3, 3)
az.plot_posterior(idata, var_names=['mu'], ax=ax3)
ax3.set_title('Posterior: mu')

ax4 = plt.subplot(3, 3, 4)
az.plot_posterior(idata, var_names=['tau'], ax=ax4)
ax4.set_title('Posterior: tau')

# Autocorrelation
ax5 = plt.subplot(3, 3, 5)
az.plot_autocorr(idata, var_names=['mu'], ax=ax5, max_lag=50)
ax5.set_title('Autocorr: mu')

ax6 = plt.subplot(3, 3, 6)
az.plot_autocorr(idata, var_names=['tau'], ax=ax6, max_lag=50)
ax6.set_title('Autocorr: tau')

# Joint distribution
ax7 = plt.subplot(3, 3, 7)
mu_flat = idata.posterior['mu'].values.flatten()
tau_flat = idata.posterior['tau'].values.flatten()
ax7.hexbin(mu_flat, tau_flat, gridsize=30, cmap='Blues')
ax7.set_xlabel('mu')
ax7.set_ylabel('tau')
ax7.set_title('Joint posterior (mu, tau)')

# R-hat and ESS
summary = az.summary(idata, var_names=['mu', 'tau', 'theta'])
ax8 = plt.subplot(3, 3, 8)
ax8.barh(range(len(summary)), summary['r_hat'].values)
ax8.axvline(1.01, color='red', linestyle='--', label='Target < 1.01')
ax8.set_yticks(range(len(summary)))
ax8.set_yticklabels(summary.index, fontsize=8)
ax8.set_xlabel('R-hat')
ax8.set_title('Convergence: R-hat')
ax8.legend()

ax9 = plt.subplot(3, 3, 9)
ax9.barh(range(len(summary)), summary['ess_bulk'].values)
ax9.axvline(400, color='red', linestyle='--', label='Target > 400')
ax9.set_yticks(range(len(summary)))
ax9.set_yticklabels(summary.index, fontsize=8)
ax9.set_xlabel('ESS bulk')
ax9.set_title('Effective Sample Size')
ax9.legend()

plt.tight_layout()
plt.savefig(PLOT_DIR / "convergence_overview.png", dpi=150, bbox_inches='tight')
print(f"   Saved: convergence_overview.png")
plt.close()

print("\nAll diagnostic plots created successfully!")
