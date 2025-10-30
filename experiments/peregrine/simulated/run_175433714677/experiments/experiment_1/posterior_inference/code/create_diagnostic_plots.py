"""
Create comprehensive diagnostic plots for posterior inference
"""

import sys
sys.path.insert(0, '/tmp/agent-home/.local/lib/python3.13/site-packages')

import arviz as az
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100

# Paths
BASE_DIR = Path("/workspace/experiments/experiment_1/posterior_inference")
DIAGNOSTICS_DIR = BASE_DIR / "diagnostics"
PLOTS_DIR = BASE_DIR / "plots"

print("Loading InferenceData...")
idata = az.from_netcdf(DIAGNOSTICS_DIR / "posterior_inference.netcdf")

print("\nCreating diagnostic plots...")

# Plot 1: Comprehensive trace plots
print("  1. Trace plots...")
fig, axes_array = plt.subplots(3, 2, figsize=(14, 10))
az.plot_trace(idata, var_names=['beta_0', 'beta_1', 'phi'], axes=axes_array)
plt.suptitle("Trace Plots: Convergence Diagnostics", fontsize=14, y=0.995)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "trace_plots.png", dpi=150, bbox_inches='tight')
plt.close()

# Plot 2: Rank plots
print("  2. Rank plots...")
fig = az.plot_rank(idata, var_names=['beta_0', 'beta_1', 'phi'])
plt.suptitle("Rank Plots: Chain Mixing Diagnostics", fontsize=14, y=0.995)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "rank_plots.png", dpi=150, bbox_inches='tight')
plt.close()

# Plot 3: Posterior distributions with HDI
print("  3. Posterior distributions...")
fig = az.plot_posterior(idata, var_names=['beta_0', 'beta_1', 'phi'], hdi_prob=0.9)
plt.suptitle("Posterior Distributions (90% HDI)", fontsize=14, y=1.0)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "posterior_distributions.png", dpi=150, bbox_inches='tight')
plt.close()

# Plot 4: Pair plot
print("  4. Pair plot...")
fig = az.plot_pair(
    idata,
    var_names=['beta_0', 'beta_1', 'phi'],
    kind='hexbin',
    marginals=True,
    figsize=(10, 10)
)
plt.suptitle("Posterior Correlations (Bivariate)", fontsize=14, y=0.995)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "pair_plot.png", dpi=150, bbox_inches='tight')
plt.close()

# Plot 5: ESS diagnostics
print("  5. ESS evolution...")
fig = az.plot_ess(idata, var_names=['beta_0', 'beta_1', 'phi'], kind='evolution')
plt.suptitle("Effective Sample Size Evolution", fontsize=14, y=0.995)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "ess_evolution.png", dpi=150, bbox_inches='tight')
plt.close()

# Plot 6: Energy plot
print("  6. Energy plot...")
fig = az.plot_energy(idata)
plt.suptitle("Energy Diagnostic (BFMI Check)", fontsize=14, y=0.995)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "energy_plot.png", dpi=150, bbox_inches='tight')
plt.close()

# Plot 7: Autocorrelation
print("  7. Autocorrelation...")
fig, axes_array = plt.subplots(1, 3, figsize=(15, 4))
for idx, var in enumerate(['beta_0', 'beta_1', 'phi']):
    az.plot_autocorr(idata, var_names=[var], ax=axes_array[idx], combined=True)
    axes_array[idx].set_title(f'{var}')
plt.suptitle("Autocorrelation Diagnostics", fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "autocorrelation.png", dpi=150, bbox_inches='tight')
plt.close()

# Plot 8: Convergence overview (compact)
print("  8. Convergence overview...")
fig, axes_array = plt.subplots(2, 2, figsize=(12, 10))

# R-hat
summary = az.summary(idata, var_names=['beta_0', 'beta_1', 'phi'])
params = summary.index.tolist()
rhats = summary['r_hat'].values

axes_array[0, 0].barh(params, rhats, color='steelblue')
axes_array[0, 0].axvline(1.01, color='red', linestyle='--', label='Threshold (1.01)')
axes_array[0, 0].set_xlabel('R-hat')
axes_array[0, 0].set_title('R-hat Values (Convergence)')
axes_array[0, 0].legend()
axes_array[0, 0].grid(axis='x', alpha=0.3)

# ESS bulk
ess_bulk = summary['ess_bulk'].values
axes_array[0, 1].barh(params, ess_bulk, color='forestgreen')
axes_array[0, 1].axvline(400, color='red', linestyle='--', label='Threshold (400)')
axes_array[0, 1].set_xlabel('ESS Bulk')
axes_array[0, 1].set_title('Effective Sample Size (Bulk)')
axes_array[0, 1].legend()
axes_array[0, 1].grid(axis='x', alpha=0.3)

# ESS tail
ess_tail = summary['ess_tail'].values
axes_array[1, 0].barh(params, ess_tail, color='darkorange')
axes_array[1, 0].axvline(400, color='red', linestyle='--', label='Threshold (400)')
axes_array[1, 0].set_xlabel('ESS Tail')
axes_array[1, 0].set_title('Effective Sample Size (Tail)')
axes_array[1, 0].legend()
axes_array[1, 0].grid(axis='x', alpha=0.3)

# MCSE
mcse = summary['mcse_mean'].values / summary['sd'].values
axes_array[1, 1].barh(params, mcse * 100, color='purple')
axes_array[1, 1].axvline(5, color='red', linestyle='--', label='Threshold (5%)')
axes_array[1, 1].set_xlabel('MCSE / SD (%)')
axes_array[1, 1].set_title('Monte Carlo Standard Error')
axes_array[1, 1].legend()
axes_array[1, 1].grid(axis='x', alpha=0.3)

plt.suptitle("Convergence Metrics Overview", fontsize=14, y=0.995)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "convergence_overview.png", dpi=150, bbox_inches='tight')
plt.close()

print("\nAll plots saved to:", PLOTS_DIR)
print("\nPlot inventory:")
print("  - trace_plots.png: Chain traces and marginal posteriors")
print("  - rank_plots.png: Rank diagnostics for mixing")
print("  - posterior_distributions.png: Posterior densities with 90% HDI")
print("  - pair_plot.png: Bivariate correlations")
print("  - ess_evolution.png: ESS as function of iteration")
print("  - energy_plot.png: Energy diagnostic (HMC quality)")
print("  - autocorrelation.png: Within-chain autocorrelation")
print("  - convergence_overview.png: Summary of all convergence metrics")
