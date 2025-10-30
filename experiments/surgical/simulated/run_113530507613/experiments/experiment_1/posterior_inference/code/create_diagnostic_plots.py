"""
Create comprehensive diagnostic plots for MCMC convergence
"""

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Set plot style
plt.style.use('default')

# Load inference data
print("Loading InferenceData...")
idata_path = "/workspace/experiments/experiment_1/posterior_inference/diagnostics/posterior_inference.netcdf"
idata = az.from_netcdf(idata_path)
print("✓ Loaded successfully\n")

# Create plots directory
plots_dir = "/workspace/experiments/experiment_1/posterior_inference/plots"

# 1. Convergence overview - trace plots for key parameters
print("Creating trace plots...")

# Plot mu
fig = plt.figure(figsize=(14, 4))
az.plot_trace(idata, var_names=['mu'], compact=False, figsize=(14, 4))
plt.suptitle('Trace Plot: Population Mean (mu)', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(f'{plots_dir}/trace_plot_mu.png', dpi=150, bbox_inches='tight')
plt.close()

# Plot tau
fig = plt.figure(figsize=(14, 4))
az.plot_trace(idata, var_names=['tau'], compact=False, figsize=(14, 4))
plt.suptitle('Trace Plot: Between-Group SD (tau)', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(f'{plots_dir}/trace_plot_tau.png', dpi=150, bbox_inches='tight')
plt.close()

# Plot selected theta parameters
fig = plt.figure(figsize=(14, 10))
az.plot_trace(idata, var_names=['theta'], coords={'theta_dim_0': [0, 1, 7]}, compact=False, figsize=(14, 10))
plt.suptitle('Trace Plots: Selected Group Parameters', fontsize=14, y=0.995)
plt.tight_layout()
plt.savefig(f'{plots_dir}/trace_plots_selected_groups.png', dpi=150, bbox_inches='tight')
print(f"  ✓ Saved trace plots")
plt.close()

# 2. Rank plots for all key parameters
print("Creating rank plots...")
fig = plt.figure(figsize=(12, 8))
az.plot_rank(idata, var_names=['mu', 'tau', 'theta_raw'], figsize=(12, 8))
plt.suptitle('Rank Plots - Chain Mixing Assessment', fontsize=16, y=0.995)
plt.tight_layout()
plt.savefig(f'{plots_dir}/rank_plots.png', dpi=150, bbox_inches='tight')
print(f"  ✓ Saved: rank_plots.png")
plt.close()

# 3. Create ESS summary plot manually
print("Creating ESS diagnostic plots...")
summary = az.summary(idata, var_names=['mu', 'tau', 'theta'])

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Bulk ESS
params = summary.index
ess_bulk = summary['ess_bulk'].values
axes[0].barh(range(len(params)), ess_bulk, color='steelblue', alpha=0.7)
axes[0].axvline(400, color='red', linestyle='--', linewidth=2, label='Target minimum (400)')
axes[0].set_yticks(range(len(params)))
axes[0].set_yticklabels(params, fontsize=8)
axes[0].set_xlabel('ESS (bulk)', fontsize=11)
axes[0].set_title('Bulk ESS (measures center of distribution)', fontsize=12)
axes[0].legend()
axes[0].grid(True, alpha=0.3, axis='x')

# Tail ESS
ess_tail = summary['ess_tail'].values
axes[1].barh(range(len(params)), ess_tail, color='coral', alpha=0.7)
axes[1].axvline(400, color='red', linestyle='--', linewidth=2, label='Target minimum (400)')
axes[1].set_yticks(range(len(params)))
axes[1].set_yticklabels(params, fontsize=8)
axes[1].set_xlabel('ESS (tail)', fontsize=11)
axes[1].set_title('Tail ESS (measures tails of distribution)', fontsize=12)
axes[1].legend()
axes[1].grid(True, alpha=0.3, axis='x')

plt.suptitle('Effective Sample Size Diagnostics', fontsize=16, y=1.02)
plt.tight_layout()
plt.savefig(f'{plots_dir}/ess_diagnostics.png', dpi=150, bbox_inches='tight')
print(f"  ✓ Saved: ess_diagnostics.png")
plt.close()

# 4. Pairs plot for hyperparameters (check for funnel)
print("Creating pairs plot...")
az.plot_pair(
    idata,
    var_names=['mu', 'tau'],
    kind='kde',
    marginals=True,
    figsize=(10, 10)
)
plt.suptitle('Hyperparameter Joint Distribution (Funnel Check)', fontsize=16, y=0.995)
plt.savefig(f'{plots_dir}/pairs_plot_hyperparameters.png', dpi=150, bbox_inches='tight')
print(f"  ✓ Saved: pairs_plot_hyperparameters.png")
plt.close()

# 5. Energy diagnostic
print("Creating energy diagnostic plot...")
fig = plt.figure(figsize=(10, 6))
az.plot_energy(idata)
plt.suptitle('Energy Diagnostic (BFMI Check)', fontsize=16, y=1.00)
plt.tight_layout()
plt.savefig(f'{plots_dir}/energy_diagnostic.png', dpi=150, bbox_inches='tight')
print(f"  ✓ Saved: energy_diagnostic.png")
plt.close()

# 6. Posterior distributions for all theta parameters
print("Creating posterior distributions for all groups...")
fig, axes = plt.subplots(4, 3, figsize=(15, 12))
axes = axes.flatten()

for i in range(12):
    az.plot_posterior(
        idata,
        var_names=['theta'],
        coords={'theta_dim_0': i},
        ax=axes[i],
        hdi_prob=0.94,
        point_estimate='mean'
    )
    axes[i].set_title(f'Group {i+1}: theta[{i}]', fontsize=10)
    axes[i].set_xlabel('Logit Success Rate', fontsize=9)

plt.suptitle('Group-Specific Posterior Distributions', fontsize=16, y=0.995)
plt.tight_layout()
plt.savefig(f'{plots_dir}/posterior_distributions_all_groups.png', dpi=150, bbox_inches='tight')
print(f"  ✓ Saved: posterior_distributions_all_groups.png")
plt.close()

# 7. Forest plot comparing all groups
print("Creating forest plot...")
fig = plt.figure(figsize=(10, 8))
az.plot_forest(
    idata,
    var_names=['theta'],
    combined=True,
    hdi_prob=0.94,
    figsize=(10, 8)
)
plt.title('Group-Specific Logit Rates with 94% HDI', fontsize=14)
plt.xlabel('Logit Success Rate (theta)', fontsize=12)
plt.tight_layout()
plt.savefig(f'{plots_dir}/forest_plot_groups.png', dpi=150, bbox_inches='tight')
print(f"  ✓ Saved: forest_plot_groups.png")
plt.close()

# 8. Probability scale comparison
print("Creating probability scale forest plot...")
fig = plt.figure(figsize=(10, 8))
az.plot_forest(
    idata,
    var_names=['p'],
    combined=True,
    hdi_prob=0.94,
    figsize=(10, 8)
)
plt.title('Group-Specific Success Probabilities with 94% HDI', fontsize=14)
plt.xlabel('Success Probability (p)', fontsize=12)
plt.tight_layout()
plt.savefig(f'{plots_dir}/forest_plot_probabilities.png', dpi=150, bbox_inches='tight')
print(f"  ✓ Saved: forest_plot_probabilities.png")
plt.close()

# 9. Autocorrelation plot for key parameters
print("Creating autocorrelation plots...")
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

az.plot_autocorr(idata, var_names=['mu'], ax=axes[0, 0], max_lag=100, combined=False)
axes[0, 0].set_title('Autocorrelation: mu', fontsize=12)

az.plot_autocorr(idata, var_names=['tau'], ax=axes[0, 1], max_lag=100, combined=False)
axes[0, 1].set_title('Autocorrelation: tau', fontsize=12)

az.plot_autocorr(idata, var_names=['theta'], coords={'theta_dim_0': 0}, ax=axes[1, 0], max_lag=100, combined=False)
axes[1, 0].set_title('Autocorrelation: theta[0]', fontsize=12)

az.plot_autocorr(idata, var_names=['theta'], coords={'theta_dim_0': 7}, ax=axes[1, 1], max_lag=100, combined=False)
axes[1, 1].set_title('Autocorrelation: theta[7]', fontsize=12)

plt.suptitle('Autocorrelation Diagnostics', fontsize=16, y=1.00)
plt.tight_layout()
plt.savefig(f'{plots_dir}/autocorrelation_diagnostics.png', dpi=150, bbox_inches='tight')
print(f"  ✓ Saved: autocorrelation_diagnostics.png")
plt.close()

# 10. Shrinkage visualization
print("Creating shrinkage comparison...")
# Load data
data_path = "/workspace/data/data.csv"
df = pd.read_csv(data_path)
n = df['n_trials'].values
r = df['r_successes'].values
obs_rates = r / n

# Get posterior means
summary = az.summary(idata, var_names=['p'])
post_means = summary['mean'].values

# Compute pooled estimate
pooled_logit = np.log((r.sum() + 1) / (n.sum() - r.sum() + 1))  # Add-1 smoothing
pooled_rate = 1 / (1 + np.exp(-pooled_logit))

fig, ax = plt.subplots(figsize=(12, 8))

# Plot observed vs posterior
groups = np.arange(1, 13)
ax.scatter(groups, obs_rates, s=100, alpha=0.6, label='Observed Rates', color='red', marker='o')
ax.scatter(groups, post_means, s=100, alpha=0.6, label='Posterior Mean (Hierarchical)', color='blue', marker='s')
ax.axhline(pooled_rate, color='green', linestyle='--', linewidth=2, label=f'Pooled Estimate ({pooled_rate:.3f})')

# Add arrows showing shrinkage
for i in range(12):
    ax.annotate('', xy=(i+1, post_means[i]), xytext=(i+1, obs_rates[i]),
                arrowprops=dict(arrowstyle='->', color='gray', alpha=0.5, lw=1.5))

ax.set_xlabel('Group', fontsize=12)
ax.set_ylabel('Success Rate', fontsize=12)
ax.set_title('Shrinkage: Observed Rates vs Posterior Estimates', fontsize=14)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xticks(groups)

plt.tight_layout()
plt.savefig(f'{plots_dir}/shrinkage_visualization.png', dpi=150, bbox_inches='tight')
print(f"  ✓ Saved: shrinkage_visualization.png")
plt.close()

print("\n✓ All diagnostic plots created successfully!")
print(f"  Location: {plots_dir}/")
