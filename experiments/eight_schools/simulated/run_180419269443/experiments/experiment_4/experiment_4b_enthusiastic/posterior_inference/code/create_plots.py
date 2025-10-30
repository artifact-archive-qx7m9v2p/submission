"""Create diagnostic plots for enthusiastic prior model"""

import numpy as np
import pandas as pd
import arviz as az
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 150

# Paths
exp_dir = Path('/workspace/experiments/experiment_4/experiment_4b_enthusiastic/posterior_inference')
diag_dir = exp_dir / 'diagnostics'
plot_dir = exp_dir / 'plots'

# Load InferenceData
idata = az.from_netcdf(diag_dir / 'posterior_inference.netcdf')

print("Creating diagnostic plots for Model 4b (Enthusiastic)...")

# 1. Convergence overview: trace + rank plots for mu
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# Trace plots
az.plot_trace(idata, var_names=['mu'], axes=axes[0, :], compact=False)
axes[0, 0].set_title('Enthusiastic Prior: mu Trace')
axes[0, 1].set_title('Enthusiastic Prior: mu Posterior')

# Rank plots
az.plot_rank(idata, var_names=['mu'], axes=axes[1, 0])
axes[1, 0].set_title('Rank Plot: mu')

# Summary stats
mu_samples = idata.posterior['mu'].values.flatten()
mu_mean = mu_samples.mean()
mu_sd = mu_samples.std()
mu_ci = np.percentile(mu_samples, [2.5, 97.5])

axes[1, 1].text(0.1, 0.7, 'Enthusiastic Prior Model\n' + '=' * 30, fontsize=12, family='monospace')
axes[1, 1].text(0.1, 0.5, f'mu: {mu_mean:.2f} ± {mu_sd:.2f}', fontsize=11, family='monospace')
axes[1, 1].text(0.1, 0.4, f'95% CI: [{mu_ci[0]:.2f}, {mu_ci[1]:.2f}]', fontsize=11, family='monospace')
axes[1, 1].text(0.1, 0.2, f'Prior: N(15, 15)', fontsize=10, family='monospace', style='italic')
axes[1, 1].axis('off')

plt.tight_layout()
plt.savefig(plot_dir / 'convergence_overview.png', dpi=150, bbox_inches='tight')
print(f"  Saved: convergence_overview.png")
plt.close()

# 2. Prior vs Posterior overlay
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# mu
x_mu = np.linspace(-15, 35, 300)
prior_mu = stats.norm.pdf(x_mu, 15, 15)
axes[0].plot(x_mu, prior_mu, 'r--', label='Prior: N(15, 15)', linewidth=2)
axes[0].hist(mu_samples, bins=50, density=True, alpha=0.6, label='Posterior', color='orange')
axes[0].axvline(15, color='r', linestyle=':', alpha=0.5, label='Prior mean')
axes[0].axvline(mu_mean, color='orange', linestyle=':', alpha=0.8, label='Posterior mean')
axes[0].set_xlabel('mu (population mean effect)')
axes[0].set_ylabel('Density')
axes[0].set_title('Enthusiastic Prior vs Posterior: mu')
axes[0].legend()
axes[0].grid(alpha=0.3)

# tau
tau_samples = idata.posterior['tau'].values.flatten()
x_tau = np.linspace(0, 15, 300)
prior_tau = stats.halfcauchy.pdf(x_tau, scale=10)
axes[1].plot(x_tau, prior_tau, 'r--', label='Prior: Half-Cauchy(0, 10)', linewidth=2)
axes[1].hist(tau_samples, bins=50, density=True, alpha=0.6, label='Posterior', color='orange')
axes[1].set_xlabel('tau (between-study SD)')
axes[1].set_ylabel('Density')
axes[1].set_title('Enthusiastic Prior vs Posterior: tau')
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(plot_dir / 'prior_posterior_overlay.png', dpi=150, bbox_inches='tight')
print(f"  Saved: prior_posterior_overlay.png")
plt.close()

# 3. Forest plot for theta
fig, ax = plt.subplots(figsize=(10, 6))
az.plot_forest(idata, var_names=['theta'], combined=True, ax=ax, colors='orange')
ax.set_title('Study-Specific Effects (Enthusiastic Prior Model)')
ax.set_xlabel('Treatment Effect')
plt.tight_layout()
plt.savefig(plot_dir / 'forest_plot.png', dpi=150, bbox_inches='tight')
print(f"  Saved: forest_plot.png")
plt.close()

print("\nAll plots created successfully!")
