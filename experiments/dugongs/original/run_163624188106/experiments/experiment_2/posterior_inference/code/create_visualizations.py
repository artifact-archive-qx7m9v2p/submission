"""
Create diagnostic visualizations for Log-Linear Heteroscedastic Model (Experiment 2)
"""

import pandas as pd
import numpy as np
import arviz as az
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats as sp_stats
import warnings
warnings.filterwarnings('ignore')

# Paths
data_path = "/workspace/data/data.csv"
idata_path = "/workspace/experiments/experiment_2/posterior_inference/diagnostics/posterior_inference.netcdf"
plots_dir = "/workspace/experiments/experiment_2/posterior_inference/plots"

print("=" * 80)
print("CREATING VISUALIZATIONS FOR EXPERIMENT 2")
print("=" * 80)

# Load data and inference
print("\n1. Loading data and inference...")
df = pd.read_csv(data_path)
idata = az.from_netcdf(idata_path)

x = df['x'].values
y = df['Y'].values
N = len(df)

print(f"   Loaded {N} observations")
print(f"   InferenceData groups: {list(idata.groups())}")

# Extract posterior samples
beta_0 = idata.posterior['beta_0'].values.flatten()
beta_1 = idata.posterior['beta_1'].values.flatten()
gamma_0 = idata.posterior['gamma_0'].values.flatten()
gamma_1 = idata.posterior['gamma_1'].values.flatten()

print("\n2. Creating convergence diagnostics...")

# Figure 1: Convergence diagnostics (trace + rank plots)
fig = plt.figure(figsize=(14, 10))
gs = gridspec.GridSpec(4, 2, figure=fig, hspace=0.4)

params = ['beta_0', 'beta_1', 'gamma_0', 'gamma_1']
param_labels = [r'$\beta_0$', r'$\beta_1$', r'$\gamma_0$', r'$\gamma_1$']

for i, (param, label) in enumerate(zip(params, param_labels)):
    # Trace plot
    ax_trace = fig.add_subplot(gs[i, 0])
    for chain in range(4):
        chain_data = idata.posterior[param].sel(chain=chain).values
        ax_trace.plot(chain_data, alpha=0.7, lw=0.5, label=f'Chain {chain+1}')
    ax_trace.set_ylabel(label, fontsize=11)
    ax_trace.set_xlabel('Iteration', fontsize=10)
    if i == 0:
        ax_trace.legend(loc='upper right', fontsize=8, ncol=4)
    ax_trace.grid(alpha=0.3)

    # Rank plot
    ax_rank = fig.add_subplot(gs[i, 1])
    az.plot_rank(idata, var_names=[param], ax=ax_rank)
    ax_rank.set_title(f'{label} Rank Plot', fontsize=10)

fig.suptitle('Convergence Diagnostics: Trace and Rank Plots', fontsize=14, fontweight='bold')
plt.savefig(f"{plots_dir}/convergence_diagnostics.png", dpi=150, bbox_inches='tight')
print(f"   Saved: convergence_diagnostics.png")
plt.close()

print("\n3. Creating posterior distributions...")

# Figure 2: Posterior distributions with priors
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

# Prior functions
prior_beta_0 = lambda x: sp_stats.norm.pdf(x, loc=1.8, scale=0.5)
prior_beta_1 = lambda x: sp_stats.norm.pdf(x, loc=0.3, scale=0.2)
prior_gamma_0 = lambda x: sp_stats.norm.pdf(x, loc=-2, scale=1)
prior_gamma_1 = lambda x: sp_stats.norm.pdf(x, loc=-0.05, scale=0.05)

posteriors = [beta_0, beta_1, gamma_0, gamma_1]
priors = [prior_beta_0, prior_beta_1, prior_gamma_0, prior_gamma_1]

for i, (post, prior, label) in enumerate(zip(posteriors, priors, param_labels)):
    ax = axes[i]

    # Posterior
    ax.hist(post, bins=50, density=True, alpha=0.6, color='steelblue', label='Posterior')

    # Prior
    x_range = np.linspace(post.min(), post.max(), 200)
    ax.plot(x_range, prior(x_range), 'r--', lw=2, label='Prior')

    # Mean and CI
    post_mean = post.mean()
    post_lower = np.percentile(post, 2.5)
    post_upper = np.percentile(post, 97.5)

    ax.axvline(post_mean, color='darkblue', linestyle='-', lw=2, label=f'Mean: {post_mean:.3f}')
    ax.axvline(post_lower, color='darkblue', linestyle=':', lw=1.5, alpha=0.7)
    ax.axvline(post_upper, color='darkblue', linestyle=':', lw=1.5, alpha=0.7)

    # Special annotation for gamma_1
    if label == r'$\gamma_1$':
        ax.axvline(0, color='red', linestyle='--', lw=2, alpha=0.7, label='Zero')
        prob_neg = np.mean(post < 0)
        ax.text(0.05, 0.95, f'P({label} < 0) = {prob_neg:.1%}',
                transform=ax.transAxes, fontsize=10, va='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax.set_xlabel(label, fontsize=12)
    ax.set_ylabel('Density', fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

fig.suptitle('Posterior Distributions vs Priors', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{plots_dir}/posterior_distributions.png", dpi=150, bbox_inches='tight')
print(f"   Saved: posterior_distributions.png")
plt.close()

print("\n4. Creating model fit visualization...")

# Figure 3: Model fit with heteroscedastic credible intervals
fig, ax = plt.subplots(figsize=(12, 8))

# Generate prediction grid
x_pred = np.linspace(x.min(), x.max(), 100)

# Compute posterior predictions
n_samples = 500
sample_indices = np.random.choice(len(beta_0), n_samples, replace=False)

mu_samples = np.zeros((n_samples, len(x_pred)))
sigma_samples = np.zeros((n_samples, len(x_pred)))

for i, idx in enumerate(sample_indices):
    mu_samples[i, :] = beta_0[idx] + beta_1[idx] * np.log(x_pred)
    log_sigma = gamma_0[idx] + gamma_1[idx] * x_pred
    sigma_samples[i, :] = np.exp(log_sigma)

# Mean predictions
mu_mean = mu_samples.mean(axis=0)
sigma_mean = sigma_samples.mean(axis=0)

# Credible intervals for mu
mu_lower = np.percentile(mu_samples, 2.5, axis=0)
mu_upper = np.percentile(mu_samples, 97.5, axis=0)

# Predictive intervals (mu ± 2*sigma)
pred_lower = mu_samples - 2 * sigma_samples
pred_upper = mu_samples + 2 * sigma_samples
pred_lower_mean = pred_lower.mean(axis=0)
pred_upper_mean = pred_upper.mean(axis=0)

# Plot
ax.scatter(x, y, color='black', s=60, alpha=0.7, zorder=3, label='Observed data')
ax.plot(x_pred, mu_mean, 'b-', lw=2, label='Mean prediction')
ax.fill_between(x_pred, mu_lower, mu_upper, alpha=0.3, color='blue', label='95% CI (mean)')
ax.fill_between(x_pred, pred_lower_mean, pred_upper_mean, alpha=0.15, color='lightblue',
                label='95% predictive interval')

ax.set_xlabel('x', fontsize=12)
ax.set_ylabel('Y', fontsize=12)
ax.set_title('Model Fit: Heteroscedastic Variance (gamma_1 ~ 0, evidence lacking)', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(f"{plots_dir}/model_fit.png", dpi=150, bbox_inches='tight')
print(f"   Saved: model_fit.png")
plt.close()

print("\n5. Creating residual diagnostics...")

# Figure 4: Residual diagnostics
fig = plt.figure(figsize=(14, 10))
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)

# Compute residuals using posterior mean
beta_0_mean = beta_0.mean()
beta_1_mean = beta_1.mean()
gamma_0_mean = gamma_0.mean()
gamma_1_mean = gamma_1.mean()

mu_fit = beta_0_mean + beta_1_mean * np.log(x)
log_sigma_fit = gamma_0_mean + gamma_1_mean * x
sigma_fit = np.exp(log_sigma_fit)
residuals = y - mu_fit
std_residuals = residuals / sigma_fit

# 1. Residuals vs x
ax1 = fig.add_subplot(gs[0, 0])
ax1.scatter(x, residuals, color='darkblue', s=60, alpha=0.7)
ax1.axhline(0, color='red', linestyle='--', lw=2)
ax1.set_xlabel('x', fontsize=11)
ax1.set_ylabel('Residuals', fontsize=11)
ax1.set_title('Residuals vs x', fontsize=12)
ax1.grid(alpha=0.3)

# 2. Standardized residuals vs x
ax2 = fig.add_subplot(gs[0, 1])
ax2.scatter(x, std_residuals, color='darkgreen', s=60, alpha=0.7)
ax2.axhline(0, color='red', linestyle='--', lw=2)
ax2.axhline(2, color='orange', linestyle=':', lw=1.5, alpha=0.7)
ax2.axhline(-2, color='orange', linestyle=':', lw=1.5, alpha=0.7)
ax2.set_xlabel('x', fontsize=11)
ax2.set_ylabel('Standardized Residuals', fontsize=11)
ax2.set_title('Standardized Residuals vs x', fontsize=12)
ax2.grid(alpha=0.3)

# 3. QQ plot
ax3 = fig.add_subplot(gs[1, 0])
sp_stats.probplot(std_residuals, dist="norm", plot=ax3)
ax3.set_title('Q-Q Plot (Standardized Residuals)', fontsize=12)
ax3.grid(alpha=0.3)

# 4. Residuals vs fitted
ax4 = fig.add_subplot(gs[1, 1])
ax4.scatter(mu_fit, residuals, color='purple', s=60, alpha=0.7)
ax4.axhline(0, color='red', linestyle='--', lw=2)
ax4.set_xlabel('Fitted values', fontsize=11)
ax4.set_ylabel('Residuals', fontsize=11)
ax4.set_title('Residuals vs Fitted', fontsize=12)
ax4.grid(alpha=0.3)

fig.suptitle('Residual Diagnostics', fontsize=14, fontweight='bold')
plt.savefig(f"{plots_dir}/residual_diagnostics.png", dpi=150, bbox_inches='tight')
print(f"   Saved: residual_diagnostics.png")
plt.close()

print("\n6. Creating variance function visualization...")

# Figure 5: Variance as function of x
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Left: Posterior variance function
ax1 = axes[0]
for idx in sample_indices[:100]:
    log_sigma_sample = gamma_0[idx] + gamma_1[idx] * x_pred
    sigma_sample = np.exp(log_sigma_sample)
    ax1.plot(x_pred, sigma_sample**2, color='steelblue', alpha=0.05, lw=0.5)

# Mean variance
log_sigma_mean = gamma_0_mean + gamma_1_mean * x_pred
sigma_mean_pred = np.exp(log_sigma_mean)
ax1.plot(x_pred, sigma_mean_pred**2, 'b-', lw=3, label='Posterior mean')

# Observed variance proxy (absolute residuals)
ax1.scatter(x, residuals**2, color='red', s=60, alpha=0.6, label='Squared residuals', zorder=3)

ax1.set_xlabel('x', fontsize=12)
ax1.set_ylabel(r'$\sigma^2$(x)', fontsize=12)
ax1.set_title('Variance Function', fontsize=13)
ax1.legend(fontsize=10)
ax1.grid(alpha=0.3)

# Right: Comparison with constant variance (Model 1)
ax2 = axes[1]

# Model 2 variance
ax2.plot(x_pred, sigma_mean_pred**2, 'b-', lw=2, label='Model 2 (heteroscedastic)')

# Model 1 constant variance (estimate from residuals)
model1_var = residuals.var()
ax2.axhline(model1_var, color='green', linestyle='--', lw=2, label='Model 1 (homoscedastic)')

# Observed squared residuals
ax2.scatter(x, residuals**2, color='red', s=60, alpha=0.6, label='Squared residuals', zorder=3)

ax2.set_xlabel('x', fontsize=12)
ax2.set_ylabel(r'$\sigma^2$', fontsize=12)
ax2.set_title('Model Comparison: Variance Structure', fontsize=13)
ax2.legend(fontsize=10)
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(f"{plots_dir}/variance_function.png", dpi=150, bbox_inches='tight')
print(f"   Saved: variance_function.png")
plt.close()

print("\n7. Creating model comparison summary...")

# Figure 6: Model comparison summary
fig = plt.figure(figsize=(12, 8))
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.4)

# Load LOO results
import json
loo2_path = "/workspace/experiments/experiment_2/posterior_inference/diagnostics/loo_results.json"
loo1_path = "/workspace/experiments/experiment_1/posterior_inference/diagnostics/loo_results.json"

with open(loo2_path, 'r') as f:
    loo2 = json.load(f)
with open(loo1_path, 'r') as f:
    loo1 = json.load(f)

# 1. ELPD comparison
ax1 = fig.add_subplot(gs[0, 0])
models = ['Model 1\n(Homoscedastic)', 'Model 2\n(Heteroscedastic)']
elpds = [loo1['elpd_loo'], loo2['elpd_loo']]
ses = [loo1['se'], loo2['se']]

bars = ax1.bar(models, elpds, yerr=ses, capsize=10, color=['green', 'steelblue'], alpha=0.7, edgecolor='black')
ax1.set_ylabel('ELPD LOO', fontsize=12)
ax1.set_title('Model Comparison: ELPD LOO', fontsize=13, fontweight='bold')
ax1.axhline(0, color='black', linestyle='-', lw=0.8)
ax1.grid(axis='y', alpha=0.3)

# Add values
for bar, elpd, se in zip(bars, elpds, ses):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
             f'{elpd:.1f}±{se:.1f}',
             ha='center', va='bottom' if height > 0 else 'top', fontsize=10, fontweight='bold')

# 2. Pareto k comparison
ax2 = fig.add_subplot(gs[0, 1])
k_categories = ['Good\n(k<0.5)', 'OK\n(0.5≤k<0.7)', 'Bad\n(0.7≤k<1)', 'Very Bad\n(k≥1)']
model1_k = [loo1['pareto_k_stats']['good_k_lt_0.5'],
            loo1['pareto_k_stats']['ok_k_0.5_to_0.7'],
            loo1['pareto_k_stats']['bad_k_0.7_to_1.0'],
            loo1['pareto_k_stats']['very_bad_k_gte_1.0']]
model2_k = [loo2['pareto_k_stats']['good_k_lt_0.5'],
            loo2['pareto_k_stats']['ok_k_0.5_to_0.7'],
            loo2['pareto_k_stats']['bad_k_0.7_to_1.0'],
            loo2['pareto_k_stats']['very_bad_k_gte_1.0']]

x_pos = np.arange(len(k_categories))
width = 0.35

ax2.bar(x_pos - width/2, model1_k, width, label='Model 1', color='green', alpha=0.7, edgecolor='black')
ax2.bar(x_pos + width/2, model2_k, width, label='Model 2', color='steelblue', alpha=0.7, edgecolor='black')

ax2.set_ylabel('Count', fontsize=12)
ax2.set_title('Pareto k Diagnostics', fontsize=13, fontweight='bold')
ax2.set_xticks(x_pos)
ax2.set_xticklabels(k_categories, fontsize=9)
ax2.legend(fontsize=10)
ax2.grid(axis='y', alpha=0.3)

# 3. gamma_1 posterior (key parameter)
ax3 = fig.add_subplot(gs[1, 0])
ax3.hist(gamma_1, bins=50, density=True, alpha=0.6, color='steelblue', edgecolor='black')
ax3.axvline(0, color='red', linestyle='--', lw=2, label='Zero')
ax3.axvline(gamma_1.mean(), color='darkblue', linestyle='-', lw=2, label=f'Mean: {gamma_1.mean():.3f}')

prob_neg = np.mean(gamma_1 < 0)
ax3.text(0.05, 0.95, f'P(γ₁ < 0) = {prob_neg:.1%}\n95% CI: [{np.percentile(gamma_1, 2.5):.3f}, {np.percentile(gamma_1, 97.5):.3f}]',
         transform=ax3.transAxes, fontsize=10, va='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

ax3.set_xlabel(r'$\gamma_1$', fontsize=12)
ax3.set_ylabel('Density', fontsize=12)
ax3.set_title(r'Heteroscedasticity Parameter ($\gamma_1$)', fontsize=13, fontweight='bold')
ax3.legend(fontsize=10)
ax3.grid(alpha=0.3)

# 4. Text summary
ax4 = fig.add_subplot(gs[1, 1])
ax4.axis('off')

delta_elpd = loo2['elpd_loo'] - loo1['elpd_loo']
delta_se = np.sqrt(loo2['se']**2 + loo1['se']**2)

summary_text = f"""
MODEL COMPARISON SUMMARY

ELPD Difference:
  Δ ELPD = {delta_elpd:.2f} ± {delta_se:.2f}

Interpretation:
  {'Model 2 strongly preferred' if delta_elpd > 2*delta_se else
   'Model 2 weakly preferred' if delta_elpd > delta_se else
   'Models equivalent' if abs(delta_elpd) < delta_se else
   'Model 1 weakly preferred' if delta_elpd > -2*delta_se else
   '>>> Model 1 STRONGLY PREFERRED <<<'}

Heteroscedasticity Evidence:
  γ₁ = {gamma_1.mean():.4f} ± {gamma_1.std():.4f}
  P(γ₁ < 0) = {prob_neg:.1%}

  >>> INSUFFICIENT EVIDENCE <<<

Recommendation:
  Use Model 1 (simpler, better LOO)
  Added complexity NOT justified

Pareto k Issues:
  Model 1: {loo1['pareto_k_stats']['bad_k_0.7_to_1.0'] + loo1['pareto_k_stats']['very_bad_k_gte_1.0']} problematic
  Model 2: {loo2['pareto_k_stats']['bad_k_0.7_to_1.0'] + loo2['pareto_k_stats']['very_bad_k_gte_1.0']} problematic
"""

ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes, fontsize=11,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))

fig.suptitle('Model Comparison: Homoscedastic vs Heteroscedastic', fontsize=14, fontweight='bold')
plt.savefig(f"{plots_dir}/model_comparison.png", dpi=150, bbox_inches='tight')
print(f"   Saved: model_comparison.png")
plt.close()

print("\n" + "=" * 80)
print("VISUALIZATION COMPLETE")
print("=" * 80)
print(f"All plots saved to: {plots_dir}")
