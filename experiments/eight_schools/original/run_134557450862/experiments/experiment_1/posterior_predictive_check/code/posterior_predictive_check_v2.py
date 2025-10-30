"""
Posterior Predictive Checks for Eight Schools Model
===================================================

Streamlined implementation for assessing model adequacy.
"""

import numpy as np
import pandas as pd
import arviz as az
import xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
import sys

# Set random seed
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Configure plotting
plt.style.use('default')
sns.set_palette("husl")

# Define paths
BASE_DIR = Path("/workspace/experiments/experiment_1/posterior_predictive_check")
INFERENCE_PATH = Path("/workspace/experiments/experiment_1/posterior_inference/diagnostics/posterior_inference.netcdf")
DATA_PATH = Path("/workspace/data/data.csv")
PLOTS_DIR = BASE_DIR / "plots"

print("="*80)
print("POSTERIOR PREDICTIVE CHECKS: Eight Schools Model")
print("="*80); sys.stdout.flush()

# Load data
print("\n[1/7] Loading data..."); sys.stdout.flush()
data = pd.read_csv(DATA_PATH)
J = len(data)
y_obs = data['y'].values
sigma_obs = data['sigma'].values
school_names = [f"School {i+1}" for i in range(J)]
print(data)

# Load posterior
print("\n[2/7] Loading posterior inference..."); sys.stdout.flush()
idata = az.from_netcdf(INFERENCE_PATH)
mu_samples = idata.posterior['mu'].values.flatten()
tau_samples = idata.posterior['tau'].values.flatten()
theta_samples = idata.posterior['theta'].values.reshape(-1, J)
n_samples = len(mu_samples)
print(f"Loaded {n_samples} posterior samples")
print(f"  mu: {mu_samples.mean():.2f} ± {mu_samples.std():.2f}")
print(f"  tau: {tau_samples.mean():.2f} ± {tau_samples.std():.2f}")

# Generate posterior predictive samples
print("\n[3/7] Generating posterior predictive samples..."); sys.stdout.flush()
y_rep = np.zeros((n_samples, J))
for i in range(n_samples):
    for j in range(J):
        y_rep[i, j] = np.random.normal(theta_samples[i, j], sigma_obs[j])
print(f"Generated {n_samples} replicated datasets")

# Add to InferenceData
print("\n[4/7] Adding to InferenceData..."); sys.stdout.flush()
if 'posterior_predictive' not in idata.groups():
    pp_dataset = xr.Dataset({
        'y': xr.DataArray(
            y_rep.reshape(4, n_samples//4, J),
            dims=['chain', 'draw', 'y_dim_0'],
            coords={'chain': idata.posterior.chain, 'draw': idata.posterior.draw, 'y_dim_0': range(J)}
        )
    })
    idata.add_groups(posterior_predictive=pp_dataset)
print("Done")

# Compute statistics
print("\n[5/7] Computing test statistics..."); sys.stdout.flush()
y_pred_means = y_rep.mean(axis=0)
y_pred_95 = np.percentile(y_rep, [2.5, 97.5], axis=0)

# Coverage
in_interval = [(y_obs[j] >= y_pred_95[0, j]) and (y_obs[j] <= y_pred_95[1, j]) for j in range(J)]
coverage_rate = np.mean(in_interval)

# Test statistics
T_obs_sd = np.std(y_obs, ddof=1)
T_obs_range = y_obs.max() - y_obs.min()
T_obs_max = y_obs.max()

T_rep_sd = np.array([np.std(y_rep[i], ddof=1) for i in range(n_samples)])
T_rep_range = np.array([y_rep[i].max() - y_rep[i].min() for i in range(n_samples)])
T_rep_max = y_rep.max(axis=1)

p_value_sd = np.mean(T_rep_sd >= T_obs_sd)
p_value_range = np.mean(T_rep_range >= T_obs_range)
p_value_max = np.mean(T_rep_max >= T_obs_max)

print(f"\nCoverage: {100*coverage_rate:.1f}% in 95% intervals")
print(f"Bayesian p-values:")
print(f"  SD: {p_value_sd:.3f}")
print(f"  Range: {p_value_range:.3f}")
print(f"  Max: {p_value_max:.3f}")

# LOO-PIT
print("\n[6/7] Computing LOO-PIT..."); sys.stdout.flush()
loo_pit = az.loo_pit(idata=idata, y='y')
ks_stat, ks_pval = stats.kstest(loo_pit, 'uniform')
print(f"LOO-PIT KS test p-value: {ks_pval:.3f}")

# Residuals
theta_mean = theta_samples.mean(axis=0)
residuals = y_obs - theta_mean
precision = 1 / sigma_obs
corr, corr_pval = stats.pearsonr(precision, np.abs(residuals))
print(f"Residual-precision correlation p-value: {corr_pval:.3f}")

# Create plots
print("\n[7/7] Creating visualizations..."); sys.stdout.flush()

# Plot 1: Individual school densities
print("  - Individual school densities..."); sys.stdout.flush()
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.flatten()
for j in range(J):
    ax = axes[j]
    ax.hist(y_rep[:, j], bins=40, density=True, alpha=0.6, color='steelblue', edgecolor='black')
    ax.axvline(y_obs[j], color='red', linewidth=3, linestyle='--', label=f'Observed={y_obs[j]}')
    ax.axvspan(y_pred_95[0, j], y_pred_95[1, j], alpha=0.2, color='green', label='95% CI')
    ax.set_title(school_names[j], fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
plt.suptitle('Posterior Predictive Check: Individual Schools', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(PLOTS_DIR / "ppc_density_overlay.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"    Saved: ppc_density_overlay.png")

# Plot 2: School-by-school intervals
print("  - School-by-school intervals..."); sys.stdout.flush()
fig, ax = plt.subplots(figsize=(12, 8))
for j in range(J):
    ax.plot([y_pred_95[0, j], y_pred_95[1, j]], [j, j], color='lightblue', linewidth=8)
    ax.plot(y_pred_means[j], j, 'o', markersize=10, color='darkblue')
    color = 'green' if in_interval[j] else 'red'
    marker = 'D' if in_interval[j] else 'X'
    ax.errorbar(y_obs[j], j, xerr=sigma_obs[j]*1.96, fmt=marker, markersize=10,
                color=color, capsize=5, linewidth=2)
ax.set_yticks(range(J))
ax.set_yticklabels(school_names)
ax.set_xlabel('Effect Size', fontweight='bold')
ax.set_title(f'Posterior Predictive Check: Coverage\n{100*coverage_rate:.1f}% in 95% intervals',
             fontsize=14, fontweight='bold')
ax.grid(alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig(PLOTS_DIR / "ppc_individual_schools.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"    Saved: ppc_individual_schools.png")

# Plot 3: ArviZ PPC
print("  - ArviZ PPC overlay..."); sys.stdout.flush()
fig, ax = plt.subplots(figsize=(12, 6))
az.plot_ppc(idata, num_pp_samples=100, random_seed=RANDOM_SEED, ax=ax)
ax.set_title('Posterior Predictive Check: Overall Density', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(PLOTS_DIR / "ppc_arviz_overlay.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"    Saved: ppc_arviz_overlay.png")

# Plot 4: Test statistics
print("  - Test statistics..."); sys.stdout.flush()
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

ax = axes[0]
ax.hist(T_rep_sd, bins=40, density=True, alpha=0.6, color='steelblue', edgecolor='black')
ax.axvline(T_obs_sd, color='red', linewidth=3, linestyle='--')
ax.set_xlabel('SD(y)')
ax.set_title(f'SD Test\n(p={p_value_sd:.3f})', fontweight='bold')
ax.grid(alpha=0.3)

ax = axes[1]
ax.hist(T_rep_range, bins=40, density=True, alpha=0.6, color='steelblue', edgecolor='black')
ax.axvline(T_obs_range, color='red', linewidth=3, linestyle='--')
ax.set_xlabel('Range(y)')
ax.set_title(f'Range Test\n(p={p_value_range:.3f})', fontweight='bold')
ax.grid(alpha=0.3)

ax = axes[2]
ax.hist(T_rep_max, bins=40, density=True, alpha=0.6, color='steelblue', edgecolor='black')
ax.axvline(T_obs_max, color='red', linewidth=3, linestyle='--')
ax.set_xlabel('Max(y)')
ax.set_title(f'Max Test\n(p={p_value_max:.3f})', fontweight='bold')
ax.grid(alpha=0.3)

plt.suptitle('Test Statistics', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(PLOTS_DIR / "ppc_test_statistics.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"    Saved: ppc_test_statistics.png")

# Plot 5: LOO-PIT
print("  - LOO-PIT calibration..."); sys.stdout.flush()
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
az.plot_loo_pit(idata=idata, y='y', ax=ax, legend=False)
ax.set_title(f'LOO-PIT Histogram\n(KS p-value={ks_pval:.3f})', fontweight='bold')

ax = axes[1]
sorted_pit = np.sort(loo_pit)
theoretical = np.linspace(0, 1, len(sorted_pit))
ax.plot(theoretical, sorted_pit, 'o', markersize=10, color='steelblue')
ax.plot([0, 1], [0, 1], 'r--', linewidth=2)
ax.set_xlabel('Theoretical Quantiles')
ax.set_ylabel('Observed Quantiles')
ax.set_title('LOO-PIT Q-Q Plot', fontweight='bold')
ax.grid(alpha=0.3)

plt.suptitle('Calibration Check', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(PLOTS_DIR / "loo_pit.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"    Saved: loo_pit.png")

# Plot 6: Residual diagnostics
print("  - Residual diagnostics..."); sys.stdout.flush()
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

ax = axes[0]
std_resid = residuals / sigma_obs
ax.bar(range(J), std_resid, color='steelblue', edgecolor='black')
ax.axhline(0, color='black', linewidth=1)
ax.axhline(2, color='red', linewidth=1, linestyle='--', alpha=0.5)
ax.axhline(-2, color='red', linewidth=1, linestyle='--', alpha=0.5)
ax.set_xticks(range(J))
ax.set_xticklabels(school_names, rotation=45, ha='right')
ax.set_ylabel('Standardized Residual')
ax.set_title('Standardized Residuals', fontweight='bold')
ax.grid(alpha=0.3, axis='y')

ax = axes[1]
ax.scatter(precision, np.abs(residuals), s=100, color='steelblue', edgecolor='black')
for j in range(J):
    ax.text(precision[j], np.abs(residuals[j]) + 0.5, school_names[j], fontsize=9, ha='center')
ax.set_xlabel('Precision (1/σ)')
ax.set_ylabel('|Residual|')
ax.set_title(f'Precision vs |Residual|\n(r={corr:.3f}, p={corr_pval:.3f})', fontweight='bold')
ax.grid(alpha=0.3)

ax = axes[2]
ax.scatter(y_obs, residuals, s=100, color='steelblue', edgecolor='black')
for j in range(J):
    ax.text(y_obs[j], residuals[j] + 0.5, school_names[j], fontsize=9, ha='center')
ax.axhline(0, color='red', linewidth=2, linestyle='--', alpha=0.5)
ax.set_xlabel('Observed Effect')
ax.set_ylabel('Residual')
ax.set_title('Residuals vs Observed', fontweight='bold')
ax.grid(alpha=0.3)

plt.suptitle('Residual Diagnostics', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(PLOTS_DIR / "residual_diagnostics.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"    Saved: residual_diagnostics.png")

# Overall assessment
print("\n" + "="*80)
print("OVERALL ASSESSMENT")
print("="*80)

checks = [
    ("Coverage rate", coverage_rate >= 0.85, f"{100*coverage_rate:.1f}% (target: >85%)"),
    ("Test stat (SD)", 0.05 <= p_value_sd <= 0.95, f"p={p_value_sd:.3f}"),
    ("Test stat (Range)", 0.05 <= p_value_range <= 0.95, f"p={p_value_range:.3f}"),
    ("Test stat (Max)", 0.05 <= p_value_max <= 0.95, f"p={p_value_max:.3f}"),
    ("LOO-PIT uniform", ks_pval > 0.05, f"KS p={ks_pval:.3f}"),
    ("Residual correlation", corr_pval > 0.05, f"p={corr_pval:.3f}"),
]

for name, passed, msg in checks:
    status = "PASS" if passed else "FAIL"
    print(f"  [{status}] {name}: {msg}")

if coverage_rate >= 0.85 and all(0.05 <= p <= 0.95 for p in [p_value_sd, p_value_range]) and ks_pval > 0.05:
    decision = "PASS"
    print("\nDECISION: PASS - Model adequately captures observed data")
elif coverage_rate >= 0.70:
    decision = "CONCERN"
    print("\nDECISION: CONCERN - Some issues but not severe")
else:
    decision = "FAIL"
    print("\nDECISION: FAIL - Model does not adequately capture data")

print("="*80)
print("\nPOSTERIOR PREDICTIVE CHECKS COMPLETE")
print(f"Plots saved to: {PLOTS_DIR}/")
