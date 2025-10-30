"""
Comprehensive Posterior Predictive Checks for Bayesian Hierarchical Meta-Analysis

This script performs rigorous model validation by comparing observed data to
posterior predictive distributions, testing the critical falsification criterion:

    REJECT if >1 study falls outside 95% posterior predictive interval

Author: Claude (Model Validation Specialist)
Date: 2025-10-28
"""

import numpy as np
import pandas as pd
import arviz as az
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import json

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10

# File paths
DATA_PATH = "/workspace/data/data.csv"
IDATA_PATH = "/workspace/experiments/experiment_1/posterior_inference/diagnostics/posterior_inference.netcdf"
OUTPUT_DIR = "/workspace/experiments/experiment_1/posterior_predictive_check"
PLOTS_DIR = f"{OUTPUT_DIR}/plots"

print("="*80)
print("POSTERIOR PREDICTIVE CHECKS: Bayesian Hierarchical Meta-Analysis")
print("="*80)

# Load observed data
print("\n[1/6] Loading observed data...")
data = pd.read_csv(DATA_PATH)
y_obs = data['y'].values
sigma = data['sigma'].values
n_studies = len(y_obs)
print(f"   - {n_studies} studies loaded")
print(f"   - Observed y: {y_obs}")
print(f"   - Standard errors: {sigma}")

# Load posterior samples
print("\n[2/6] Loading posterior samples...")
idata = az.from_netcdf(IDATA_PATH)

# Extract posterior predictive samples (y_rep)
# Check what's in the posterior_predictive group
print(f"   - Available groups: {idata.groups()}")

if 'posterior_predictive' in idata.groups():
    print("   - Found posterior_predictive group")
    print(f"   - Variables: {list(idata.posterior_predictive.data_vars)}")

    # Get y_rep - might be called 'y_obs', 'y_rep', or 'y_pred'
    if 'y_rep' in idata.posterior_predictive:
        y_rep = idata.posterior_predictive['y_rep'].values
    elif 'y_obs' in idata.posterior_predictive:
        y_rep = idata.posterior_predictive['y_obs'].values
    elif 'y_pred' in idata.posterior_predictive:
        y_rep = idata.posterior_predictive['y_pred'].values
    else:
        raise ValueError(f"Could not find posterior predictive samples. Available: {list(idata.posterior_predictive.data_vars)}")

    print(f"   - y_rep shape: {y_rep.shape}")
    # Reshape to (n_samples, n_studies)
    if y_rep.ndim == 3:  # (chains, draws, studies)
        n_chains, n_draws, n_studies_check = y_rep.shape
        y_rep = y_rep.reshape(-1, n_studies_check)
        print(f"   - Reshaped from ({n_chains}, {n_draws}, {n_studies_check}) to {y_rep.shape}")
else:
    # Generate posterior predictive samples from posterior
    print("   - No posterior_predictive group found, generating from posterior...")

    # Extract posterior samples
    theta = idata.posterior['theta'].values  # (chains, draws, studies)
    n_chains, n_draws, n_studies_check = theta.shape
    theta = theta.reshape(-1, n_studies_check)

    # Generate y_rep ~ Normal(theta, sigma)
    n_samples = theta.shape[0]
    y_rep = np.random.normal(theta, sigma.reshape(1, -1))
    print(f"   - Generated {n_samples} posterior predictive samples per study")

n_samples = y_rep.shape[0]
print(f"   - Total posterior predictive samples: {n_samples} per study")

# ============================================================================
# 3. COMPUTE POSTERIOR PREDICTIVE INTERVALS AND P-VALUES
# ============================================================================

print("\n[3/6] Computing posterior predictive statistics...")

results = []
n_outliers = 0

for i in range(n_studies):
    y_i = y_obs[i]
    y_rep_i = y_rep[:, i]

    # Posterior predictive intervals
    ppi_50 = np.percentile(y_rep_i, [25, 75])
    ppi_95 = np.percentile(y_rep_i, [2.5, 97.5])

    # Posterior predictive p-value: P(y_rep < y_obs)
    ppc_pval = np.mean(y_rep_i < y_i)
    # Two-sided version (distance from center)
    ppc_pval_2sided = 2 * min(ppc_pval, 1 - ppc_pval)

    # Check if outlier (outside 95% PPI)
    is_outlier = (y_i < ppi_95[0]) or (y_i > ppi_95[1])
    if is_outlier:
        n_outliers += 1

    # Mean and SD of posterior predictive
    y_rep_mean = np.mean(y_rep_i)
    y_rep_sd = np.std(y_rep_i)

    results.append({
        'study': i + 1,
        'y_obs': y_i,
        'sigma': sigma[i],
        'y_rep_mean': y_rep_mean,
        'y_rep_sd': y_rep_sd,
        'ppi_95_lower': ppi_95[0],
        'ppi_95_upper': ppi_95[1],
        'ppi_50_lower': ppi_50[0],
        'ppi_50_upper': ppi_50[1],
        'ppc_pval': ppc_pval,
        'ppc_pval_2sided': ppc_pval_2sided,
        'is_outlier': is_outlier,
        'residual': y_i - y_rep_mean,
        'std_residual': (y_i - y_rep_mean) / y_rep_sd
    })

results_df = pd.DataFrame(results)

print(f"\n   STUDY-LEVEL RESULTS:")
print(f"   {'Study':<8} {'y_obs':<8} {'95% PPI':<20} {'p-value':<10} {'Outlier?':<10}")
print(f"   {'-'*70}")
for _, row in results_df.iterrows():
    ppi_str = f"[{row['ppi_95_lower']:6.1f}, {row['ppi_95_upper']:6.1f}]"
    outlier_str = "YES ***" if row['is_outlier'] else "No"
    print(f"   {row['study']:<8.0f} {row['y_obs']:<8.1f} {ppi_str:<20} {row['ppc_pval_2sided']:<10.3f} {outlier_str:<10}")

print(f"\n   FALSIFICATION CRITERION:")
print(f"   - Number of outliers (outside 95% PPI): {n_outliers}")
print(f"   - Criterion: REJECT if >1 outlier")
if n_outliers > 1:
    print(f"   - VERDICT: MODEL REJECTED (>1 outlier detected)")
elif n_outliers == 1:
    print(f"   - VERDICT: MODEL ACCEPTABLE (exactly 1 outlier, within tolerance)")
else:
    print(f"   - VERDICT: MODEL EXCELLENT (no outliers)")

# Save results
results_df.to_csv(f"{OUTPUT_DIR}/ppc_study_results.csv", index=False)
print(f"\n   - Saved: ppc_study_results.csv")

# ============================================================================
# 4. GLOBAL TEST STATISTICS
# ============================================================================

print("\n[4/6] Computing global test statistics...")

# Define test statistics
def compute_test_stats(y):
    """Compute test statistics for a set of observations"""
    return {
        'mean': np.mean(y),
        'sd': np.std(y, ddof=1),
        'min': np.min(y),
        'max': np.max(y),
        'range': np.max(y) - np.min(y),
        'q25': np.percentile(y, 25),
        'q75': np.percentile(y, 75),
        'iqr': np.percentile(y, 75) - np.percentile(y, 25),
        'max_abs': np.max(np.abs(y)),
        'n_positive': np.sum(y > 0),
        'n_negative': np.sum(y < 0)
    }

# Observed test statistics
T_obs = compute_test_stats(y_obs)

# Replicated test statistics
T_rep = {key: [] for key in T_obs.keys()}
for s in range(n_samples):
    T_s = compute_test_stats(y_rep[s, :])
    for key in T_obs.keys():
        T_rep[key].append(T_s[key])

# Convert to arrays
for key in T_rep.keys():
    T_rep[key] = np.array(T_rep[key])

# Compute posterior predictive p-values
global_results = []
for key in T_obs.keys():
    pval = np.mean(T_rep[key] >= T_obs[key])
    # Two-sided
    pval_2sided = 2 * min(pval, 1 - pval)

    global_results.append({
        'statistic': key,
        'T_obs': T_obs[key],
        'T_rep_mean': np.mean(T_rep[key]),
        'T_rep_sd': np.std(T_rep[key]),
        'T_rep_q025': np.percentile(T_rep[key], 2.5),
        'T_rep_q975': np.percentile(T_rep[key], 97.5),
        'ppc_pval': pval,
        'ppc_pval_2sided': pval_2sided,
        'extreme': (pval < 0.025) or (pval > 0.975)
    })

global_df = pd.DataFrame(global_results)

print(f"\n   GLOBAL TEST STATISTICS:")
print(f"   {'Statistic':<12} {'T_obs':<10} {'T_rep (95% PI)':<25} {'p-value':<10} {'Extreme?':<10}")
print(f"   {'-'*80}")
for _, row in global_df.iterrows():
    t_rep_str = f"[{row['T_rep_q025']:6.1f}, {row['T_rep_q975']:6.1f}]"
    extreme_str = "YES" if row['extreme'] else "No"
    print(f"   {row['statistic']:<12} {row['T_obs']:<10.2f} {t_rep_str:<25} {row['ppc_pval_2sided']:<10.3f} {extreme_str:<10}")

global_df.to_csv(f"{OUTPUT_DIR}/ppc_global_statistics.csv", index=False)
print(f"\n   - Saved: ppc_global_statistics.csv")

# ============================================================================
# 5. VISUALIZATIONS
# ============================================================================

print("\n[5/6] Creating diagnostic visualizations...")

# ------------------------------------------------------------------------
# PLOT 1: Study-by-study PPC (8-panel)
# ------------------------------------------------------------------------
print("   - Creating study_by_study_ppc.png...")
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.flatten()

for i in range(n_studies):
    ax = axes[i]
    y_rep_i = y_rep[:, i]
    y_i = y_obs[i]

    # Density plot
    ax.hist(y_rep_i, bins=50, density=True, alpha=0.6, color='steelblue',
            edgecolor='none', label='Posterior predictive')

    # Observed value
    ax.axvline(y_i, color='red', linewidth=2, linestyle='--', label=f'Observed ({y_i:.0f})')

    # 95% PPI
    ppi_95 = np.percentile(y_rep_i, [2.5, 97.5])
    ax.axvline(ppi_95[0], color='gray', linewidth=1, linestyle=':', alpha=0.7)
    ax.axvline(ppi_95[1], color='gray', linewidth=1, linestyle=':', alpha=0.7)
    ax.axvspan(ppi_95[0], ppi_95[1], alpha=0.1, color='gray', label='95% PPI')

    # Mark if outlier
    is_outlier = results_df.iloc[i]['is_outlier']
    title_suffix = " [OUTLIER]" if is_outlier else ""
    ax.set_title(f"Study {i+1}{title_suffix}", fontweight='bold' if is_outlier else 'normal',
                color='red' if is_outlier else 'black')

    ax.set_xlabel("Effect size")
    ax.set_ylabel("Density")
    if i == 0:
        ax.legend(fontsize=8, loc='upper right')

plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/study_by_study_ppc.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"      Saved: study_by_study_ppc.png")

# ------------------------------------------------------------------------
# PLOT 2: Overall PPC summary (forest plot style)
# ------------------------------------------------------------------------
print("   - Creating ppc_summary_intervals.png...")
fig, ax = plt.subplots(figsize=(10, 6))

studies = results_df['study'].values
y_obs_vals = results_df['y_obs'].values
ppi_95_lower = results_df['ppi_95_lower'].values
ppi_95_upper = results_df['ppi_95_upper'].values
ppi_50_lower = results_df['ppi_50_lower'].values
ppi_50_upper = results_df['ppi_50_upper'].values
y_rep_means = results_df['y_rep_mean'].values
is_outlier = results_df['is_outlier'].values

# Plot intervals
for i in range(n_studies):
    color = 'red' if is_outlier[i] else 'steelblue'
    linewidth = 2 if is_outlier[i] else 1

    # 95% PPI
    ax.plot([ppi_95_lower[i], ppi_95_upper[i]], [i, i],
            color=color, linewidth=linewidth, alpha=0.5)

    # 50% PPI
    ax.plot([ppi_50_lower[i], ppi_50_upper[i]], [i, i],
            color=color, linewidth=linewidth*2, alpha=0.8)

    # Mean
    ax.plot(y_rep_means[i], i, 'o', color=color, markersize=6, alpha=0.8)

    # Observed
    marker = 'X' if is_outlier[i] else 'D'
    markersize = 10 if is_outlier[i] else 7
    ax.plot(y_obs_vals[i], i, marker=marker, color='black',
            markersize=markersize, markeredgewidth=1.5, markerfacecolor='white',
            label='Observed' if i == 0 else '')

ax.set_yticks(range(n_studies))
ax.set_yticklabels([f"Study {s}" for s in studies])
ax.set_xlabel("Effect size")
ax.set_title("Posterior Predictive Check: Observed vs Predicted\n" +
             "(Black diamonds = observed; Thick line = 50% PPI; Thin line = 95% PPI)")
ax.axvline(0, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
ax.grid(True, alpha=0.3)
ax.legend()

plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/ppc_summary_intervals.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"      Saved: ppc_summary_intervals.png")

# ------------------------------------------------------------------------
# PLOT 3: Calibration plot (observed vs predicted)
# ------------------------------------------------------------------------
print("   - Creating calibration_plot.png...")
fig, ax = plt.subplots(figsize=(8, 8))

# Scatter with error bars
ax.errorbar(y_rep_means, y_obs_vals,
            xerr=[y_rep_means - ppi_95_lower, ppi_95_upper - y_rep_means],
            fmt='none', color='gray', alpha=0.5, linewidth=1, capsize=3)

# Points
colors = ['red' if outlier else 'steelblue' for outlier in is_outlier]
sizes = [150 if outlier else 80 for outlier in is_outlier]
ax.scatter(y_rep_means, y_obs_vals, c=colors, s=sizes, alpha=0.7, edgecolors='black', linewidth=1)

# Labels
for i in range(n_studies):
    offset = 1.5 if is_outlier[i] else 1.0
    ax.annotate(f"{int(studies[i])}", (y_rep_means[i], y_obs_vals[i]),
                fontsize=9, ha='center', va='center', fontweight='bold' if is_outlier[i] else 'normal')

# Perfect calibration line
lims = [min(ax.get_xlim()[0], ax.get_ylim()[0]),
        max(ax.get_xlim()[1], ax.get_ylim()[1])]
ax.plot(lims, lims, 'k--', alpha=0.5, linewidth=1.5, label='Perfect calibration')

ax.set_xlabel("Posterior predictive mean", fontsize=12)
ax.set_ylabel("Observed value", fontsize=12)
ax.set_title("Calibration Plot: Observed vs Predicted\n(Error bars = 95% PPI; Red = outliers)",
             fontsize=13)
ax.grid(True, alpha=0.3)
ax.legend()
ax.set_aspect('equal')

plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/calibration_plot.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"      Saved: calibration_plot.png")

# ------------------------------------------------------------------------
# PLOT 4: Residual diagnostics
# ------------------------------------------------------------------------
print("   - Creating residual_diagnostics.png...")
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# (a) Residuals vs predicted
ax = axes[0, 0]
colors = ['red' if outlier else 'steelblue' for outlier in is_outlier]
sizes = [150 if outlier else 80 for outlier in is_outlier]
ax.scatter(y_rep_means, results_df['residual'], c=colors, s=sizes, alpha=0.7,
          edgecolors='black', linewidth=1)
for i in range(n_studies):
    ax.annotate(f"{int(studies[i])}", (y_rep_means[i], results_df['residual'].values[i]),
                fontsize=9, ha='center', va='center')
ax.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
ax.set_xlabel("Posterior predictive mean")
ax.set_ylabel("Residual (observed - predicted)")
ax.set_title("(a) Residuals vs Predicted")
ax.grid(True, alpha=0.3)

# (b) Standardized residuals
ax = axes[0, 1]
ax.scatter(range(1, n_studies+1), results_df['std_residual'], c=colors, s=sizes, alpha=0.7,
          edgecolors='black', linewidth=1)
ax.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
ax.axhline(2, color='red', linestyle=':', linewidth=1, alpha=0.5, label='±2 SD')
ax.axhline(-2, color='red', linestyle=':', linewidth=1, alpha=0.5)
ax.set_xlabel("Study")
ax.set_ylabel("Standardized residual")
ax.set_title("(b) Standardized Residuals")
ax.set_xticks(range(1, n_studies+1))
ax.grid(True, alpha=0.3)
ax.legend()

# (c) Q-Q plot
ax = axes[1, 0]
stats.probplot(results_df['std_residual'], dist="norm", plot=ax)
ax.get_lines()[0].set_marker('o')
ax.get_lines()[0].set_markersize(8)
ax.get_lines()[0].set_markerfacecolor('steelblue')
ax.get_lines()[0].set_markeredgecolor('black')
ax.get_lines()[0].set_alpha(0.7)
ax.set_title("(c) Q-Q Plot (Standardized Residuals)")
ax.grid(True, alpha=0.3)

# (d) Histogram of residuals
ax = axes[1, 1]
ax.hist(results_df['residual'], bins=8, density=True, alpha=0.6,
       color='steelblue', edgecolor='black')
# Overlay normal distribution
x = np.linspace(results_df['residual'].min(), results_df['residual'].max(), 100)
ax.plot(x, stats.norm.pdf(x, results_df['residual'].mean(), results_df['residual'].std()),
       'r-', linewidth=2, label='Normal fit')
ax.set_xlabel("Residual")
ax.set_ylabel("Density")
ax.set_title("(d) Distribution of Residuals")
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/residual_diagnostics.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"      Saved: residual_diagnostics.png")

# ------------------------------------------------------------------------
# PLOT 5: Test statistic distributions
# ------------------------------------------------------------------------
print("   - Creating test_statistic_distributions.png...")

# Select key test statistics
key_stats = ['max', 'min', 'range', 'sd', 'max_abs']
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for idx, stat in enumerate(key_stats):
    ax = axes[idx]

    # Histogram of T_rep
    ax.hist(T_rep[stat], bins=50, density=True, alpha=0.6,
           color='steelblue', edgecolor='none', label='Posterior predictive')

    # Observed value
    T_obs_val = T_obs[stat]
    ax.axvline(T_obs_val, color='red', linewidth=2, linestyle='--',
              label=f'Observed ({T_obs_val:.1f})')

    # 95% interval
    q025, q975 = np.percentile(T_rep[stat], [2.5, 97.5])
    ax.axvline(q025, color='gray', linewidth=1, linestyle=':', alpha=0.7)
    ax.axvline(q975, color='gray', linewidth=1, linestyle=':', alpha=0.7)
    ax.axvspan(q025, q975, alpha=0.1, color='gray')

    # P-value
    pval = global_df[global_df['statistic'] == stat]['ppc_pval_2sided'].values[0]

    ax.set_xlabel(stat.upper())
    ax.set_ylabel("Density")
    ax.set_title(f"{stat.upper()} (p = {pval:.3f})")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

# Remove extra subplot
axes[-1].axis('off')

plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/test_statistic_distributions.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"      Saved: test_statistic_distributions.png")

# ------------------------------------------------------------------------
# PLOT 6: ArviZ built-in PPC plot
# ------------------------------------------------------------------------
print("   - Creating arviz_ppc.png...")
try:
    # Use ArviZ plot_ppc
    fig, ax = plt.subplots(figsize=(10, 6))
    az.plot_ppc(idata, ax=ax, num_pp_samples=100, random_seed=42)
    ax.set_title("ArviZ Posterior Predictive Check\n(100 replications overlaid)")
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/arviz_ppc.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"      Saved: arviz_ppc.png")
except Exception as e:
    print(f"      Warning: Could not create ArviZ PPC plot: {e}")

# ------------------------------------------------------------------------
# PLOT 7: LOO-PIT (Probability Integral Transform)
# ------------------------------------------------------------------------
print("   - Creating loo_pit.png...")
try:
    fig, ax = plt.subplots(figsize=(10, 6))
    az.plot_loo_pit(idata, y='y_obs', ecdf=True, ax=ax)
    ax.set_title("LOO-PIT: Calibration Check\n(Uniform distribution = well-calibrated)")
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/loo_pit.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"      Saved: loo_pit.png")
except Exception as e:
    print(f"      Warning: Could not create LOO-PIT plot: {e}")

# ============================================================================
# 6. SUMMARY STATISTICS AND EXPORT
# ============================================================================

print("\n[6/6] Generating summary statistics...")

summary = {
    'n_studies': int(n_studies),
    'n_posterior_samples': int(n_samples),
    'n_outliers': int(n_outliers),
    'outlier_studies': results_df[results_df['is_outlier']]['study'].tolist(),
    'falsification_criterion_met': bool(n_outliers > 1),
    'verdict': 'REJECT' if n_outliers > 1 else ('ACCEPTABLE' if n_outliers == 1 else 'EXCELLENT'),
    'extreme_test_statistics': global_df[global_df['extreme']]['statistic'].tolist(),
    'global_mean_pval': float(global_df[global_df['statistic'] == 'mean']['ppc_pval_2sided'].values[0]),
    'global_sd_pval': float(global_df[global_df['statistic'] == 'sd']['ppc_pval_2sided'].values[0]),
    'global_range_pval': float(global_df[global_df['statistic'] == 'range']['ppc_pval_2sided'].values[0])
}

with open(f"{OUTPUT_DIR}/ppc_summary.json", 'w') as f:
    json.dump(summary, f, indent=2)

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"Number of studies: {summary['n_studies']}")
print(f"Posterior predictive samples: {summary['n_posterior_samples']}")
print(f"Studies outside 95% PPI: {summary['n_outliers']}")
if summary['n_outliers'] > 0:
    print(f"   - Outlier studies: {summary['outlier_studies']}")
print(f"Falsification criterion (>1 outlier): {'MET (REJECT)' if summary['falsification_criterion_met'] else 'NOT MET'}")
print(f"Overall verdict: {summary['verdict']}")
print(f"\nGlobal test statistics:")
print(f"   - Mean: p = {summary['global_mean_pval']:.3f}")
print(f"   - SD: p = {summary['global_sd_pval']:.3f}")
print(f"   - Range: p = {summary['global_range_pval']:.3f}")
if summary['extreme_test_statistics']:
    print(f"   - Extreme statistics: {summary['extreme_test_statistics']}")

print("\n" + "="*80)
print("FILES SAVED")
print("="*80)
print(f"CSV files:")
print(f"   - {OUTPUT_DIR}/ppc_study_results.csv")
print(f"   - {OUTPUT_DIR}/ppc_global_statistics.csv")
print(f"\nJSON:")
print(f"   - {OUTPUT_DIR}/ppc_summary.json")
print(f"\nPlots:")
print(f"   - {PLOTS_DIR}/study_by_study_ppc.png")
print(f"   - {PLOTS_DIR}/ppc_summary_intervals.png")
print(f"   - {PLOTS_DIR}/calibration_plot.png")
print(f"   - {PLOTS_DIR}/residual_diagnostics.png")
print(f"   - {PLOTS_DIR}/test_statistic_distributions.png")
print(f"   - {PLOTS_DIR}/arviz_ppc.png")
print(f"   - {PLOTS_DIR}/loo_pit.png")

print("\n" + "="*80)
print("POSTERIOR PREDICTIVE CHECK COMPLETE")
print("="*80)
