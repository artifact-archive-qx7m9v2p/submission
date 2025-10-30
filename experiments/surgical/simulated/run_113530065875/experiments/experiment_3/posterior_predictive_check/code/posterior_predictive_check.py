"""
Posterior Predictive Check for Experiment 3: Beta-Binomial Model

This script performs comprehensive posterior predictive checks to assess
whether the fitted Beta-Binomial model can reproduce key features of the
observed data, with special focus on:

1. Overdispersion capture (CRITICAL - model's main purpose)
2. LOO-CV diagnostics (CRITICAL - main advantage over hierarchical)
3. Group-level fit
4. Comparison to Experiment 1

Author: Model Validation Specialist
Date: 2025-10-30
"""

import sys
sys.path.append('/tmp/agent-home/.local/lib/python3.13/site-packages')

import numpy as np
import pandas as pd
import arviz as az
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# Configuration
DATA_FILE = '/workspace/data/data.csv'
POSTERIOR_FILE = '/workspace/experiments/experiment_3/posterior_inference/diagnostics/posterior_inference.netcdf'
OUTPUT_DIR = '/workspace/experiments/experiment_3/posterior_predictive_check'
PLOTS_DIR = f'{OUTPUT_DIR}/plots'
DIAGNOSTICS_DIR = f'{OUTPUT_DIR}/diagnostics'

# Plotting configuration
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10

print("="*80)
print("POSTERIOR PREDICTIVE CHECK: BETA-BINOMIAL MODEL (EXPERIMENT 3)")
print("="*80)
print()

# ============================================================================
# 1. LOAD DATA AND POSTERIOR
# ============================================================================

print("1. LOADING DATA AND POSTERIOR SAMPLES")
print("-" * 80)

# Load observed data
data = pd.read_csv(DATA_FILE)
J = len(data)
n_obs = data['n'].values
r_obs = data['r'].values
p_obs = r_obs / n_obs

print(f"Loaded data: {J} groups")
print(f"Sample sizes: {n_obs.min()} to {n_obs.max()}")
print(f"Successes: {r_obs.min()} to {r_obs.max()}")
print(f"Observed rates: {p_obs.min():.3f} to {p_obs.max():.3f}")
print()

# Load posterior
idata = az.from_netcdf(POSTERIOR_FILE)
print(f"Loaded posterior: {idata.posterior.dims}")
print(f"Parameters: {list(idata.posterior.data_vars)}")
print()

# Extract posterior samples
mu_p_samples = idata.posterior['mu_p'].values.flatten()
kappa_samples = idata.posterior['kappa'].values.flatten()
phi_samples = idata.posterior['phi'].values.flatten()

n_samples = len(mu_p_samples)
print(f"Total posterior samples: {n_samples}")
print()

# Posterior summary
print("Posterior Summary:")
print(f"  mu_p:  {mu_p_samples.mean():.4f} ± {mu_p_samples.std():.4f}")
print(f"  kappa: {kappa_samples.mean():.2f} ± {kappa_samples.std():.2f}")
print(f"  phi:   {phi_samples.mean():.4f} ± {phi_samples.std():.4f}")
print()

# ============================================================================
# 2. GENERATE POSTERIOR PREDICTIVE SAMPLES
# ============================================================================

print("2. GENERATING POSTERIOR PREDICTIVE SAMPLES")
print("-" * 80)

# Generate posterior predictive samples
# Beta-Binomial: r ~ BetaBinomial(n, alpha, beta)
# where alpha = mu_p * kappa, beta = (1 - mu_p) * kappa

n_pp_samples = 1000  # Use 1000 PP replicates
pp_indices = np.random.choice(n_samples, size=n_pp_samples, replace=False)

print(f"Generating {n_pp_samples} posterior predictive datasets...")

# Storage for PP samples
r_pp = np.zeros((n_pp_samples, J))

for i, idx in enumerate(pp_indices):
    mu_p = mu_p_samples[idx]
    kappa = kappa_samples[idx]

    alpha = mu_p * kappa
    beta = (1 - mu_p) * kappa

    # Generate beta-binomial samples for each group
    for j in range(J):
        # Beta-binomial: first draw p ~ Beta(alpha, beta), then r ~ Binomial(n, p)
        p_j = np.random.beta(alpha, beta)
        r_pp[i, j] = np.random.binomial(n_obs[j], p_j)

print(f"Generated {r_pp.shape[0]} × {r_pp.shape[1]} PP samples")
print()

# Compute PP rates
p_pp = r_pp / n_obs[np.newaxis, :]

# ============================================================================
# 3. TEST STATISTICS
# ============================================================================

print("3. COMPUTING TEST STATISTICS")
print("-" * 80)

# ============================================================================
# 3A. OVERDISPERSION CHECK (CRITICAL)
# ============================================================================

print("\n3A. OVERDISPERSION CHECK (CRITICAL)")
print("-" * 40)

# Observed overdispersion
# phi = variance_observed / variance_binomial
p_bar_obs = r_obs.sum() / n_obs.sum()
var_binomial_obs = ((n_obs * p_bar_obs * (1 - p_bar_obs)) * ((J - 1) / J)).sum() / J
var_observed = np.sum(n_obs * (p_obs - p_bar_obs)**2) / (J - 1)
phi_obs = var_observed / var_binomial_obs

print(f"Observed variance: {var_observed:.4f}")
print(f"Binomial variance: {var_binomial_obs:.4f}")
print(f"Observed φ: {phi_obs:.4f}")
print()

# Compute φ for each PP dataset
phi_pp = np.zeros(n_pp_samples)
for i in range(n_pp_samples):
    r_rep = r_pp[i, :]
    p_rep = r_rep / n_obs
    p_bar_rep = r_rep.sum() / n_obs.sum()

    var_binomial_rep = ((n_obs * p_bar_rep * (1 - p_bar_rep)) * ((J - 1) / J)).sum() / J
    var_rep = np.sum(n_obs * (p_rep - p_bar_rep)**2) / (J - 1)
    phi_pp[i] = var_rep / var_binomial_rep

# Test results
phi_pp_lower = np.percentile(phi_pp, 2.5)
phi_pp_upper = np.percentile(phi_pp, 97.5)
phi_pp_10 = np.percentile(phi_pp, 5)
phi_pp_90 = np.percentile(phi_pp, 95)
p_value_phi = np.mean(phi_pp >= phi_obs)

print(f"PP φ distribution:")
print(f"  Median: {np.median(phi_pp):.4f}")
print(f"  95% CI: [{phi_pp_lower:.4f}, {phi_pp_upper:.4f}]")
print(f"  90% CI: [{phi_pp_10:.4f}, {phi_pp_90:.4f}]")
print()

print(f"Overdispersion Test:")
print(f"  φ_obs = {phi_obs:.4f}")
print(f"  Bayesian p-value: {p_value_phi:.4f}")

# Decision criteria
in_95_interval = phi_pp_lower <= phi_obs <= phi_pp_upper
in_90_interval = phi_pp_10 <= phi_obs <= phi_pp_90

if in_95_interval:
    status = "PASS"
    print(f"  Status: {status} - φ_obs within 95% PP interval")
elif in_90_interval:
    status = "INVESTIGATE"
    print(f"  Status: {status} - φ_obs outside 95% but within 90% PP interval")
elif p_value_phi >= 0.05:
    status = "INVESTIGATE"
    print(f"  Status: {status} - φ_obs outside 90% interval but p-value acceptable")
else:
    status = "FAIL"
    print(f"  Status: {status} - φ_obs badly outside PP interval (p < 0.05)")

overdispersion_status = status
print()

# ============================================================================
# 3B. RANGE CHECK
# ============================================================================

print("3B. RANGE CHECK")
print("-" * 40)

# Observed range
min_obs = p_obs.min()
max_obs = p_obs.max()

# PP ranges
min_pp = p_pp.min(axis=1)
max_pp = p_pp.max(axis=1)

# Test: Can model generate observed extremes?
p_value_min = np.mean(min_pp <= min_obs)
p_value_max = np.mean(max_pp >= max_obs)

print(f"Observed range: [{min_obs:.4f}, {max_obs:.4f}]")
print(f"PP min range: [{np.percentile(min_pp, 2.5):.4f}, {np.percentile(min_pp, 97.5):.4f}]")
print(f"PP max range: [{np.percentile(max_pp, 2.5):.4f}, {np.percentile(max_pp, 97.5):.4f}]")
print()
print(f"Can generate observed minimum? p = {p_value_min:.4f}")
print(f"Can generate observed maximum? p = {p_value_max:.4f}")

if p_value_min >= 0.025 and p_value_max >= 0.025:
    range_status = "PASS"
else:
    range_status = "CONCERN"
print(f"Range check: {range_status}")
print()

# ============================================================================
# 3C. LOO CROSS-VALIDATION (CRITICAL)
# ============================================================================

print("3C. LOO CROSS-VALIDATION (CRITICAL)")
print("-" * 40)

# Compute LOO
try:
    loo = az.loo(idata, pointwise=True)

    # Extract Pareto k values
    pareto_k = loo.pareto_k.values

    print(f"LOO-CV Results:")
    print(f"  ELPD LOO: {loo.elpd_loo:.2f} ± {loo.se:.2f}")
    print(f"  p_loo: {loo.p_loo:.2f}")
    print()

    print(f"Pareto k diagnostics:")
    print(f"  Mean k: {pareto_k.mean():.4f}")
    print(f"  Max k: {pareto_k.max():.4f}")
    print(f"  Groups with k < 0.5 (good): {np.sum(pareto_k < 0.5)}/{J}")
    print(f"  Groups with 0.5 ≤ k < 0.7 (ok): {np.sum((pareto_k >= 0.5) & (pareto_k < 0.7))}/{J}")
    print(f"  Groups with k ≥ 0.7 (bad): {np.sum(pareto_k >= 0.7)}/{J}")
    print()

    # Individual group k values
    print("Individual Pareto k values:")
    for j in range(J):
        k_val = pareto_k[j]
        if k_val < 0.5:
            quality = "good"
        elif k_val < 0.7:
            quality = "ok"
        else:
            quality = "BAD"
        print(f"  Group {j+1:2d}: k = {k_val:.4f} ({quality})")
    print()

    # Decision
    n_bad = np.sum(pareto_k >= 0.7)
    if n_bad == 0:
        loo_status = "PASS"
        print(f"LOO Status: {loo_status} - All k < 0.7")
    elif n_bad <= 3:
        loo_status = "INVESTIGATE"
        print(f"LOO Status: {loo_status} - Few groups with k ≥ 0.7")
    else:
        loo_status = "FAIL"
        print(f"LOO Status: {loo_status} - Many groups with k ≥ 0.7")

    # Compare to Exp 1
    print()
    print("COMPARISON TO EXPERIMENT 1:")
    print("  Exp 1 (Hierarchical): 10/12 groups with k > 0.7 (FAIL)")
    print(f"  Exp 3 (Beta-Binomial): {n_bad}/12 groups with k ≥ 0.7")
    if n_bad < 10:
        print("  ADVANTAGE: Exp 3 has better LOO diagnostics")
    else:
        print("  NO ADVANTAGE: Both models have LOO issues")

except Exception as e:
    print(f"LOO computation failed: {e}")
    loo = None
    pareto_k = None
    loo_status = "ERROR"

print()

# ============================================================================
# 3D. INDIVIDUAL GROUP FIT
# ============================================================================

print("3D. INDIVIDUAL GROUP FIT")
print("-" * 40)

# Compute Bayesian p-values for each group
p_values_group = np.zeros(J)

for j in range(J):
    # Test statistic: observed count
    r_obs_j = r_obs[j]
    r_pp_j = r_pp[:, j]

    # Two-sided p-value
    p_values_group[j] = 2 * min(np.mean(r_pp_j >= r_obs_j), np.mean(r_pp_j <= r_obs_j))

print("Individual group Bayesian p-values:")
for j in range(J):
    p_val = p_values_group[j]
    if p_val < 0.05:
        flag = "CONCERN"
    else:
        flag = "OK"
    print(f"  Group {j+1:2d}: p = {p_val:.4f} ({flag})")

# Check extreme groups from EDA (groups 2, 4, 8)
extreme_groups = [1, 3, 7]  # 0-indexed
print()
print("Extreme groups from EDA:")
for idx in extreme_groups:
    print(f"  Group {idx+1}: p = {p_values_group[idx]:.4f}")

n_concern = np.sum(p_values_group < 0.05)
if n_concern == 0:
    group_status = "PASS"
elif n_concern <= 2:
    group_status = "INVESTIGATE"
else:
    group_status = "FAIL"

print()
print(f"Groups with p < 0.05: {n_concern}/{J}")
print(f"Group fit status: {group_status}")
print()

# ============================================================================
# 3E. SUMMARY STATISTICS
# ============================================================================

print("3E. SUMMARY STATISTICS COMPARISON")
print("-" * 40)

# Compute summary statistics
def compute_summary_stats(r, n):
    p = r / n
    return {
        'mean_p': p.mean(),
        'sd_p': p.std(),
        'min_p': p.min(),
        'max_p': p.max(),
        'q25_p': np.percentile(p, 25),
        'q75_p': np.percentile(p, 75)
    }

stats_obs = compute_summary_stats(r_obs, n_obs)

# Compute for each PP dataset
stats_pp = {key: [] for key in stats_obs.keys()}
for i in range(n_pp_samples):
    stats_i = compute_summary_stats(r_pp[i, :], n_obs)
    for key in stats_obs.keys():
        stats_pp[key].append(stats_i[key])

# Convert to arrays
for key in stats_pp.keys():
    stats_pp[key] = np.array(stats_pp[key])

# Print comparison
print("Summary Statistics Comparison:")
print(f"{'Statistic':<15} {'Observed':>10} {'PP Median':>10} {'PP 95% CI':>25}")
print("-" * 65)

for key in stats_obs.keys():
    obs_val = stats_obs[key]
    pp_median = np.median(stats_pp[key])
    pp_lower = np.percentile(stats_pp[key], 2.5)
    pp_upper = np.percentile(stats_pp[key], 97.5)

    print(f"{key:<15} {obs_val:>10.4f} {pp_median:>10.4f} [{pp_lower:>7.4f}, {pp_upper:>7.4f}]")

print()

# ============================================================================
# 4. VISUALIZATIONS
# ============================================================================

print("4. CREATING DIAGNOSTIC VISUALIZATIONS")
print("-" * 80)

# ============================================================================
# PLOT 1: OVERDISPERSION DIAGNOSTIC
# ============================================================================

print("Creating Plot 1: Overdispersion Diagnostic...")

fig, ax = plt.subplots(figsize=(10, 6))

# Histogram of PP φ
ax.hist(phi_pp, bins=50, density=True, alpha=0.6, color='steelblue',
        edgecolor='black', label='Posterior Predictive')

# Observed φ
ax.axvline(phi_obs, color='red', linewidth=2.5, label=f'Observed φ = {phi_obs:.3f}')

# PP intervals
ax.axvline(phi_pp_lower, color='steelblue', linewidth=1.5, linestyle='--',
           label=f'95% PP CI [{phi_pp_lower:.3f}, {phi_pp_upper:.3f}]')
ax.axvline(phi_pp_upper, color='steelblue', linewidth=1.5, linestyle='--')

# Formatting
ax.set_xlabel('Overdispersion φ (Var_obs / Var_binomial)', fontsize=12)
ax.set_ylabel('Density', fontsize=12)
ax.set_title('Overdispersion Check: Can Beta-Binomial Model Reproduce Observed Variance?\n' +
             f'Bayesian p-value = {p_value_phi:.3f} | Status: {overdispersion_status}',
             fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# Add text box with interpretation
textstr = f'Observed φ from EDA: 3.59\n'
textstr += f'Posterior φ (median): {np.median(phi_samples):.3f}\n'
textstr += f'PP φ (median): {np.median(phi_pp):.3f}\n\n'

if overdispersion_status == "PASS":
    textstr += 'Model successfully captures\nobserved overdispersion'
    box_color = 'lightgreen'
elif overdispersion_status == "INVESTIGATE":
    textstr += 'Model marginally underfits\nobserved overdispersion'
    box_color = 'lightyellow'
else:
    textstr += 'Model fails to capture\nobserved overdispersion'
    box_color = 'lightcoral'

props = dict(boxstyle='round', facecolor=box_color, alpha=0.8)
ax.text(0.98, 0.97, textstr, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', horizontalalignment='right', bbox=props)

plt.tight_layout()
plt.savefig(f'{PLOTS_DIR}/1_overdispersion_diagnostic.png', dpi=300, bbox_inches='tight')
plt.close()

print("  Saved: 1_overdispersion_diagnostic.png")

# ============================================================================
# PLOT 2: POSTERIOR PREDICTIVE CHECK (ALL GROUPS)
# ============================================================================

print("Creating Plot 2: Posterior Predictive Check...")

# Use ArviZ PPC plot with custom styling
fig, ax = plt.subplots(figsize=(12, 6))

# Compute success rates for plotting
p_pp_plot = r_pp / n_obs[np.newaxis, :]

# Plot PP distributions for each group
for j in range(J):
    # Plot PP distribution as violin or density
    positions = np.random.normal(j + 1, 0.1, size=len(p_pp_plot[:, j]))
    ax.scatter(positions, p_pp_plot[:, j], alpha=0.02, color='steelblue', s=1)

    # Plot PP quantiles
    q25, q50, q75 = np.percentile(p_pp_plot[:, j], [25, 50, 75])
    ax.plot([j+1, j+1], [q25, q75], color='steelblue', linewidth=2, alpha=0.8)
    ax.scatter(j+1, q50, color='steelblue', s=50, zorder=5, alpha=0.8)

# Overlay observed
ax.scatter(range(1, J+1), p_obs, color='red', s=100, marker='D',
           label='Observed', zorder=10, edgecolor='black', linewidth=1)

# Formatting
ax.set_xlabel('Group', fontsize=12)
ax.set_ylabel('Success Rate', fontsize=12)
ax.set_title('Posterior Predictive Check: Observed vs Replicated Data (12 Groups)\n' +
             'Blue: PP distribution (median + IQR), Red diamonds: Observed',
             fontsize=13, fontweight='bold')
ax.set_xticks(range(1, J+1))
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, axis='y')

# Highlight extreme groups
extreme_labels = [2, 4, 8]  # 1-indexed
for eg in extreme_labels:
    ax.axvline(eg, color='orange', linewidth=1, linestyle='--', alpha=0.5)
    ax.text(eg, ax.get_ylim()[1] * 0.95, f'G{eg}', ha='center', fontsize=9,
            color='orange', fontweight='bold')

plt.tight_layout()
plt.savefig(f'{PLOTS_DIR}/2_ppc_all_groups.png', dpi=300, bbox_inches='tight')
plt.close()

print("  Saved: 2_ppc_all_groups.png")

# ============================================================================
# PLOT 3: LOO PARETO K COMPARISON
# ============================================================================

print("Creating Plot 3: LOO Pareto k Comparison...")

if loo is not None and pareto_k is not None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Left panel: Pareto k values
    colors = ['green' if k < 0.5 else 'orange' if k < 0.7 else 'red' for k in pareto_k]
    ax1.bar(range(1, J+1), pareto_k, color=colors, edgecolor='black', linewidth=1)

    # Threshold lines
    ax1.axhline(0.5, color='orange', linestyle='--', linewidth=1.5, label='k = 0.5 (threshold)')
    ax1.axhline(0.7, color='red', linestyle='--', linewidth=1.5, label='k = 0.7 (bad)')

    ax1.set_xlabel('Group', fontsize=11)
    ax1.set_ylabel('Pareto k', fontsize=11)
    ax1.set_title(f'Experiment 3: Beta-Binomial LOO Diagnostics\n' +
                  f'{np.sum(pareto_k >= 0.7)}/12 groups with k ≥ 0.7',
                  fontsize=12, fontweight='bold')
    ax1.set_xticks(range(1, J+1))
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3, axis='y')

    # Right panel: Comparison to Exp 1
    exp1_bad = 10  # From findings
    exp3_bad = np.sum(pareto_k >= 0.7)

    experiments = ['Exp 1\n(Hierarchical)', 'Exp 3\n(Beta-Binomial)']
    bad_counts = [exp1_bad, exp3_bad]
    good_counts = [12 - exp1_bad, 12 - exp3_bad]

    x_pos = np.arange(len(experiments))
    width = 0.6

    p1 = ax2.bar(x_pos, good_counts, width, label='k < 0.7 (good/ok)',
                 color='green', edgecolor='black')
    p2 = ax2.bar(x_pos, bad_counts, width, bottom=good_counts,
                 label='k ≥ 0.7 (bad)', color='red', edgecolor='black')

    # Add counts on bars
    for i, (good, bad) in enumerate(zip(good_counts, bad_counts)):
        if good > 0:
            ax2.text(i, good/2, str(good), ha='center', va='center',
                    fontweight='bold', fontsize=11, color='white')
        if bad > 0:
            ax2.text(i, good + bad/2, str(bad), ha='center', va='center',
                    fontweight='bold', fontsize=11, color='white')

    ax2.set_ylabel('Number of Groups', fontsize=11)
    ax2.set_title('LOO Reliability: Experiment 1 vs 3\n' +
                  'Lower is Better (Goal: All k < 0.7)',
                  fontsize=12, fontweight='bold')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(experiments)
    ax2.set_ylim(0, 14)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3, axis='y')

    # Add advantage note
    if exp3_bad < exp1_bad:
        advantage_text = f'ADVANTAGE: Exp 3 has {exp1_bad - exp3_bad} fewer bad groups'
        ax2.text(0.5, 13, advantage_text, ha='center', fontsize=10,
                fontweight='bold', color='green',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    else:
        advantage_text = 'NO ADVANTAGE: Both models have LOO issues'
        ax2.text(0.5, 13, advantage_text, ha='center', fontsize=10,
                fontweight='bold', color='red',
                bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))

    plt.tight_layout()
    plt.savefig(f'{PLOTS_DIR}/3_loo_pareto_k_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("  Saved: 3_loo_pareto_k_comparison.png")
else:
    print("  Skipped: LOO computation failed")

# ============================================================================
# PLOT 4: EXTREME GROUPS FOCUS
# ============================================================================

print("Creating Plot 4: Extreme Groups Focus...")

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

extreme_groups_1idx = [2, 4, 8]  # 1-indexed
extreme_groups_0idx = [1, 3, 7]  # 0-indexed

for i, (g1, g0) in enumerate(zip(extreme_groups_1idx, extreme_groups_0idx)):
    ax = axes[i]

    # Histogram of PP
    r_pp_g = r_pp[:, g0]
    ax.hist(r_pp_g, bins=30, density=True, alpha=0.6, color='steelblue',
            edgecolor='black', label='Posterior Predictive')

    # Observed
    ax.axvline(r_obs[g0], color='red', linewidth=2.5, label=f'Observed = {r_obs[g0]}')

    # PP quantiles
    q025, q975 = np.percentile(r_pp_g, [2.5, 97.5])
    ax.axvline(q025, color='steelblue', linewidth=1.5, linestyle='--')
    ax.axvline(q975, color='steelblue', linewidth=1.5, linestyle='--')

    # Formatting
    ax.set_xlabel('Number of Successes', fontsize=10)
    ax.set_ylabel('Density', fontsize=10)

    # Rate info
    obs_rate = r_obs[g0] / n_obs[g0]
    pp_rate_med = np.median(r_pp_g / n_obs[g0])

    ax.set_title(f'Group {g1} (n={n_obs[g0]})\n' +
                 f'Obs rate: {obs_rate:.3f}, PP rate: {pp_rate_med:.3f}\n' +
                 f'p-value: {p_values_group[g0]:.3f}',
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

plt.suptitle('Extreme Groups from EDA: Individual Fit Assessment',
             fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(f'{PLOTS_DIR}/4_extreme_groups.png', dpi=300, bbox_inches='tight')
plt.close()

print("  Saved: 4_extreme_groups.png")

# ============================================================================
# PLOT 5: OBSERVED VS PREDICTED
# ============================================================================

print("Creating Plot 5: Observed vs Predicted...")

fig, ax = plt.subplots(figsize=(10, 10))

# Compute PP means and intervals
r_pp_mean = r_pp.mean(axis=0)
r_pp_lower = np.percentile(r_pp, 2.5, axis=0)
r_pp_upper = np.percentile(r_pp, 97.5, axis=0)

# Scatter plot
ax.scatter(r_obs, r_pp_mean, s=100, alpha=0.7, c=n_obs, cmap='viridis',
           edgecolor='black', linewidth=1)

# Error bars
for j in range(J):
    ax.plot([r_obs[j], r_obs[j]], [r_pp_lower[j], r_pp_upper[j]],
            color='gray', alpha=0.5, linewidth=1)

# 1:1 line
max_val = max(r_obs.max(), r_pp_upper.max())
ax.plot([0, max_val], [0, max_val], 'k--', linewidth=2, alpha=0.5, label='Perfect fit')

# Formatting
ax.set_xlabel('Observed Successes', fontsize=12)
ax.set_ylabel('Predicted Successes (PP mean)', fontsize=12)
ax.set_title('Observed vs Predicted: Group-Level Fit\n' +
             'Error bars: 95% PP intervals, Color: Sample size',
             fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# Add colorbar
cbar = plt.colorbar(ax.collections[0], ax=ax, label='Sample Size (n)')

# Add group labels for extreme groups
for eg in extreme_groups_1idx:
    j = eg - 1
    ax.annotate(f'G{eg}', (r_obs[j], r_pp_mean[j]),
                xytext=(5, 5), textcoords='offset points',
                fontsize=9, color='red', fontweight='bold')

plt.tight_layout()
plt.savefig(f'{PLOTS_DIR}/5_observed_vs_predicted.png', dpi=300, bbox_inches='tight')
plt.close()

print("  Saved: 5_observed_vs_predicted.png")

# ============================================================================
# PLOT 6: RANGE DIAGNOSTIC
# ============================================================================

print("Creating Plot 6: Range Diagnostic...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Left: Minimum rates
ax1.hist(min_pp, bins=40, density=True, alpha=0.6, color='steelblue',
         edgecolor='black', label='PP Minimum')
ax1.axvline(min_obs, color='red', linewidth=2.5, label=f'Observed = {min_obs:.4f}')
ax1.axvline(np.percentile(min_pp, 2.5), color='steelblue', linewidth=1.5,
            linestyle='--', alpha=0.7)
ax1.axvline(np.percentile(min_pp, 97.5), color='steelblue', linewidth=1.5,
            linestyle='--', alpha=0.7)
ax1.set_xlabel('Minimum Success Rate', fontsize=11)
ax1.set_ylabel('Density', fontsize=11)
ax1.set_title(f'Can Model Generate Observed Minimum?\n' +
              f'p-value = {p_value_min:.3f}',
              fontsize=12, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

# Right: Maximum rates
ax2.hist(max_pp, bins=40, density=True, alpha=0.6, color='steelblue',
         edgecolor='black', label='PP Maximum')
ax2.axvline(max_obs, color='red', linewidth=2.5, label=f'Observed = {max_obs:.4f}')
ax2.axvline(np.percentile(max_pp, 2.5), color='steelblue', linewidth=1.5,
            linestyle='--', alpha=0.7)
ax2.axvline(np.percentile(max_pp, 97.5), color='steelblue', linewidth=1.5,
            linestyle='--', alpha=0.7)
ax2.set_xlabel('Maximum Success Rate', fontsize=11)
ax2.set_ylabel('Density', fontsize=11)
ax2.set_title(f'Can Model Generate Observed Maximum?\n' +
              f'p-value = {p_value_max:.3f}',
              fontsize=12, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{PLOTS_DIR}/6_range_diagnostic.png', dpi=300, bbox_inches='tight')
plt.close()

print("  Saved: 6_range_diagnostic.png")

# ============================================================================
# PLOT 7: SUMMARY STATISTICS COMPARISON
# ============================================================================

print("Creating Plot 7: Summary Statistics Comparison...")

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

stat_names = ['mean_p', 'sd_p', 'min_p', 'max_p', 'q25_p', 'q75_p']
stat_labels = ['Mean Rate', 'SD Rate', 'Min Rate', 'Max Rate', 'Q25 Rate', 'Q75 Rate']

for i, (stat_name, stat_label) in enumerate(zip(stat_names, stat_labels)):
    ax = axes[i]

    # Histogram
    ax.hist(stats_pp[stat_name], bins=40, density=True, alpha=0.6,
            color='steelblue', edgecolor='black', label='PP')

    # Observed
    obs_val = stats_obs[stat_name]
    ax.axvline(obs_val, color='red', linewidth=2.5, label=f'Obs = {obs_val:.4f}')

    # PP intervals
    pp_lower = np.percentile(stats_pp[stat_name], 2.5)
    pp_upper = np.percentile(stats_pp[stat_name], 97.5)
    ax.axvline(pp_lower, color='steelblue', linewidth=1.5, linestyle='--', alpha=0.7)
    ax.axvline(pp_upper, color='steelblue', linewidth=1.5, linestyle='--', alpha=0.7)

    # p-value
    p_val = 2 * min(np.mean(stats_pp[stat_name] >= obs_val),
                    np.mean(stats_pp[stat_name] <= obs_val))

    ax.set_xlabel(stat_label, fontsize=10)
    ax.set_ylabel('Density', fontsize=10)
    ax.set_title(f'{stat_label}\np-value = {p_val:.3f}', fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

plt.suptitle('Summary Statistics: Observed vs Posterior Predictive',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{PLOTS_DIR}/7_summary_statistics.png', dpi=300, bbox_inches='tight')
plt.close()

print("  Saved: 7_summary_statistics.png")

print()
print("All plots created successfully!")
print()

# ============================================================================
# 5. SAVE RESULTS
# ============================================================================

print("5. SAVING RESULTS")
print("-" * 80)

# Save summary table
summary_data = {
    'Test': ['Overdispersion', 'Range (min)', 'Range (max)', 'LOO Pareto k', 'Group Fit'],
    'Result': [
        f'φ_obs = {phi_obs:.3f}, p = {p_value_phi:.3f}',
        f'p = {p_value_min:.3f}',
        f'p = {p_value_max:.3f}',
        f'{np.sum(pareto_k >= 0.7) if pareto_k is not None else "N/A"}/12 groups k ≥ 0.7',
        f'{n_concern}/12 groups p < 0.05'
    ],
    'Status': [
        overdispersion_status,
        range_status,
        range_status,
        loo_status if loo is not None else "ERROR",
        group_status
    ]
}

summary_df = pd.DataFrame(summary_data)
summary_df.to_csv(f'{DIAGNOSTICS_DIR}/ppc_summary.csv', index=False)

print(f"Saved: {DIAGNOSTICS_DIR}/ppc_summary.csv")

# Save detailed results
results = {
    'overdispersion': {
        'phi_obs': float(phi_obs),
        'phi_pp_median': float(np.median(phi_pp)),
        'phi_pp_95ci': [float(phi_pp_lower), float(phi_pp_upper)],
        'p_value': float(p_value_phi),
        'status': overdispersion_status
    },
    'range': {
        'min_obs': float(min_obs),
        'max_obs': float(max_obs),
        'min_pp_95ci': [float(np.percentile(min_pp, 2.5)), float(np.percentile(min_pp, 97.5))],
        'max_pp_95ci': [float(np.percentile(max_pp, 2.5)), float(np.percentile(max_pp, 97.5))],
        'p_value_min': float(p_value_min),
        'p_value_max': float(p_value_max),
        'status': range_status
    },
    'loo': {
        'pareto_k': pareto_k.tolist() if pareto_k is not None else None,
        'n_bad': int(np.sum(pareto_k >= 0.7)) if pareto_k is not None else None,
        'status': loo_status
    },
    'group_fit': {
        'p_values': p_values_group.tolist(),
        'n_concern': int(n_concern),
        'status': group_status
    }
}

import json
with open(f'{DIAGNOSTICS_DIR}/ppc_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"Saved: {DIAGNOSTICS_DIR}/ppc_results.json")
print()

# ============================================================================
# 6. OVERALL ASSESSMENT
# ============================================================================

print("6. OVERALL ASSESSMENT")
print("=" * 80)

# Decision logic
statuses = [overdispersion_status, range_status, loo_status, group_status]
n_pass = statuses.count('PASS')
n_investigate = statuses.count('INVESTIGATE')
n_fail = statuses.count('FAIL')

print(f"Test Results: {n_pass} PASS, {n_investigate} INVESTIGATE, {n_fail} FAIL")
print()

if n_fail > 1 or loo_status == 'FAIL':
    overall_decision = "FAIL"
elif n_fail == 1 or n_investigate > 1:
    overall_decision = "INVESTIGATE"
else:
    overall_decision = "PASS"

print(f"OVERALL DECISION: {overall_decision}")
print()

# Key findings
print("KEY FINDINGS:")
print("-" * 40)

if overdispersion_status == "PASS":
    print("✓ Model successfully captures observed overdispersion")
else:
    print("✗ Model underestimates observed overdispersion")

if loo is not None:
    exp3_bad = np.sum(pareto_k >= 0.7)
    exp1_bad = 10
    if exp3_bad < exp1_bad:
        print(f"✓ LOO diagnostics improved over Exp 1 ({exp3_bad} vs {exp1_bad} bad groups)")
    else:
        print(f"✗ LOO diagnostics no better than Exp 1 ({exp3_bad} vs {exp1_bad} bad groups)")

if group_status == "PASS":
    print("✓ All groups well-fit (no systematic misprediction)")
else:
    print(f"⚠ {n_concern} groups with poor fit (p < 0.05)")

print()
print("INTERPRETATION:")
print("-" * 40)

if overall_decision == "PASS":
    print("The Beta-Binomial model adequately captures the observed data.")
    print("Proceed to model comparison (Phase 4) to choose between:")
    print("  - Exp 1: Complex, group-specific, questionable LOO")
    print("  - Exp 3: Simple, population-level, better LOO")
elif overall_decision == "INVESTIGATE":
    print("The Beta-Binomial model shows acceptable but imperfect fit.")
    print("Consider trade-offs:")
    print("  - Simplicity and parsimony (2 vs 14 parameters)")
    print("  - Population-level vs group-specific inference")
    print("  - LOO reliability")
    print("Proceed to model comparison with documented concerns.")
else:
    print("The Beta-Binomial model is inadequate for these data.")
    print("Consider proceeding with Exp 1 only, despite LOO concerns.")

print()
print("=" * 80)
print("POSTERIOR PREDICTIVE CHECK COMPLETE")
print("=" * 80)
