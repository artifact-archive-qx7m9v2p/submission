"""
Comprehensive Posterior Predictive Checks for Hierarchical Binomial Model
Experiment 1: Assess model fit through posterior predictive distributions

Tests:
1. Overdispersion check (critical for hierarchical model validation)
2. Extreme groups check (Groups 2, 4, 8)
3. Shrinkage validation (small-n vs large-n groups)
4. Calibration check (LOO-PIT)
5. Individual group fit (Bayesian p-values)
6. LOO cross-validation (Pareto k diagnostics)
"""

import sys
sys.path.insert(0, '/tmp/agent-home/.local/lib/python3.13/site-packages')

import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import xarray as xr

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Paths
DATA_PATH = "/workspace/data/data.csv"
POSTERIOR_PATH = "/workspace/experiments/experiment_1/posterior_inference/diagnostics/posterior_inference.netcdf"
OUTPUT_DIR = "/workspace/experiments/experiment_1/posterior_predictive_check"
PLOTS_DIR = f"{OUTPUT_DIR}/plots"

# Load data
print("=" * 80)
print("POSTERIOR PREDICTIVE CHECK: Hierarchical Binomial Model")
print("=" * 80)
print()

data = pd.read_csv(DATA_PATH)
n_obs = data['n'].values
r_obs = data['r'].values
J = len(data)

print(f"Loaded data: {J} groups")
print(f"n: {n_obs}")
print(f"r: {r_obs}")
print()

# Load posterior
print("Loading posterior samples...")
idata = az.from_netcdf(POSTERIOR_PATH)
print(f"Posterior shape: {idata.posterior.dims}")
print()

# Extract posterior samples
p_samples = idata.posterior['p'].values  # Shape: (chains, draws, groups)
n_chains, n_draws, n_groups = p_samples.shape
n_samples = n_chains * n_draws

# Reshape to (samples, groups)
p_samples_flat = p_samples.reshape(-1, n_groups)

print(f"Extracted {n_samples} posterior samples for {n_groups} groups")
print()

# ============================================================================
# 1. GENERATE POSTERIOR PREDICTIVE DATA
# ============================================================================
print("=" * 80)
print("1. GENERATING POSTERIOR PREDICTIVE DATA")
print("=" * 80)
print()

print(f"Generating posterior predictive samples for each of {n_samples} posterior draws...")
r_rep = np.zeros((n_samples, n_groups), dtype=int)

for i in range(n_samples):
    for j in range(n_groups):
        r_rep[i, j] = np.random.binomial(n_obs[j], p_samples_flat[i, j])

print("Posterior predictive generation complete.")
print(f"Shape: {r_rep.shape}")
print()

# Add to InferenceData for ArviZ functions
# Create posterior_predictive group as xarray Dataset
r_rep_reshaped = r_rep.reshape(n_chains, n_draws, n_groups)
posterior_predictive = xr.Dataset(
    {
        'r_rep': xr.DataArray(
            r_rep_reshaped,
            dims=['chain', 'draw', 'group'],
            coords={
                'chain': idata.posterior.chain,
                'draw': idata.posterior.draw,
                'group': range(n_groups)
            }
        )
    }
)
idata.add_groups(posterior_predictive=posterior_predictive)

print("Added posterior predictive samples to InferenceData")
print()

# ============================================================================
# 2. TEST STATISTIC A: OVERDISPERSION CHECK
# ============================================================================
print("=" * 80)
print("2A. OVERDISPERSION CHECK (CRITICAL)")
print("=" * 80)
print()

def compute_overdispersion(r, n):
    """Compute overdispersion parameter phi = var / binomial_var"""
    p_hat = r / n
    # Binomial variance under independence
    var_binomial = np.mean(n * p_hat * (1 - p_hat))
    # Observed variance
    var_observed = np.var(r)
    phi = var_observed / var_binomial
    return phi

# Observed overdispersion
phi_obs = compute_overdispersion(r_obs, n_obs)
print(f"Observed overdispersion: phi_obs = {phi_obs:.3f}")
print(f"Expected from EDA: phi = 3.59")
print()

# Posterior predictive overdispersion
phi_rep = np.zeros(n_samples)
for i in range(n_samples):
    phi_rep[i] = compute_overdispersion(r_rep[i], n_obs)

# Bayesian p-value
p_value_phi = np.mean(phi_rep >= phi_obs)
phi_quantiles = np.percentile(phi_rep, [2.5, 50, 97.5])

print("Posterior predictive overdispersion:")
print(f"  Median: {phi_quantiles[1]:.3f}")
print(f"  95% CI: [{phi_quantiles[0]:.3f}, {phi_quantiles[2]:.3f}]")
print(f"  Bayesian p-value: {p_value_phi:.3f}")
print()

# Check pass criteria
phi_pass = (phi_obs >= phi_quantiles[0]) and (phi_obs <= phi_quantiles[2])
phi_pvalue_pass = (p_value_phi > 0.01) and (p_value_phi < 0.99)

print("OVERDISPERSION CHECK:")
if phi_pass and phi_pvalue_pass:
    print(f"  PASS: phi_obs within 95% PP interval and p-value reasonable")
else:
    if not phi_pass:
        print(f"  FAIL: phi_obs = {phi_obs:.3f} outside [{phi_quantiles[0]:.3f}, {phi_quantiles[2]:.3f}]")
    if not phi_pvalue_pass:
        print(f"  FAIL: Bayesian p-value = {p_value_phi:.3f} extreme (< 0.01 or > 0.99)")
print()

# ============================================================================
# 2. TEST STATISTIC B: EXTREME GROUPS CHECK
# ============================================================================
print("=" * 80)
print("2B. EXTREME GROUPS CHECK (Groups 2, 4, 8)")
print("=" * 80)
print()

extreme_groups = [1, 3, 7]  # 0-indexed: groups 2, 4, 8
extreme_group_labels = [2, 4, 8]

# Compute standardized residuals for each group
r_rep_mean = np.mean(r_rep, axis=0)
r_rep_std = np.std(r_rep, axis=0)
z_scores = (r_obs - r_rep_mean) / r_rep_std

print("Standardized residuals for all groups:")
for j in range(n_groups):
    marker = " <-- EXTREME" if j in extreme_groups else ""
    print(f"  Group {j+1:2d}: z = {z_scores[j]:6.3f}{marker}")
print()

# Check extreme groups
extreme_pass = True
for i, j in enumerate(extreme_groups):
    abs_z = abs(z_scores[j])
    status = "PASS" if abs_z < 3 else "FAIL"
    if abs_z >= 3:
        extreme_pass = False
    print(f"Group {extreme_group_labels[i]}: |z| = {abs_z:.3f} - {status}")
print()

print("EXTREME GROUPS CHECK:")
if extreme_pass:
    print("  PASS: All extreme groups have |z| < 3")
else:
    print("  FAIL: One or more extreme groups have |z| >= 3")
print()

# ============================================================================
# 2. TEST STATISTIC C: SHRINKAGE VALIDATION
# ============================================================================
print("=" * 80)
print("2C. SHRINKAGE VALIDATION")
print("=" * 80)
print()

# Calculate shrinkage
p_mle = r_obs / n_obs  # MLE (no pooling)
p_pooled = np.sum(r_obs) / np.sum(n_obs)  # Complete pooling
p_posterior = np.mean(p_samples_flat, axis=0)  # Partial pooling (posterior mean)

# Shrinkage: how much posterior moves from MLE toward pooled estimate
# shrinkage = (MLE - posterior) / (MLE - pooled)
shrinkage = (p_mle - p_posterior) / (p_mle - p_pooled)
shrinkage_pct = shrinkage * 100

print("Shrinkage analysis:")
print(f"{'Group':<8} {'n':<8} {'MLE':<10} {'Posterior':<10} {'Shrinkage':<12}")
print("-" * 58)

# Small-n groups (n < 100): expect 60-72% shrinkage
small_n_groups = []
for j in range(n_groups):
    if n_obs[j] < 100:
        small_n_groups.append(j)
        print(f"{j+1:<8} {n_obs[j]:<8} {p_mle[j]:.4f}    {p_posterior[j]:.4f}    {shrinkage_pct[j]:6.1f}%  (small-n)")

print()

# Large-n groups (n > 250): expect 19-30% shrinkage
large_n_groups = []
for j in range(n_groups):
    if n_obs[j] > 250:
        large_n_groups.append(j)
        print(f"{j+1:<8} {n_obs[j]:<8} {p_mle[j]:.4f}    {p_posterior[j]:.4f}    {shrinkage_pct[j]:6.1f}%  (large-n)")

print()

# Check expectations
small_n_shrinkage = [shrinkage_pct[j] for j in small_n_groups]
large_n_shrinkage = [shrinkage_pct[j] for j in large_n_groups]

print("Expected ranges:")
print(f"  Small-n (n<100): 60-72% shrinkage")
print(f"  Large-n (n>250): 19-30% shrinkage")
print()

print("Observed ranges:")
if small_n_shrinkage:
    print(f"  Small-n: {min(small_n_shrinkage):.1f}% - {max(small_n_shrinkage):.1f}%")
if large_n_shrinkage:
    print(f"  Large-n: {min(large_n_shrinkage):.1f}% - {max(large_n_shrinkage):.1f}%")
print()

# Check pass criteria (within ±20% of expected)
shrinkage_pass = True
if small_n_shrinkage:
    small_in_range = all(40 <= s <= 92 for s in small_n_shrinkage)  # 60±32, 72±20
    if not small_in_range:
        shrinkage_pass = False
if large_n_shrinkage:
    large_in_range = all(0 <= s <= 50 for s in large_n_shrinkage)  # 19-30 with ±20 margin
    if not large_in_range:
        shrinkage_pass = False

print("SHRINKAGE VALIDATION:")
if shrinkage_pass:
    print("  PASS: Observed shrinkage within expected ranges")
else:
    print("  INVESTIGATE: Some shrinkage values outside expected ranges")
print()

# ============================================================================
# 2. TEST STATISTIC D: CALIBRATION CHECK (LOO-PIT)
# ============================================================================
print("=" * 80)
print("2D. CALIBRATION CHECK (LOO-PIT)")
print("=" * 80)
print()

print("Computing LOO-PIT (Probability Integral Transform)...")
try:
    loo_pit = az.loo_pit(idata, y='y_obs', y_hat='r_rep')
    print(f"LOO-PIT computed: {len(loo_pit.loo_pit.values)} values")
    print()

    # Check uniformity
    ks_stat, ks_pvalue = stats.kstest(loo_pit.loo_pit.values, 'uniform')
    print(f"Kolmogorov-Smirnov test for uniformity:")
    print(f"  KS statistic: {ks_stat:.4f}")
    print(f"  p-value: {ks_pvalue:.4f}")
    print()

    # Check for extreme clustering
    n_near_zero = np.sum(loo_pit.loo_pit.values < 0.1)
    n_near_one = np.sum(loo_pit.loo_pit.values > 0.9)
    print(f"Extreme values:")
    print(f"  Near 0 (< 0.1): {n_near_zero}/{J}")
    print(f"  Near 1 (> 0.9): {n_near_one}/{J}")
    print()

    # Pass criteria
    pit_pass = (ks_pvalue > 0.05) and (n_near_zero < J/3) and (n_near_one < J/3)

    print("CALIBRATION CHECK:")
    if pit_pass:
        print("  PASS: LOO-PIT approximately uniform, no extreme clustering")
    else:
        print("  INVESTIGATE: LOO-PIT shows calibration issues")
    print()

    loo_pit_computed = True
except Exception as e:
    print(f"WARNING: Could not compute LOO-PIT: {e}")
    print("This may require re-running with log_likelihood in the model.")
    print()
    loo_pit_computed = False
    pit_pass = None
    ks_pvalue = None
    n_near_zero = None
    n_near_one = None

# ============================================================================
# 2. TEST STATISTIC E: INDIVIDUAL GROUP FIT
# ============================================================================
print("=" * 80)
print("2E. INDIVIDUAL GROUP FIT (Bayesian p-values)")
print("=" * 80)
print()

# Compute Bayesian p-value for each group
# p[j] = P(r_rep[j] >= r_obs[j] | data)
p_values_group = np.zeros(n_groups)
for j in range(n_groups):
    p_values_group[j] = np.mean(r_rep[:, j] >= r_obs[j])

print("Bayesian p-values by group:")
print(f"{'Group':<8} {'n':<8} {'y_obs':<8} {'p-value':<10} {'Status'}")
print("-" * 50)

extreme_pvalue_groups = []
for j in range(n_groups):
    if p_values_group[j] < 0.05 or p_values_group[j] > 0.95:
        status = "EXTREME"
        extreme_pvalue_groups.append(j+1)
    else:
        status = "OK"
    print(f"{j+1:<8} {n_obs[j]:<8} {r_obs[j]:<8} {p_values_group[j]:<10.3f} {status}")

print()

# Pass criteria
group_fit_pass = len(extreme_pvalue_groups) == 0

print("INDIVIDUAL GROUP FIT:")
if group_fit_pass:
    print("  PASS: All groups have p-values in [0.05, 0.95]")
else:
    print(f"  INVESTIGATE: {len(extreme_pvalue_groups)} groups with extreme p-values: {extreme_pvalue_groups}")
print()

# ============================================================================
# 3. LOO CROSS-VALIDATION ANALYSIS
# ============================================================================
print("=" * 80)
print("3. LOO CROSS-VALIDATION ANALYSIS")
print("=" * 80)
print()

# Compute LOO if not already in idata
if 'y_obs' in idata.log_likelihood:
    print("Computing LOO-CV...")
    loo = az.loo(idata, var_name='y_obs')
    print()
    print("LOO-CV Results:")
    print(f"  ELPD LOO: {loo.elpd_loo:.2f} ± {loo.se:.2f}")
    print(f"  p_loo: {loo.p_loo:.2f}")
    print()

    # Check Pareto k diagnostics
    pareto_k = loo.pareto_k.values
    print("Pareto k diagnostics:")
    print(f"  k < 0.5 (good): {np.sum(pareto_k < 0.5)}/{J}")
    print(f"  0.5 <= k < 0.7 (ok): {np.sum((pareto_k >= 0.5) & (pareto_k < 0.7))}/{J}")
    print(f"  0.7 <= k < 1 (bad): {np.sum((pareto_k >= 0.7) & (pareto_k < 1))}/{J}")
    print(f"  k >= 1 (very bad): {np.sum(pareto_k >= 1)}/{J}")
    print()

    # Identify high k groups
    high_k_groups = []
    print("Pareto k by group:")
    for j in range(n_groups):
        k = pareto_k[j]
        if k > 0.7:
            status = "HIGH"
            high_k_groups.append(j+1)
        elif k > 0.5:
            status = "OK"
        else:
            status = "GOOD"
        print(f"  Group {j+1:2d}: k = {k:.4f}  ({status})")
    print()

    # Pass criteria
    n_high_k = len(high_k_groups)
    loo_pass = (n_high_k == 0)
    loo_investigate = (n_high_k <= 2) and (np.max(pareto_k) < 1.4)

    print("LOO DIAGNOSTICS:")
    if loo_pass:
        print("  PASS: All Pareto k < 0.7")
    elif loo_investigate:
        print(f"  INVESTIGATE: {n_high_k} groups with k > 0.7: {high_k_groups}")
    else:
        print(f"  FAIL: {n_high_k} groups with k > 0.7 or max k >= 1.4")
    print()

    loo_computed = True
else:
    print("WARNING: log_likelihood not found in InferenceData")
    print("LOO-CV cannot be computed without log-likelihood values")
    print()
    loo_computed = False
    loo_pass = None
    pareto_k = None
    high_k_groups = []

# ============================================================================
# 4. VISUALIZATION: Overdispersion Diagnostic
# ============================================================================
print("=" * 80)
print("4. CREATING VISUALIZATIONS")
print("=" * 80)
print()

print("Plot 1: Overdispersion diagnostic...")
fig, ax = plt.subplots(figsize=(10, 6))

# Histogram of phi_rep
ax.hist(phi_rep, bins=50, alpha=0.7, density=True, color='steelblue', edgecolor='black')

# Add observed value
ax.axvline(phi_obs, color='red', linewidth=2, linestyle='--', label=f'Observed (φ = {phi_obs:.2f})')

# Add 95% interval
ax.axvline(phi_quantiles[0], color='black', linewidth=1, linestyle=':', alpha=0.5)
ax.axvline(phi_quantiles[2], color='black', linewidth=1, linestyle=':', alpha=0.5)
ax.axvspan(phi_quantiles[0], phi_quantiles[2],
           alpha=0.2, color='gray', label='95% PP Interval')

ax.set_xlabel('Overdispersion Parameter (φ)', fontsize=12)
ax.set_ylabel('Density', fontsize=12)
ax.set_title('Overdispersion Check: Can Model Reproduce Observed Variance?', fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# Add text with results
textstr = f'Bayesian p-value: {p_value_phi:.3f}\n'
textstr += f'95% PP: [{phi_quantiles[0]:.2f}, {phi_quantiles[2]:.2f}]'
ax.text(0.98, 0.97, textstr, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig(f'{PLOTS_DIR}/1_overdispersion_diagnostic.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"  Saved: {PLOTS_DIR}/1_overdispersion_diagnostic.png")

# ============================================================================
# 5. VISUALIZATION: PP Check Plot (Observed vs Replicated)
# ============================================================================
print("Plot 2: Posterior predictive check (observed vs replicated)...")

fig, ax = plt.subplots(figsize=(12, 8))

# Plot cumulative distribution of replicated data
for i in range(min(100, n_samples)):  # Plot 100 random samples
    r_sorted = np.sort(r_rep[i])
    cumprob = np.arange(1, len(r_sorted)+1) / len(r_sorted)
    ax.plot(r_sorted, cumprob, color='steelblue', alpha=0.05, linewidth=0.5)

# Plot observed data
r_obs_sorted = np.sort(r_obs)
cumprob_obs = np.arange(1, len(r_obs_sorted)+1) / len(r_obs_sorted)
ax.plot(r_obs_sorted, cumprob_obs, color='red', linewidth=2, label='Observed', marker='o', markersize=6)

ax.set_xlabel('Number of Successes (r)', fontsize=12)
ax.set_ylabel('Cumulative Probability', fontsize=12)
ax.set_title('Posterior Predictive Check: Observed vs Replicated Data',
             fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{PLOTS_DIR}/2_ppc_cumulative.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"  Saved: {PLOTS_DIR}/2_ppc_cumulative.png")

# ============================================================================
# 6. VISUALIZATION: Residual Plot
# ============================================================================
print("Plot 3: Standardized residuals...")

fig, ax = plt.subplots(figsize=(12, 6))

# Plot standardized residuals
colors = ['red' if j in extreme_groups else 'steelblue' for j in range(n_groups)]
ax.scatter(range(1, n_groups+1), z_scores, c=colors, s=100, alpha=0.7, edgecolors='black')

# Add reference lines
ax.axhline(0, color='black', linewidth=1, linestyle='-', alpha=0.5)
ax.axhline(3, color='red', linewidth=1, linestyle='--', alpha=0.5, label='|z| = 3 threshold')
ax.axhline(-3, color='red', linewidth=1, linestyle='--', alpha=0.5)
ax.fill_between(range(1, n_groups+1), -2, 2, alpha=0.2, color='green', label='Expected range (95%)')

# Labels for extreme groups
for i, j in enumerate(extreme_groups):
    ax.annotate(f'Group {extreme_group_labels[i]}',
                xy=(j+1, z_scores[j]), xytext=(10, 10),
                textcoords='offset points', fontsize=10,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

ax.set_xlabel('Group', fontsize=12)
ax.set_ylabel('Standardized Residual (z-score)', fontsize=12)
ax.set_title('Standardized Residuals: Observed vs Posterior Predictive Mean',
             fontsize=14, fontweight='bold')
ax.set_xticks(range(1, n_groups+1))
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{PLOTS_DIR}/3_standardized_residuals.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"  Saved: {PLOTS_DIR}/3_standardized_residuals.png")

# ============================================================================
# 7. VISUALIZATION: Shrinkage Validation
# ============================================================================
print("Plot 4: Shrinkage validation...")

fig, ax = plt.subplots(figsize=(12, 6))

# Plot shrinkage by group
colors_shrink = []
for j in range(n_groups):
    if n_obs[j] < 100:
        colors_shrink.append('orange')  # small-n
    elif n_obs[j] > 250:
        colors_shrink.append('purple')  # large-n
    else:
        colors_shrink.append('gray')  # medium-n

ax.scatter(range(1, n_groups+1), shrinkage_pct, c=colors_shrink, s=100, alpha=0.7, edgecolors='black')

# Add expected ranges
ax.axhspan(60, 72, alpha=0.2, color='orange', label='Expected: Small-n (60-72%)')
ax.axhspan(19, 30, alpha=0.2, color='purple', label='Expected: Large-n (19-30%)')
ax.axhline(0, color='black', linewidth=1, linestyle='-', alpha=0.5)

# Add group labels with sample sizes
for j in range(n_groups):
    ax.annotate(f'n={n_obs[j]}', xy=(j+1, shrinkage_pct[j]),
                xytext=(0, -15), textcoords='offset points',
                fontsize=8, ha='center', alpha=0.7)

ax.set_xlabel('Group', fontsize=12)
ax.set_ylabel('Shrinkage (%)', fontsize=12)
ax.set_title('Shrinkage Validation: Posterior Mean vs MLE',
             fontsize=14, fontweight='bold')
ax.set_xticks(range(1, n_groups+1))
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{PLOTS_DIR}/4_shrinkage_validation.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"  Saved: {PLOTS_DIR}/4_shrinkage_validation.png")

# ============================================================================
# 8. VISUALIZATION: LOO-PIT Histogram
# ============================================================================
if loo_pit_computed:
    print("Plot 5: LOO-PIT histogram...")

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot LOO-PIT
    az.plot_loo_pit(idata, y='y_obs', y_hat='r_rep', ax=ax)

    ax.set_title('LOO-PIT: Calibration Check (Should be Uniform)',
                 fontsize=14, fontweight='bold')

    # Add KS test result
    textstr = f'KS test p-value: {ks_pvalue:.4f}\n'
    textstr += f'Near 0: {n_near_zero}/{J}\n'
    textstr += f'Near 1: {n_near_one}/{J}'
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='left',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(f'{PLOTS_DIR}/5_loo_pit.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {PLOTS_DIR}/5_loo_pit.png")
else:
    print("Plot 5: LOO-PIT - SKIPPED (not computed)")

# ============================================================================
# 9. VISUALIZATION: Pareto k Diagnostic
# ============================================================================
if loo_computed and pareto_k is not None:
    print("Plot 6: Pareto k diagnostic...")

    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot Pareto k by group
    colors_k = []
    for k in pareto_k:
        if k < 0.5:
            colors_k.append('green')
        elif k < 0.7:
            colors_k.append('yellow')
        elif k < 1:
            colors_k.append('orange')
        else:
            colors_k.append('red')

    ax.scatter(range(1, n_groups+1), pareto_k, c=colors_k, s=100, alpha=0.7, edgecolors='black')

    # Add threshold lines
    ax.axhline(0.5, color='green', linewidth=1, linestyle='--', alpha=0.5, label='k = 0.5 (good)')
    ax.axhline(0.7, color='orange', linewidth=1, linestyle='--', alpha=0.5, label='k = 0.7 (ok)')
    ax.axhline(1.0, color='red', linewidth=1, linestyle='--', alpha=0.5, label='k = 1.0 (bad)')

    # Annotate high k groups
    for j in range(n_groups):
        if pareto_k[j] > 0.7:
            ax.annotate(f'Group {j+1}',
                        xy=(j+1, pareto_k[j]), xytext=(10, 10),
                        textcoords='offset points', fontsize=10,
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

    ax.set_xlabel('Group', fontsize=12)
    ax.set_ylabel('Pareto k', fontsize=12)
    ax.set_title('Pareto k Diagnostic: LOO-CV Reliability',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(range(1, n_groups+1))
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{PLOTS_DIR}/6_pareto_k.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {PLOTS_DIR}/6_pareto_k.png")
else:
    print("Plot 6: Pareto k - SKIPPED (not computed)")

# ============================================================================
# 10. VISUALIZATION: Group-level PP Checks (Small Multiples)
# ============================================================================
print("Plot 7: Group-level posterior predictive checks...")

fig, axes = plt.subplots(3, 4, figsize=(16, 12))
axes = axes.flatten()

for j in range(n_groups):
    ax = axes[j]

    # Histogram of replicated data for this group
    ax.hist(r_rep[:, j], bins=30, alpha=0.7, density=True, color='steelblue', edgecolor='black')

    # Add observed value
    ax.axvline(r_obs[j], color='red', linewidth=2, linestyle='--', label='Observed')

    # Add mean and 95% interval
    r_mean = np.mean(r_rep[:, j])
    r_lower, r_upper = np.percentile(r_rep[:, j], [2.5, 97.5])
    ax.axvline(r_mean, color='black', linewidth=1, linestyle='-', alpha=0.5, label='PP Mean')
    ax.axvspan(r_lower, r_upper, alpha=0.2, color='gray', label='95% PP')

    # Title and labels
    ax.set_title(f'Group {j+1} (n={n_obs[j]}, r={r_obs[j]})', fontsize=10, fontweight='bold')
    ax.set_xlabel('r', fontsize=9)
    ax.set_ylabel('Density', fontsize=9)

    # Add p-value
    textstr = f'p = {p_values_group[j]:.3f}\nz = {z_scores[j]:.2f}'
    ax.text(0.98, 0.97, textstr, transform=ax.transAxes, fontsize=8,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax.grid(True, alpha=0.3)
    if j == 0:
        ax.legend(fontsize=7, loc='upper left')

plt.suptitle('Group-Level Posterior Predictive Checks', fontsize=16, fontweight='bold', y=1.00)
plt.tight_layout()
plt.savefig(f'{PLOTS_DIR}/7_group_level_ppc.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"  Saved: {PLOTS_DIR}/7_group_level_ppc.png")

# ============================================================================
# 11. VISUALIZATION: Observed vs Predicted Scatter
# ============================================================================
print("Plot 8: Observed vs predicted scatter...")

fig, ax = plt.subplots(figsize=(10, 10))

# Posterior predictive mean and intervals
r_pred_mean = np.mean(r_rep, axis=0)
r_pred_lower = np.percentile(r_rep, 2.5, axis=0)
r_pred_upper = np.percentile(r_rep, 97.5, axis=0)

# Scatter plot with error bars
for j in range(n_groups):
    color = 'red' if j in extreme_groups else 'steelblue'
    ax.errorbar(r_obs[j], r_pred_mean[j],
                yerr=[[r_pred_mean[j] - r_pred_lower[j]], [r_pred_upper[j] - r_pred_mean[j]]],
                fmt='o', color=color, markersize=8, capsize=5, alpha=0.7)

    # Label extreme groups
    if j in extreme_groups:
        ax.annotate(f'Group {j+1}', xy=(r_obs[j], r_pred_mean[j]),
                    xytext=(10, 10), textcoords='offset points', fontsize=10,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

# Add diagonal line
min_val = min(r_obs.min(), r_pred_mean.min())
max_val = max(r_obs.max(), r_pred_mean.max())
ax.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2, alpha=0.5, label='Perfect fit')

ax.set_xlabel('Observed Successes (r)', fontsize=12)
ax.set_ylabel('Predicted Successes (posterior mean)', fontsize=12)
ax.set_title('Observed vs Predicted: Model Fit Assessment', fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_aspect('equal')

plt.tight_layout()
plt.savefig(f'{PLOTS_DIR}/8_observed_vs_predicted.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"  Saved: {PLOTS_DIR}/8_observed_vs_predicted.png")

print()
print("All visualizations complete!")
print()

# ============================================================================
# SUMMARY AND DECISION
# ============================================================================
print("=" * 80)
print("FINAL ASSESSMENT")
print("=" * 80)
print()

# Collect all test results
tests = {
    'Overdispersion Check': phi_pass and phi_pvalue_pass,
    'Extreme Groups Check': extreme_pass,
    'Shrinkage Validation': shrinkage_pass,
    'Calibration Check (LOO-PIT)': pit_pass if loo_pit_computed else None,
    'Individual Group Fit': group_fit_pass,
    'LOO Diagnostics': loo_pass if loo_computed else None
}

print("Test Results:")
print("-" * 50)
for test_name, result in tests.items():
    if result is None:
        status = "NOT COMPUTED"
    elif result:
        status = "PASS"
    else:
        status = "FAIL/INVESTIGATE"
    print(f"  {test_name:<35} {status}")
print()

# Overall decision
n_pass = sum(1 for r in tests.values() if r is True)
n_fail = sum(1 for r in tests.values() if r is False)
n_total = sum(1 for r in tests.values() if r is not None)

print("Overall Decision:")
print(f"  Passed: {n_pass}/{n_total} tests")
print(f"  Failed/Investigate: {n_fail}/{n_total} tests")
print()

if n_fail == 0:
    decision = "PASS"
    recommendation = "Proceed to model critique (likely ACCEPT)"
elif n_fail <= 2:
    decision = "INVESTIGATE"
    recommendation = "Proceed to model critique with documented concerns"
else:
    decision = "FAIL"
    recommendation = "Try Experiment 2 (Robust Student-t) or 3 (Beta-binomial)"

print(f"DECISION: {decision}")
print(f"RECOMMENDATION: {recommendation}")
print()

# Save summary
summary = {
    'phi_obs': phi_obs,
    'phi_median': phi_quantiles[1],
    'phi_ci_lower': phi_quantiles[0],
    'phi_ci_upper': phi_quantiles[2],
    'phi_pvalue': p_value_phi,
    'phi_pass': phi_pass and phi_pvalue_pass,
    'extreme_groups_pass': extreme_pass,
    'shrinkage_pass': shrinkage_pass,
    'pit_pass': pit_pass if loo_pit_computed else None,
    'group_fit_pass': group_fit_pass,
    'loo_pass': loo_pass if loo_computed else None,
    'decision': decision,
    'recommendation': recommendation
}

summary_df = pd.DataFrame([summary])
summary_df.to_csv(f'{OUTPUT_DIR}/ppc_summary.csv', index=False)
print(f"Summary saved to: {OUTPUT_DIR}/ppc_summary.csv")

print()
print("=" * 80)
print("POSTERIOR PREDICTIVE CHECK COMPLETE")
print("=" * 80)
