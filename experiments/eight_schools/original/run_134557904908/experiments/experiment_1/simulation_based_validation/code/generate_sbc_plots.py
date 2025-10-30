"""
Generate comprehensive visualizations for SBC results.

This script creates all diagnostic plots to assess:
1. Rank uniformity (primary SBC diagnostic)
2. Coverage calibration
3. Parameter recovery and bias
4. Uncertainty calibration
5. Z-score distribution
6. Computational diagnostics
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path
import json

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Paths
OUTPUT_DIR = Path("/workspace/experiments/experiment_1/simulation_based_validation")
PLOTS_DIR = OUTPUT_DIR / "plots"
CODE_DIR = OUTPUT_DIR / "code"

# Load results
results_df = pd.read_csv(CODE_DIR / "sbc_results.csv")
with open(CODE_DIR / "sbc_summary.json", "r") as f:
    summary = json.load(f)

N_SIMS = len(results_df)
N_POSTERIOR_SAMPLES = 2000  # N_SAMPLES * N_CHAINS

print("="*80)
print("GENERATING SBC DIAGNOSTIC PLOTS")
print("="*80)
print(f"\nLoaded {N_SIMS} simulation results")

# ============================================================================
# PLOT 1: RANK HISTOGRAM (PRIMARY SBC DIAGNOSTIC)
# ============================================================================

print("\n1. Generating rank histogram...")

fig, ax = plt.subplots(figsize=(12, 7))

ranks = results_df["rank"].values
n_bins = 20
bin_edges = np.linspace(0, N_POSTERIOR_SAMPLES, n_bins + 1)

# Plot histogram
counts, _, patches = ax.hist(ranks, bins=bin_edges, edgecolor='black',
                             alpha=0.7, color='steelblue', label='Observed ranks')

# Expected uniform count
expected_count = N_SIMS / n_bins
ax.axhline(expected_count, color='red', linestyle='--', lw=2,
          label=f'Expected (uniform): {expected_count:.1f}')

# Add confidence band for uniform distribution (95%)
# Using binomial approximation
se = np.sqrt(N_SIMS * (1/n_bins) * (1 - 1/n_bins))
lower = expected_count - 1.96 * se
upper = expected_count + 1.96 * se
ax.axhspan(lower, upper, alpha=0.2, color='red',
          label=f'95% CI: [{lower:.1f}, {upper:.1f}]')

# Styling
ax.set_xlabel('Rank of θ_true in posterior samples', fontsize=12)
ax.set_ylabel('Frequency', fontsize=12)
ax.set_title('SBC Rank Histogram (Primary Calibration Diagnostic)',
            fontsize=14, fontweight='bold')
ax.legend(loc='upper right', fontsize=11)
ax.grid(True, alpha=0.3, axis='y')

# Add statistical test results
chi2_stat = summary["rank_statistics"]["chi2_stat"]
chi2_pval = summary["rank_statistics"]["chi2_pval"]
ks_stat = summary["rank_statistics"]["ks_stat"]
ks_pval = summary["rank_statistics"]["ks_pval"]

test_text = f"""
Statistical Tests:
  χ² = {chi2_stat:.2f}, p = {chi2_pval:.4f}
  KS = {ks_stat:.4f}, p = {ks_pval:.4f}

Result: {'PASS' if chi2_pval > 0.05 else 'FAIL'}
"""
ax.text(0.02, 0.98, test_text, transform=ax.transAxes,
       verticalalignment='top', fontsize=10, family='monospace',
       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.tight_layout()
plt.savefig(PLOTS_DIR / "rank_histogram.png", dpi=300, bbox_inches='tight')
print(f"   Saved: rank_histogram.png")
plt.close()

# ============================================================================
# PLOT 2: ECDF OF RANKS VS UNIFORM
# ============================================================================

print("2. Generating rank ECDF...")

fig, ax = plt.subplots(figsize=(10, 7))

# Empirical CDF
sorted_ranks = np.sort(ranks)
ecdf_values = np.arange(1, len(sorted_ranks) + 1) / len(sorted_ranks)

# Theoretical uniform CDF
uniform_cdf = sorted_ranks / N_POSTERIOR_SAMPLES

# Plot
ax.plot(sorted_ranks, ecdf_values, 'b-', lw=2, label='Empirical CDF')
ax.plot(sorted_ranks, uniform_cdf, 'r--', lw=2, label='Uniform CDF')
ax.plot([0, N_POSTERIOR_SAMPLES], [0, 1], 'k:', lw=1, alpha=0.5)

# Add confidence band
# Using Dvoretzky–Kiefer–Wolfowitz inequality
alpha = 0.05
epsilon = np.sqrt(np.log(2/alpha) / (2 * N_SIMS))
ax.fill_between(sorted_ranks,
               np.clip(uniform_cdf - epsilon, 0, 1),
               np.clip(uniform_cdf + epsilon, 0, 1),
               alpha=0.2, color='red', label=f'95% CB (ε={epsilon:.3f})')

ax.set_xlabel('Rank', fontsize=12)
ax.set_ylabel('Cumulative Probability', fontsize=12)
ax.set_title('ECDF of Ranks: Uniformity Assessment', fontsize=14, fontweight='bold')
ax.legend(loc='upper left', fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_xlim(0, N_POSTERIOR_SAMPLES)
ax.set_ylim(0, 1)

# Add KS statistic
max_diff = np.max(np.abs(ecdf_values - uniform_cdf))
ax.text(0.98, 0.02, f'Max |ECDF - Uniform| = {max_diff:.4f}\nKS p-value = {ks_pval:.4f}',
       transform=ax.transAxes, horizontalalignment='right',
       verticalalignment='bottom', fontsize=10,
       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout()
plt.savefig(PLOTS_DIR / "rank_ecdf.png", dpi=300, bbox_inches='tight')
print(f"   Saved: rank_ecdf.png")
plt.close()

# ============================================================================
# PLOT 3: COVERAGE CALIBRATION
# ============================================================================

print("3. Generating coverage calibration plot...")

fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Panel A: Coverage by nominal level
ax = axes[0]

nominal_levels = np.array([50, 90, 95])
observed_coverage = np.array([
    summary["coverage"]["coverage_50"],
    summary["coverage"]["coverage_90"],
    summary["coverage"]["coverage_95"]
])

# Plot perfect calibration line
ax.plot([0, 100], [0, 100], 'k--', lw=2, label='Perfect calibration')
ax.fill_between([0, 100], [0, 100], [0, 100],
               where=np.array([True, True]),
               alpha=0.1, color='green',
               transform=ax.transData)

# Add tolerance bands
for level in nominal_levels:
    ax.axhspan(level - 5, level + 5, alpha=0.1, color='yellow')

# Plot observed coverage
ax.plot(nominal_levels, observed_coverage, 'o-', markersize=12,
       lw=2, color='steelblue', label='Observed coverage')

# Annotate points
for nom, obs in zip(nominal_levels, observed_coverage):
    diff = obs - nom
    color = 'green' if abs(diff) < 5 else 'red'
    ax.annotate(f'{obs:.1f}%\n(Δ={diff:+.1f}%)',
               xy=(nom, obs), xytext=(10, 0),
               textcoords='offset points',
               fontsize=9, color=color, fontweight='bold')

ax.set_xlabel('Nominal Coverage (%)', fontsize=12)
ax.set_ylabel('Observed Coverage (%)', fontsize=12)
ax.set_title('Coverage Calibration', fontsize=13, fontweight='bold')
ax.legend(loc='upper left', fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xlim(40, 100)
ax.set_ylim(40, 100)

# Panel B: Coverage by credible interval width
ax = axes[1]

ci_widths = results_df["ci_95_upper"] - results_df["ci_95_lower"]
covered = results_df["coverage_95"].values

# Bin by width
width_bins = pd.qcut(ci_widths, q=10, duplicates='drop')
coverage_by_width = results_df.groupby(width_bins)["coverage_95"].mean() * 100
width_centers = results_df.groupby(width_bins)["ci_95_upper"].apply(
    lambda x: (x - results_df.loc[x.index, "ci_95_lower"]).mean()
)

ax.plot(width_centers, coverage_by_width, 'o-', markersize=8,
       lw=2, color='steelblue', label='Observed coverage')
ax.axhline(95, color='red', linestyle='--', lw=2, label='Nominal 95%')
ax.axhspan(90, 100, alpha=0.1, color='green', label='±5% tolerance')

ax.set_xlabel('95% CI Width', fontsize=12)
ax.set_ylabel('Observed Coverage (%)', fontsize=12)
ax.set_title('Coverage by Interval Width', fontsize=13, fontweight='bold')
ax.legend(loc='lower right', fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(PLOTS_DIR / "coverage_calibration.png", dpi=300, bbox_inches='tight')
print(f"   Saved: coverage_calibration.png")
plt.close()

# ============================================================================
# PLOT 4: PARAMETER RECOVERY (θ_true vs θ̂)
# ============================================================================

print("4. Generating parameter recovery plot...")

fig, axes = plt.subplots(1, 2, figsize=(15, 7))

theta_true = results_df["theta_true"].values
theta_mean = results_df["theta_mean"].values
theta_sd = results_df["theta_sd"].values

# Panel A: Scatter plot with 45-degree line
ax = axes[0]

# Hexbin for density
hb = ax.hexbin(theta_true, theta_mean, gridsize=30, cmap='Blues', mincnt=1)
cb = plt.colorbar(hb, ax=ax, label='Count')

# Perfect recovery line
min_val = min(theta_true.min(), theta_mean.min())
max_val = max(theta_true.max(), theta_mean.max())
ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2,
       label='Perfect recovery', zorder=10)

# Add regression line
slope = summary["correlation"]["slope"]
intercept = summary["correlation"]["intercept"]
x_line = np.array([min_val, max_val])
y_line = intercept + slope * x_line
ax.plot(x_line, y_line, 'g-', lw=2,
       label=f'Fit: θ̂ = {intercept:.3f} + {slope:.3f}·θ', zorder=10)

ax.set_xlabel('θ_true', fontsize=12)
ax.set_ylabel('θ̂ (posterior mean)', fontsize=12)
ax.set_title('Parameter Recovery', fontsize=13, fontweight='bold')
ax.legend(loc='upper left', fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_aspect('equal')

# Add statistics
r2 = summary["correlation"]["r_squared"]
bias = summary["bias"]["mean_bias"]
stats_text = f"""
R² = {r2:.4f}
Bias = {bias:.4f}
Slope = {slope:.4f}
"""
ax.text(0.98, 0.02, stats_text, transform=ax.transAxes,
       horizontalalignment='right', verticalalignment='bottom',
       fontsize=10, family='monospace',
       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

# Panel B: Residuals vs true values
ax = axes[1]

residuals = theta_mean - theta_true

# Scatter with error bars
ax.scatter(theta_true, residuals, alpha=0.3, s=30, color='steelblue')

# Add zero line
ax.axhline(0, color='red', linestyle='--', lw=2, label='No bias')

# Add ±1 SD band
residual_sd = np.std(residuals)
ax.axhspan(-residual_sd, residual_sd, alpha=0.2, color='green',
          label=f'±1 SD ({residual_sd:.2f})')

# Running mean
bins = pd.qcut(theta_true, q=10, duplicates='drop')
residual_means = results_df.groupby(bins)["bias"].mean()
theta_means = results_df.groupby(bins)["theta_true"].mean()
ax.plot(theta_means, residual_means, 'ro-', lw=2, markersize=8,
       label='Binned mean')

ax.set_xlabel('θ_true', fontsize=12)
ax.set_ylabel('Residual (θ̂ - θ_true)', fontsize=12)
ax.set_title('Recovery Residuals', fontsize=13, fontweight='bold')
ax.legend(loc='upper right', fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(PLOTS_DIR / "parameter_recovery.png", dpi=300, bbox_inches='tight')
print(f"   Saved: parameter_recovery.png")
plt.close()

# ============================================================================
# PLOT 5: Z-SCORE DISTRIBUTION
# ============================================================================

print("5. Generating z-score distribution...")

fig, axes = plt.subplots(1, 2, figsize=(15, 6))

z_scores = results_df["z_score"].values

# Panel A: Histogram with N(0,1) overlay
ax = axes[0]

ax.hist(z_scores, bins=50, density=True, alpha=0.7,
       edgecolor='black', color='steelblue', label='Observed z-scores')

# Overlay theoretical N(0,1)
x = np.linspace(-4, 4, 200)
ax.plot(x, stats.norm.pdf(x, 0, 1), 'r-', lw=2,
       label='N(0, 1)')

# Mark ±2 SD
ax.axvline(-2, color='orange', linestyle='--', lw=1.5, alpha=0.7)
ax.axvline(2, color='orange', linestyle='--', lw=1.5, alpha=0.7)
ax.axvspan(-2, 2, alpha=0.1, color='green', label='95% expected')

ax.set_xlabel('Z-score: (θ_true - θ̂) / SD(θ̂)', fontsize=12)
ax.set_ylabel('Density', fontsize=12)
ax.set_title('Z-score Distribution', fontsize=13, fontweight='bold')
ax.legend(loc='upper right', fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xlim(-4, 4)

# Add statistics
z_mean = summary["uncertainty"]["z_mean"]
z_sd = summary["uncertainty"]["z_sd"]
shapiro_pval = summary["uncertainty"]["shapiro_pval"]

stats_text = f"""
Mean: {z_mean:.4f}
SD: {z_sd:.4f}

Shapiro-Wilk:
p = {shapiro_pval:.4f}
"""
ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
       verticalalignment='top', fontsize=10, family='monospace',
       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

# Panel B: Q-Q plot
ax = axes[1]

stats.probplot(z_scores, dist="norm", plot=ax)
ax.get_lines()[0].set_markerfacecolor('steelblue')
ax.get_lines()[0].set_markeredgecolor('black')
ax.get_lines()[0].set_markersize(5)
ax.get_lines()[0].set_alpha(0.6)
ax.get_lines()[1].set_color('red')
ax.get_lines()[1].set_linewidth(2)

ax.set_title('Q-Q Plot: Z-scores vs Normal', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(PLOTS_DIR / "z_score_calibration.png", dpi=300, bbox_inches='tight')
print(f"   Saved: z_score_calibration.png")
plt.close()

# ============================================================================
# PLOT 6: UNCERTAINTY CALIBRATION
# ============================================================================

print("6. Generating uncertainty calibration plot...")

fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Panel A: Posterior SD vs empirical SD
ax = axes[0]

# Bin by true value
bins = pd.qcut(theta_true, q=10, duplicates='drop')
posterior_sds = results_df.groupby(bins)["theta_sd"].mean()
empirical_sds = results_df.groupby(bins).apply(
    lambda x: x["bias"].std()
)
theta_bin_centers = results_df.groupby(bins)["theta_true"].mean()

# Scatter plot
ax.scatter(theta_bin_centers, posterior_sds, s=100, alpha=0.7,
          color='blue', edgecolor='black', linewidth=1.5,
          label='Mean posterior SD')
ax.scatter(theta_bin_centers, empirical_sds, s=100, alpha=0.7,
          color='red', edgecolor='black', linewidth=1.5,
          label='Empirical SD')

# Connect points
for tc, ps, es in zip(theta_bin_centers, posterior_sds, empirical_sds):
    ax.plot([tc, tc], [ps, es], 'k-', lw=1, alpha=0.3)

# Overall means
overall_posterior_sd = summary["uncertainty"]["posterior_sd"]
overall_empirical_sd = summary["uncertainty"]["empirical_sd"]
ax.axhline(overall_posterior_sd, color='blue', linestyle='--', lw=2,
          alpha=0.5, label=f'Overall posterior: {overall_posterior_sd:.3f}')
ax.axhline(overall_empirical_sd, color='red', linestyle='--', lw=2,
          alpha=0.5, label=f'Overall empirical: {overall_empirical_sd:.3f}')

ax.set_xlabel('θ_true (binned)', fontsize=12)
ax.set_ylabel('Standard Deviation', fontsize=12)
ax.set_title('Uncertainty Calibration by True Value', fontsize=13, fontweight='bold')
ax.legend(loc='best', fontsize=9)
ax.grid(True, alpha=0.3)

# Panel B: Coverage vs posterior SD
ax = axes[1]

# Bin by posterior SD
sd_bins = pd.qcut(results_df["theta_sd"], q=10, duplicates='drop')
coverage_by_sd = results_df.groupby(sd_bins)["coverage_95"].mean() * 100
sd_centers = results_df.groupby(sd_bins)["theta_sd"].mean()

ax.plot(sd_centers, coverage_by_sd, 'o-', markersize=10,
       lw=2, color='steelblue', label='Observed coverage')
ax.axhline(95, color='red', linestyle='--', lw=2, label='Nominal 95%')
ax.axhspan(90, 100, alpha=0.1, color='green', label='±5% tolerance')

ax.set_xlabel('Posterior SD (binned)', fontsize=12)
ax.set_ylabel('95% CI Coverage (%)', fontsize=12)
ax.set_title('Coverage by Posterior Uncertainty', fontsize=13, fontweight='bold')
ax.legend(loc='best', fontsize=10)
ax.grid(True, alpha=0.3)

# Add ratio
ratio = summary["uncertainty"]["sd_ratio"]
ax.text(0.02, 0.02, f'SD Ratio = {ratio:.4f}\n(posterior/empirical)',
       transform=ax.transAxes, verticalalignment='bottom',
       fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.tight_layout()
plt.savefig(PLOTS_DIR / "uncertainty_calibration.png", dpi=300, bbox_inches='tight')
print(f"   Saved: uncertainty_calibration.png")
plt.close()

# ============================================================================
# PLOT 7: COMPUTATIONAL DIAGNOSTICS
# ============================================================================

print("7. Generating computational diagnostics...")

fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Panel A: R-hat distribution
ax = axes[0, 0]

rhat_values = results_df["rhat"].values
ax.hist(rhat_values, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
ax.axvline(1.0, color='green', linestyle='--', lw=2, label='Perfect (1.0)')
ax.axvline(1.01, color='orange', linestyle='--', lw=2, label='Threshold (1.01)')

ax.set_xlabel('R-hat', fontsize=12)
ax.set_ylabel('Frequency', fontsize=12)
ax.set_title('Convergence: R-hat Distribution', fontsize=13, fontweight='bold')
ax.legend(loc='upper right', fontsize=10)
ax.grid(True, alpha=0.3, axis='y')

# Add statistics
mean_rhat = summary["computational"]["mean_rhat"]
max_rhat = summary["computational"]["max_rhat"]
pct_converged = summary["computational"]["pct_converged"]

stats_text = f"""
Mean: {mean_rhat:.6f}
Max: {max_rhat:.6f}
% < 1.01: {pct_converged:.1f}%
"""
ax.text(0.98, 0.98, stats_text, transform=ax.transAxes,
       horizontalalignment='right', verticalalignment='top',
       fontsize=10, family='monospace',
       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

# Panel B: ESS distributions
ax = axes[0, 1]

ess_bulk = results_df["ess_bulk"].values
ess_tail = results_df["ess_tail"].values

ax.hist(ess_bulk, bins=50, alpha=0.6, edgecolor='black',
       color='blue', label='Bulk ESS')
ax.hist(ess_tail, bins=50, alpha=0.6, edgecolor='black',
       color='red', label='Tail ESS')
ax.axvline(400, color='orange', linestyle='--', lw=2,
          label='Adequate (400)')

ax.set_xlabel('Effective Sample Size', fontsize=12)
ax.set_ylabel('Frequency', fontsize=12)
ax.set_title('Effective Sample Size Distribution', fontsize=13, fontweight='bold')
ax.legend(loc='upper left', fontsize=10)
ax.grid(True, alpha=0.3, axis='y')

# Add statistics
mean_bulk = summary["computational"]["mean_ess_bulk"]
min_bulk = summary["computational"]["min_ess_bulk"]
pct_adequate = summary["computational"]["pct_adequate_ess"]

stats_text = f"""
Bulk ESS:
  Mean: {mean_bulk:.0f}
  Min: {min_bulk:.0f}

% adequate: {pct_adequate:.1f}%
"""
ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
       verticalalignment='top', fontsize=10, family='monospace',
       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

# Panel C: R-hat vs ESS
ax = axes[1, 0]

ax.scatter(rhat_values, ess_bulk, alpha=0.5, s=30, color='blue',
          label='Bulk ESS')
ax.scatter(rhat_values, ess_tail, alpha=0.5, s=30, color='red',
          label='Tail ESS')

ax.axvline(1.01, color='orange', linestyle='--', lw=1.5, alpha=0.7)
ax.axhline(400, color='orange', linestyle='--', lw=1.5, alpha=0.7)

# Highlight problematic region
ax.axvspan(1.01, ax.get_xlim()[1], alpha=0.1, color='red')
ax.axhspan(0, 400, alpha=0.1, color='red')

ax.set_xlabel('R-hat', fontsize=12)
ax.set_ylabel('ESS', fontsize=12)
ax.set_title('Convergence vs Sample Size', fontsize=13, fontweight='bold')
ax.legend(loc='upper right', fontsize=10)
ax.grid(True, alpha=0.3)

# Panel D: Simulation success rate
ax = axes[1, 1]

n_total = summary["n_simulations"]
n_success = summary["n_successful"]
n_converged = int(pct_converged * n_success / 100)
n_adequate_ess = int(pct_adequate * n_success / 100)
n_no_divergences = n_success - summary["computational"]["n_sims_with_divergences"]

categories = ['Completed', 'Converged\n(R̂<1.01)', 'Adequate ESS\n(>400)', 'No Divergences']
counts = [n_success, n_converged, n_adequate_ess, n_no_divergences]
percentages = [c/n_total*100 for c in counts]

bars = ax.bar(categories, percentages, color=['steelblue', 'green', 'orange', 'purple'],
             alpha=0.7, edgecolor='black', linewidth=1.5)

# Add percentage labels
for bar, pct, cnt in zip(bars, percentages, counts):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
           f'{pct:.1f}%\n({cnt}/{n_total})',
           ha='center', va='bottom', fontsize=10, fontweight='bold')

ax.axhline(100, color='red', linestyle='--', lw=2, alpha=0.5, label='Target: 100%')
ax.set_ylabel('Success Rate (%)', fontsize=12)
ax.set_title('Computational Performance Summary', fontsize=13, fontweight='bold')
ax.set_ylim(0, 110)
ax.legend(loc='lower right', fontsize=10)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(PLOTS_DIR / "computational_diagnostics.png", dpi=300, bbox_inches='tight')
print(f"   Saved: computational_diagnostics.png")
plt.close()

# ============================================================================
# PLOT 8: STRATIFIED ANALYSIS
# ============================================================================

print("8. Generating stratified analysis...")

fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Create magnitude categories
results_df["magnitude_category"] = pd.cut(
    np.abs(results_df["theta_true"]),
    bins=[0, 5, 15, np.inf],
    labels=["Small\n(|θ| < 5)", "Medium\n(5 ≤ |θ| < 15)", "Large\n(|θ| ≥ 15)"]
)

categories = ["Small\n(|θ| < 5)", "Medium\n(5 ≤ |θ| < 15)", "Large\n(|θ| ≥ 15)"]
colors_cat = ['green', 'orange', 'red']

# Panel A: Bias by category
ax = axes[0, 0]

bias_by_cat = [results_df[results_df["magnitude_category"] == cat]["bias"].values
              for cat in categories]

bp = ax.boxplot(bias_by_cat, labels=categories, patch_artist=True,
               widths=0.6, showmeans=True)

for patch, color in zip(bp['boxes'], colors_cat):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)

ax.axhline(0, color='red', linestyle='--', lw=2, label='No bias')
ax.set_ylabel('Bias (θ̂ - θ_true)', fontsize=12)
ax.set_title('Bias by Parameter Magnitude', fontsize=13, fontweight='bold')
ax.legend(loc='upper right', fontsize=10)
ax.grid(True, alpha=0.3, axis='y')

# Panel B: Coverage by category
ax = axes[0, 1]

coverage_50 = [results_df[results_df["magnitude_category"] == cat]["coverage_50"].mean() * 100
              for cat in categories]
coverage_90 = [results_df[results_df["magnitude_category"] == cat]["coverage_90"].mean() * 100
              for cat in categories]
coverage_95 = [results_df[results_df["magnitude_category"] == cat]["coverage_95"].mean() * 100
              for cat in categories]

x = np.arange(len(categories))
width = 0.25

ax.bar(x - width, coverage_50, width, label='50% CI', alpha=0.7,
      color='lightblue', edgecolor='black')
ax.bar(x, coverage_90, width, label='90% CI', alpha=0.7,
      color='steelblue', edgecolor='black')
ax.bar(x + width, coverage_95, width, label='95% CI', alpha=0.7,
      color='darkblue', edgecolor='black')

# Add nominal lines
ax.axhline(50, color='lightblue', linestyle='--', lw=1.5, alpha=0.5)
ax.axhline(90, color='steelblue', linestyle='--', lw=1.5, alpha=0.5)
ax.axhline(95, color='darkblue', linestyle='--', lw=1.5, alpha=0.5)

ax.set_ylabel('Coverage (%)', fontsize=12)
ax.set_title('Coverage by Parameter Magnitude', fontsize=13, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(categories)
ax.legend(loc='lower right', fontsize=10)
ax.grid(True, alpha=0.3, axis='y')
ax.set_ylim(0, 100)

# Panel C: R-hat by category
ax = axes[1, 0]

rhat_by_cat = [results_df[results_df["magnitude_category"] == cat]["rhat"].values
              for cat in categories]

bp = ax.boxplot(rhat_by_cat, labels=categories, patch_artist=True,
               widths=0.6, showmeans=True)

for patch, color in zip(bp['boxes'], colors_cat):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)

ax.axhline(1.0, color='green', linestyle='--', lw=2, label='Perfect (1.0)')
ax.axhline(1.01, color='orange', linestyle='--', lw=2, label='Threshold (1.01)')
ax.set_ylabel('R-hat', fontsize=12)
ax.set_title('Convergence by Parameter Magnitude', fontsize=13, fontweight='bold')
ax.legend(loc='upper right', fontsize=10)
ax.grid(True, alpha=0.3, axis='y')

# Panel D: Sample sizes by category
ax = axes[1, 1]

counts = [len(results_df[results_df["magnitude_category"] == cat]) for cat in categories]

bars = ax.bar(categories, counts, color=colors_cat, alpha=0.7,
             edgecolor='black', linewidth=1.5)

# Add count labels
for bar, count in zip(bars, counts):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
           f'{count}\n({count/N_SIMS*100:.1f}%)',
           ha='center', va='bottom', fontsize=11, fontweight='bold')

ax.set_ylabel('Number of Simulations', fontsize=12)
ax.set_title('Sample Distribution by Parameter Magnitude', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(PLOTS_DIR / "stratified_analysis.png", dpi=300, bbox_inches='tight')
print(f"   Saved: stratified_analysis.png")
plt.close()

# ============================================================================
# PLOT 9: COMPREHENSIVE SUMMARY DASHBOARD
# ============================================================================

print("9. Generating summary dashboard...")

fig = plt.figure(figsize=(18, 12))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# Panel 1: Rank histogram (compact)
ax1 = fig.add_subplot(gs[0, :2])
ax1.hist(ranks, bins=20, edgecolor='black', alpha=0.7, color='steelblue')
ax1.axhline(N_SIMS/20, color='red', linestyle='--', lw=2)
se = np.sqrt(N_SIMS * (1/20) * (1 - 1/20))
ax1.axhspan(N_SIMS/20 - 1.96*se, N_SIMS/20 + 1.96*se, alpha=0.2, color='red')
ax1.set_xlabel('Rank', fontsize=10)
ax1.set_ylabel('Frequency', fontsize=10)
ax1.set_title('A. Rank Uniformity', fontsize=11, fontweight='bold')
ax1.grid(True, alpha=0.3, axis='y')
ax1.text(0.98, 0.98, f"χ² p={chi2_pval:.4f}", transform=ax1.transAxes,
        ha='right', va='top', fontsize=9,
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

# Panel 2: Pass/Fail Summary
ax2 = fig.add_subplot(gs[0, 2])
ax2.axis('off')

checks = summary["checks"]
n_passed = sum(checks.values())
n_total = len(checks)

check_summary = "VALIDATION CHECKS\n" + "="*30 + "\n\n"
for check_name, passed in checks.items():
    status = "✓" if passed else "✗"
    check_summary += f"{status} {check_name}\n"

check_summary += f"\n{'='*30}\n"
check_summary += f"PASSED: {n_passed}/{n_total}\n"
check_summary += f"\nOVERALL: {'PASS' if summary['overall_pass'] else 'FAIL'}"

ax2.text(0.05, 0.95, check_summary, transform=ax2.transAxes,
        verticalalignment='top', fontsize=8, family='monospace',
        bbox=dict(boxstyle='round',
                 facecolor='lightgreen' if summary['overall_pass'] else 'lightcoral',
                 alpha=0.8))

# Panel 3: Parameter recovery
ax3 = fig.add_subplot(gs[1, 0])
ax3.hexbin(theta_true, theta_mean, gridsize=20, cmap='Blues', mincnt=1)
min_val = min(theta_true.min(), theta_mean.min())
max_val = max(theta_true.max(), theta_mean.max())
ax3.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
ax3.set_xlabel('θ_true', fontsize=10)
ax3.set_ylabel('θ̂', fontsize=10)
ax3.set_title('B. Parameter Recovery', fontsize=11, fontweight='bold')
ax3.set_aspect('equal')
ax3.grid(True, alpha=0.3)
ax3.text(0.02, 0.98, f"R²={summary['correlation']['r_squared']:.4f}",
        transform=ax3.transAxes, va='top', fontsize=9,
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

# Panel 4: Coverage
ax4 = fig.add_subplot(gs[1, 1])
nominal = [50, 90, 95]
observed = [summary["coverage"]["coverage_50"],
           summary["coverage"]["coverage_90"],
           summary["coverage"]["coverage_95"]]
ax4.plot([0, 100], [0, 100], 'k--', lw=2)
ax4.plot(nominal, observed, 'o-', markersize=10, lw=2, color='steelblue')
for nom, obs in zip(nominal, observed):
    ax4.annotate(f'{obs:.1f}%', xy=(nom, obs), xytext=(5, 5),
                textcoords='offset points', fontsize=8)
ax4.set_xlabel('Nominal (%)', fontsize=10)
ax4.set_ylabel('Observed (%)', fontsize=10)
ax4.set_title('C. Coverage Calibration', fontsize=11, fontweight='bold')
ax4.grid(True, alpha=0.3)
ax4.set_xlim(40, 100)
ax4.set_ylim(40, 100)

# Panel 5: Z-scores
ax5 = fig.add_subplot(gs[1, 2])
ax5.hist(z_scores, bins=40, density=True, alpha=0.7,
        edgecolor='black', color='steelblue')
x = np.linspace(-4, 4, 200)
ax5.plot(x, stats.norm.pdf(x, 0, 1), 'r-', lw=2)
ax5.axvspan(-2, 2, alpha=0.1, color='green')
ax5.set_xlabel('Z-score', fontsize=10)
ax5.set_ylabel('Density', fontsize=10)
ax5.set_title('D. Z-score Distribution', fontsize=11, fontweight='bold')
ax5.set_xlim(-4, 4)
ax5.grid(True, alpha=0.3)
ax5.text(0.02, 0.98, f"μ={z_mean:.3f}, σ={z_sd:.3f}",
        transform=ax5.transAxes, va='top', fontsize=9,
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

# Panel 6: R-hat
ax6 = fig.add_subplot(gs[2, 0])
ax6.hist(rhat_values, bins=40, edgecolor='black', alpha=0.7, color='steelblue')
ax6.axvline(1.01, color='orange', linestyle='--', lw=2)
ax6.set_xlabel('R-hat', fontsize=10)
ax6.set_ylabel('Frequency', fontsize=10)
ax6.set_title('E. Convergence Diagnostic', fontsize=11, fontweight='bold')
ax6.grid(True, alpha=0.3, axis='y')
ax6.text(0.98, 0.98, f"{pct_converged:.1f}% < 1.01",
        transform=ax6.transAxes, ha='right', va='top', fontsize=9,
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

# Panel 7: ESS
ax7 = fig.add_subplot(gs[2, 1])
ax7.hist(ess_bulk, bins=40, alpha=0.7, edgecolor='black',
        color='steelblue', label='Bulk')
ax7.axvline(400, color='orange', linestyle='--', lw=2)
ax7.set_xlabel('Bulk ESS', fontsize=10)
ax7.set_ylabel('Frequency', fontsize=10)
ax7.set_title('F. Effective Sample Size', fontsize=11, fontweight='bold')
ax7.grid(True, alpha=0.3, axis='y')
ax7.text(0.02, 0.98, f"{pct_adequate:.1f}% > 400",
        transform=ax7.transAxes, va='top', fontsize=9,
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

# Panel 8: Summary statistics table
ax8 = fig.add_subplot(gs[2, 2])
ax8.axis('off')

summary_text = f"""KEY METRICS
{'='*25}

Rank Statistics:
  χ² p-value: {chi2_pval:.4f}
  KS p-value: {ks_pval:.4f}

Coverage (95% CI):
  Observed: {summary['coverage']['coverage_95']:.1f}%

Bias:
  Mean: {bias:.4f}

Recovery:
  R²: {r_squared:.4f}
  Slope: {slope:.4f}

Uncertainty:
  SD ratio: {sd_ratio:.4f}

Computational:
  Converged: {pct_converged:.1f}%
  Adequate ESS: {pct_adequate:.1f}%
"""

ax8.text(0.05, 0.95, summary_text, transform=ax8.transAxes,
        verticalalignment='top', fontsize=8, family='monospace',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

overall_text = f"\n\n\nOVERALL RESULT:\n{'PASS' if summary['overall_pass'] else 'FAIL'}"
color = 'lightgreen' if summary['overall_pass'] else 'lightcoral'
ax8.text(0.5, 0.15, overall_text, transform=ax8.transAxes,
        ha='center', va='center', fontsize=12, fontweight='bold',
        bbox=dict(boxstyle='round', facecolor=color, alpha=0.9))

plt.suptitle('SIMULATION-BASED CALIBRATION: Summary Dashboard',
            fontsize=16, fontweight='bold', y=0.995)

plt.savefig(PLOTS_DIR / "sbc_summary_dashboard.png", dpi=300, bbox_inches='tight')
print(f"   Saved: sbc_summary_dashboard.png")
plt.close()

print("\n" + "="*80)
print(f"COMPLETE: Generated {len(list(PLOTS_DIR.glob('*.png')))} diagnostic plots")
print(f"All plots saved to: {PLOTS_DIR}")
print("="*80)
