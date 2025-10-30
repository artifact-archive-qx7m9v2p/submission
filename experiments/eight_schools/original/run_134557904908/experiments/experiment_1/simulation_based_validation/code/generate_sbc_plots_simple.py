"""
Generate SBC diagnostic plots (simplified version for analytical posterior).
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
N_POSTERIOR_SAMPLES = 2000

print("="*80)
print("GENERATING SBC DIAGNOSTIC PLOTS")
print("="*80)
print(f"\nLoaded {N_SIMS} simulation results")

# Extract data
ranks = results_df["rank"].values
theta_true = results_df["theta_true"].values
theta_mean = results_df["theta_mean"].values
theta_sd = results_df["theta_sd"].values
z_scores = results_df["z_score"].values
bias = results_df["bias"].values

chi2_pval = summary["rank_statistics"]["chi2_pval"]
ks_pval = summary["rank_statistics"]["ks_pval"]
coverage_50 = summary["coverage"]["coverage_50"]
coverage_90 = summary["coverage"]["coverage_90"]
coverage_95 = summary["coverage"]["coverage_95"]
r_squared = summary["correlation"]["r_squared"]
slope = summary["correlation"]["slope"]
intercept = summary["correlation"]["intercept"]
sd_ratio = summary["uncertainty"]["sd_ratio"]
z_mean = summary["uncertainty"]["z_mean"]
z_sd = summary["uncertainty"]["z_sd"]
overall_pass = summary["overall_pass"]

# ============================================================================
# COMPREHENSIVE SUMMARY DASHBOARD (Main Figure)
# ============================================================================

print("\nGenerating comprehensive summary dashboard...")

fig = plt.figure(figsize=(20, 14))
gs = fig.add_gridspec(3, 4, hspace=0.35, wspace=0.35)

# ===== Panel 1: Rank histogram =====
ax1 = fig.add_subplot(gs[0, :2])
n_bins = 20
bin_edges = np.linspace(0, N_POSTERIOR_SAMPLES, n_bins + 1)
counts, _, patches = ax1.hist(ranks, bins=bin_edges, edgecolor='black',
                             alpha=0.7, color='steelblue')
expected_count = N_SIMS / n_bins
ax1.axhline(expected_count, color='red', linestyle='--', lw=2,
          label=f'Expected (uniform): {expected_count:.1f}')
se = np.sqrt(N_SIMS * (1/n_bins) * (1 - 1/n_bins))
ax1.axhspan(expected_count - 1.96*se, expected_count + 1.96*se,
          alpha=0.2, color='red', label='95% CI')
ax1.set_xlabel('Rank of θ_true in posterior samples', fontsize=11)
ax1.set_ylabel('Frequency', fontsize=11)
ax1.set_title('A. Rank Histogram (Primary SBC Diagnostic)', fontsize=13, fontweight='bold')
ax1.legend(loc='upper right', fontsize=10)
ax1.grid(True, alpha=0.3, axis='y')
ax1.text(0.02, 0.98, f"χ² p={chi2_pval:.4f}\nKS p={ks_pval:.4f}",
        transform=ax1.transAxes, va='top', fontsize=9,
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

# ===== Panel 2: ECDF of ranks =====
ax2 = fig.add_subplot(gs[0, 2:])
sorted_ranks = np.sort(ranks)
ecdf_values = np.arange(1, len(sorted_ranks) + 1) / len(sorted_ranks)
uniform_cdf = sorted_ranks / N_POSTERIOR_SAMPLES
ax2.plot(sorted_ranks, ecdf_values, 'b-', lw=2, label='Empirical CDF')
ax2.plot(sorted_ranks, uniform_cdf, 'r--', lw=2, label='Uniform CDF')
ax2.plot([0, N_POSTERIOR_SAMPLES], [0, 1], 'k:', lw=1, alpha=0.5)
alpha = 0.05
epsilon = np.sqrt(np.log(2/alpha) / (2 * N_SIMS))
ax2.fill_between(sorted_ranks,
               np.clip(uniform_cdf - epsilon, 0, 1),
               np.clip(uniform_cdf + epsilon, 0, 1),
               alpha=0.2, color='red', label=f'95% CB')
ax2.set_xlabel('Rank', fontsize=11)
ax2.set_ylabel('Cumulative Probability', fontsize=11)
ax2.set_title('B. ECDF of Ranks', fontsize=13, fontweight='bold')
ax2.legend(loc='upper left', fontsize=10)
ax2.grid(True, alpha=0.3)
ax2.set_xlim(0, N_POSTERIOR_SAMPLES)
ax2.set_ylim(0, 1)

# ===== Panel 3: Parameter recovery =====
ax3 = fig.add_subplot(gs[1, 0])
hb = ax3.hexbin(theta_true, theta_mean, gridsize=25, cmap='Blues', mincnt=1)
min_val = min(theta_true.min(), theta_mean.min())
max_val = max(theta_true.max(), theta_mean.max())
ax3.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect recovery')
x_line = np.array([min_val, max_val])
y_line = intercept + slope * x_line
ax3.plot(x_line, y_line, 'g-', lw=2, alpha=0.7, label=f'Fit: y={intercept:.2f}+{slope:.3f}x')
ax3.set_xlabel('θ_true', fontsize=11)
ax3.set_ylabel('θ̂ (posterior mean)', fontsize=11)
ax3.set_title('C. Parameter Recovery', fontsize=13, fontweight='bold')
ax3.legend(loc='upper left', fontsize=9)
ax3.set_aspect('equal')
ax3.grid(True, alpha=0.3)
ax3.text(0.98, 0.02, f"R²={r_squared:.4f}", transform=ax3.transAxes,
        ha='right', va='bottom', fontsize=9,
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

# ===== Panel 4: Coverage calibration =====
ax4 = fig.add_subplot(gs[1, 1])
nominal = [50, 90, 95]
observed = [coverage_50, coverage_90, coverage_95]
ax4.plot([0, 100], [0, 100], 'k--', lw=2, label='Perfect calibration')
for level in nominal:
    ax4.axhspan(level - 5, level + 5, alpha=0.1, color='yellow')
ax4.plot(nominal, observed, 'o-', markersize=12, lw=2, color='steelblue',
        label='Observed')
for nom, obs in zip(nominal, observed):
    diff = obs - nom
    color = 'green' if abs(diff) < 5 else 'red'
    ax4.annotate(f'{obs:.1f}%', xy=(nom, obs), xytext=(8, 0),
                textcoords='offset points', fontsize=9,
                color=color, fontweight='bold')
ax4.set_xlabel('Nominal Coverage (%)', fontsize=11)
ax4.set_ylabel('Observed Coverage (%)', fontsize=11)
ax4.set_title('D. Coverage Calibration', fontsize=13, fontweight='bold')
ax4.legend(loc='upper left', fontsize=9)
ax4.grid(True, alpha=0.3)
ax4.set_xlim(40, 100)
ax4.set_ylim(40, 100)

# ===== Panel 5: Z-score distribution =====
ax5 = fig.add_subplot(gs[1, 2])
ax5.hist(z_scores, bins=40, density=True, alpha=0.7,
        edgecolor='black', color='steelblue', label='Observed')
x = np.linspace(-4, 4, 200)
ax5.plot(x, stats.norm.pdf(x, 0, 1), 'r-', lw=2, label='N(0,1)')
ax5.axvspan(-2, 2, alpha=0.1, color='green', label='95% expected')
ax5.set_xlabel('Z-score', fontsize=11)
ax5.set_ylabel('Density', fontsize=11)
ax5.set_title('E. Z-score Distribution', fontsize=13, fontweight='bold')
ax5.set_xlim(-4, 4)
ax5.legend(loc='upper right', fontsize=9)
ax5.grid(True, alpha=0.3)
ax5.text(0.02, 0.98, f"μ={z_mean:.3f}\nσ={z_sd:.3f}",
        transform=ax5.transAxes, va='top', fontsize=9,
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

# ===== Panel 6: Q-Q plot =====
ax6 = fig.add_subplot(gs[1, 3])
stats.probplot(z_scores, dist="norm", plot=ax6)
ax6.get_lines()[0].set_markerfacecolor('steelblue')
ax6.get_lines()[0].set_markeredgecolor('black')
ax6.get_lines()[0].set_markersize(5)
ax6.get_lines()[0].set_alpha(0.6)
ax6.get_lines()[1].set_color('red')
ax6.get_lines()[1].set_linewidth(2)
ax6.set_title('F. Q-Q Plot (Z-scores)', fontsize=13, fontweight='bold')
ax6.grid(True, alpha=0.3)

# ===== Panel 7: Residuals plot =====
ax7 = fig.add_subplot(gs[2, 0])
residuals = theta_mean - theta_true
ax7.scatter(theta_true, residuals, alpha=0.3, s=30, color='steelblue')
ax7.axhline(0, color='red', linestyle='--', lw=2, label='No bias')
residual_sd = np.std(residuals)
ax7.axhspan(-residual_sd, residual_sd, alpha=0.2, color='green',
          label=f'±1 SD ({residual_sd:.2f})')
# Binned mean
bins = pd.qcut(theta_true, q=10, duplicates='drop')
residual_means = results_df.groupby(bins, observed=False)["bias"].mean()
theta_means = results_df.groupby(bins, observed=False)["theta_true"].mean()
ax7.plot(theta_means, residual_means, 'ro-', lw=2, markersize=8, label='Binned mean')
ax7.set_xlabel('θ_true', fontsize=11)
ax7.set_ylabel('Residual (θ̂ - θ_true)', fontsize=11)
ax7.set_title('G. Recovery Residuals', fontsize=13, fontweight='bold')
ax7.legend(loc='upper right', fontsize=9)
ax7.grid(True, alpha=0.3)

# ===== Panel 8: Uncertainty calibration =====
ax8 = fig.add_subplot(gs[2, 1])
posterior_sd_mean = summary["uncertainty"]["posterior_sd"]
empirical_sd = summary["uncertainty"]["empirical_sd"]
categories = ['Posterior\nSD', 'Empirical\nSD']
values = [posterior_sd_mean, empirical_sd]
colors = ['blue', 'red']
bars = ax8.bar(categories, values, color=colors, alpha=0.7,
             edgecolor='black', linewidth=1.5)
for bar, val in zip(bars, values):
    height = bar.get_height()
    ax8.text(bar.get_x() + bar.get_width()/2., height,
           f'{val:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
ax8.set_ylabel('Standard Deviation', fontsize=11)
ax8.set_title('H. Uncertainty Calibration', fontsize=13, fontweight='bold')
ax8.grid(True, alpha=0.3, axis='y')
ax8.text(0.5, 0.98, f"Ratio: {sd_ratio:.4f}",
        transform=ax8.transAxes, ha='center', va='top', fontsize=10,
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

# ===== Panel 9: Stratified analysis =====
ax9 = fig.add_subplot(gs[2, 2])
results_df["magnitude_category"] = pd.cut(
    np.abs(results_df["theta_true"]),
    bins=[0, 5, 15, np.inf],
    labels=["Small\n(|θ|<5)", "Medium\n(5≤|θ|<15)", "Large\n(|θ|≥15)"]
)
categories = ["Small\n(|θ|<5)", "Medium\n(5≤|θ|<15)", "Large\n(|θ|≥15)"]
coverage_by_cat = [
    results_df[results_df["magnitude_category"] == cat]["coverage_95"].mean() * 100
    for cat in categories
]
colors_cat = ['green', 'orange', 'red']
bars = ax9.bar(categories, coverage_by_cat, color=colors_cat, alpha=0.7,
             edgecolor='black', linewidth=1.5)
ax9.axhline(95, color='blue', linestyle='--', lw=2, label='Nominal 95%')
ax9.axhspan(90, 100, alpha=0.1, color='green')
for bar, cov in zip(bars, coverage_by_cat):
    height = bar.get_height()
    ax9.text(bar.get_x() + bar.get_width()/2., height,
           f'{cov:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
ax9.set_ylabel('95% CI Coverage (%)', fontsize=11)
ax9.set_title('I. Coverage by Parameter Range', fontsize=13, fontweight='bold')
ax9.legend(loc='lower right', fontsize=9)
ax9.grid(True, alpha=0.3, axis='y')
ax9.set_ylim(80, 105)

# ===== Panel 10: Pass/Fail Summary =====
ax10 = fig.add_subplot(gs[2, 3])
ax10.axis('off')

checks = summary["checks"]
n_passed = sum(checks.values())
n_total = len(checks)

check_text = "VALIDATION CHECKS\n" + "="*32 + "\n\n"
for check_name, passed in checks.items():
    status = "PASS" if passed else "FAIL"
    symbol = "✓" if passed else "✗"
    check_text += f"{symbol} {check_name}\n"

check_text += f"\n{'='*32}\n"
check_text += f"PASSED: {n_passed}/{n_total}\n"
check_text += f"\nOVERALL: {'PASS' if overall_pass else 'FAIL'}"

ax10.text(0.05, 0.95, check_text, transform=ax10.transAxes,
        verticalalignment='top', fontsize=8, family='monospace',
        bbox=dict(boxstyle='round',
                 facecolor='lightgreen' if overall_pass else 'lightcoral',
                 alpha=0.9))

plt.suptitle('SIMULATION-BASED CALIBRATION: Comprehensive Summary',
            fontsize=18, fontweight='bold', y=0.995)

plt.savefig(PLOTS_DIR / "sbc_comprehensive_summary.png", dpi=300, bbox_inches='tight')
print("   Saved: sbc_comprehensive_summary.png")
plt.close()

# ============================================================================
# Additional detailed plots
# ============================================================================

# Plot: Coverage by interval width
print("\nGenerating coverage by interval width...")
fig, ax = plt.subplots(figsize=(10, 7))
ci_widths = results_df["ci_95_upper"] - results_df["ci_95_lower"]
width_bins = pd.qcut(ci_widths, q=10, duplicates='drop')
coverage_by_width = results_df.groupby(width_bins, observed=False)["coverage_95"].mean() * 100
width_centers = results_df.groupby(width_bins, observed=False).apply(
    lambda x: (x["ci_95_upper"] - x["ci_95_lower"]).mean()
)
ax.plot(width_centers, coverage_by_width, 'o-', markersize=10,
       lw=2, color='steelblue', label='Observed coverage')
ax.axhline(95, color='red', linestyle='--', lw=2, label='Nominal 95%')
ax.axhspan(90, 100, alpha=0.1, color='green', label='±5% tolerance')
ax.set_xlabel('95% CI Width', fontsize=12)
ax.set_ylabel('Observed Coverage (%)', fontsize=12)
ax.set_title('Coverage Calibration by Interval Width', fontsize=14, fontweight='bold')
ax.legend(loc='best', fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "coverage_by_width.png", dpi=300, bbox_inches='tight')
print("   Saved: coverage_by_width.png")
plt.close()

# Plot: Stratified bias analysis
print("\nGenerating stratified bias analysis...")
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Bias by magnitude
ax = axes[0]
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

# Sample counts by magnitude
ax = axes[1]
counts = [len(results_df[results_df["magnitude_category"] == cat]) for cat in categories]
bars = ax.bar(categories, counts, color=colors_cat, alpha=0.7,
             edgecolor='black', linewidth=1.5)
for bar, count in zip(bars, counts):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
           f'{count}\n({count/N_SIMS*100:.1f}%)',
           ha='center', va='bottom', fontsize=11, fontweight='bold')
ax.set_ylabel('Number of Simulations', fontsize=12)
ax.set_title('Distribution by Parameter Magnitude', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(PLOTS_DIR / "stratified_analysis.png", dpi=300, bbox_inches='tight')
print("   Saved: stratified_analysis.png")
plt.close()

print("\n" + "="*80)
print(f"COMPLETE: Generated all diagnostic plots")
print(f"All plots saved to: {PLOTS_DIR}")
print("="*80)
