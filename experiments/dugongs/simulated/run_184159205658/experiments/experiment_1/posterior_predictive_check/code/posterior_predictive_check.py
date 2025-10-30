"""
Comprehensive Posterior Predictive Checks for Logarithmic Regression Model
Experiment 1: Y = β₀ + β₁·log(x) + ε

This script performs extensive validation to assess whether the fitted model
can reproduce key features of the observed data.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import arviz as az
from scipy import stats
from scipy.stats import shapiro, kstest
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# File paths
DATA_PATH = '/workspace/data/data.csv'
IDATA_PATH = '/workspace/experiments/experiment_1/posterior_inference/diagnostics/posterior_inference.netcdf'
RESIDUALS_PATH = '/workspace/experiments/experiment_1/posterior_inference/diagnostics/residuals.csv'
OUTPUT_DIR = '/workspace/experiments/experiment_1/posterior_predictive_check'

print("=" * 80)
print("POSTERIOR PREDICTIVE CHECKS - LOGARITHMIC REGRESSION MODEL")
print("=" * 80)

# Load data
print("\n1. Loading data...")
data = pd.read_csv(DATA_PATH)
residuals_df = pd.read_csv(RESIDUALS_PATH)
idata = az.from_netcdf(IDATA_PATH)

print(f"   Observations: {len(data)}")
print(f"   x range: [{data['x'].min():.1f}, {data['x'].max():.1f}]")
print(f"   Y range: [{data['Y'].min():.3f}, {data['Y'].max():.3f}]")

# Extract posterior samples
print("\n2. Extracting posterior samples...")
beta0_samples = idata.posterior['beta0'].values.flatten()
beta1_samples = idata.posterior['beta1'].values.flatten()
sigma_samples = idata.posterior['sigma'].values.flatten()
n_samples = len(beta0_samples)
print(f"   Posterior samples: {n_samples}")

# Extract existing posterior predictive samples
print("\n3. Extracting posterior predictive samples...")
x_obs = data['x'].values
y_obs = data['Y'].values
n_obs = len(x_obs)

# Get posterior predictive from idata (shape: chains x draws x obs)
y_rep_full = idata.posterior_predictive['Y'].values
# Reshape to (n_draws, n_obs)
y_rep = y_rep_full.reshape(-1, n_obs)
n_ppc_samples = y_rep.shape[0]

print(f"   Posterior predictive samples: {n_ppc_samples}")
print(f"   Shape: {y_rep.shape}")

# ============================================================================
# VISUAL POSTERIOR PREDICTIVE CHECKS
# ============================================================================
print("\n" + "=" * 80)
print("VISUAL POSTERIOR PREDICTIVE CHECKS")
print("=" * 80)

# Plot 1: PPC Overlays - Multiple posterior predictive draws vs observed data
print("\n4. Creating PPC overlay plots...")
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Panel A: Posterior predictive draws overlaid on data
ax = axes[0, 0]
# Plot 50 random draws
draw_indices = np.random.choice(n_ppc_samples, size=min(50, n_ppc_samples), replace=False)
for i in draw_indices:
    ax.plot(x_obs, y_rep[i, :], 'o-', alpha=0.1, color='steelblue', markersize=3)
ax.plot(x_obs, y_obs, 'o', color='darkred', markersize=8, label='Observed', zorder=10)
ax.set_xlabel('x', fontsize=12)
ax.set_ylabel('Y', fontsize=12)
ax.set_title('A. Posterior Predictive Draws vs Observed Data', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(alpha=0.3)

# Panel B: Density overlay
ax = axes[0, 1]
ax.hist(y_obs, bins=15, density=True, alpha=0.6, color='darkred',
        label='Observed', edgecolor='black')
for i in np.random.choice(n_ppc_samples, size=min(100, n_ppc_samples), replace=False):
    ax.hist(y_rep[i, :], bins=15, density=True, alpha=0.02, color='steelblue')
ax.hist(y_rep[0, :], bins=15, density=True, alpha=0.02, color='steelblue',
        label='Predicted (100 draws)')
ax.set_xlabel('Y', fontsize=12)
ax.set_ylabel('Density', fontsize=12)
ax.set_title('B. Distribution: Observed vs Predicted', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(alpha=0.3)

# Panel C: Pointwise predictive intervals
ax = axes[1, 0]
# Sort by x for clean plotting
sort_idx = np.argsort(x_obs)
x_sorted = x_obs[sort_idx]
y_sorted = y_obs[sort_idx]
y_rep_sorted = y_rep[:, sort_idx]

# Compute predictive intervals
lower_50 = np.percentile(y_rep_sorted, 25, axis=0)
upper_50 = np.percentile(y_rep_sorted, 75, axis=0)
lower_95 = np.percentile(y_rep_sorted, 2.5, axis=0)
upper_95 = np.percentile(y_rep_sorted, 97.5, axis=0)
median_pred = np.median(y_rep_sorted, axis=0)

ax.fill_between(x_sorted, lower_95, upper_95, alpha=0.3, color='steelblue',
                label='95% Predictive Interval')
ax.fill_between(x_sorted, lower_50, upper_50, alpha=0.5, color='steelblue',
                label='50% Predictive Interval')
ax.plot(x_sorted, median_pred, '-', color='navy', linewidth=2, label='Median Prediction')
ax.plot(x_sorted, y_sorted, 'o', color='darkred', markersize=8, label='Observed', zorder=10)
ax.set_xlabel('x', fontsize=12)
ax.set_ylabel('Y', fontsize=12)
ax.set_title('C. Predictive Intervals and Coverage', fontsize=13, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

# Panel D: Residuals from median prediction
ax = axes[1, 1]
residuals_ppc = y_sorted - median_pred
ax.axhline(0, color='black', linestyle='--', linewidth=1)
ax.scatter(x_sorted, residuals_ppc, s=80, alpha=0.7, color='darkred', edgecolor='black')
ax.set_xlabel('x', fontsize=12)
ax.set_ylabel('Residual', fontsize=12)
ax.set_title('D. Residuals from Median Prediction', fontsize=13, fontweight='bold')
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/plots/ppc_overlays.png', dpi=300, bbox_inches='tight')
print(f"   Saved: {OUTPUT_DIR}/plots/ppc_overlays.png")
plt.close()

# ============================================================================
# QUANTITATIVE CALIBRATION CHECKS
# ============================================================================
print("\n" + "=" * 80)
print("QUANTITATIVE CALIBRATION CHECKS")
print("=" * 80)

print("\n5. Computing test statistics on observed and replicated data...")

# Test statistics to evaluate
def compute_test_statistics(y):
    """Compute various test statistics for a dataset"""
    return {
        'mean': np.mean(y),
        'std': np.std(y, ddof=1),
        'min': np.min(y),
        'max': np.max(y),
        'range': np.max(y) - np.min(y),
        'q25': np.percentile(y, 25),
        'median': np.median(y),
        'q75': np.percentile(y, 75),
        'skewness': stats.skew(y),
        'kurtosis': stats.kurtosis(y)
    }

# Observed statistics
obs_stats = compute_test_statistics(y_obs)

# Replicated statistics
rep_stats = {key: [] for key in obs_stats.keys()}
for i in range(n_ppc_samples):
    stats_i = compute_test_statistics(y_rep[i, :])
    for key in rep_stats:
        rep_stats[key].append(stats_i[key])

# Convert to arrays
for key in rep_stats:
    rep_stats[key] = np.array(rep_stats[key])

# Compute posterior predictive p-values
print("\n   Posterior Predictive P-values:")
print("   " + "-" * 50)
pp_pvalues = {}
for key in obs_stats:
    p_value = np.mean(rep_stats[key] >= obs_stats[key])
    pp_pvalues[key] = p_value
    status = "GOOD" if 0.05 <= p_value <= 0.95 else "EXTREME"
    print(f"   {key:12s}: {p_value:.3f} [{status}]")

# Coverage analysis
print("\n6. Computing predictive interval coverage...")
in_50 = np.sum((y_obs >= np.percentile(y_rep, 25, axis=0)) &
               (y_obs <= np.percentile(y_rep, 75, axis=0)))
in_95 = np.sum((y_obs >= np.percentile(y_rep, 2.5, axis=0)) &
               (y_obs <= np.percentile(y_rep, 97.5, axis=0)))

coverage_50 = 100 * in_50 / n_obs
coverage_95 = 100 * in_95 / n_obs

print(f"   50% Interval Coverage: {coverage_50:.1f}% (target: ~50%)")
print(f"   95% Interval Coverage: {coverage_95:.1f}% (target: 90-98%)")

# Assessment
if coverage_95 >= 90:
    coverage_status = "GOOD"
elif coverage_95 >= 85:
    coverage_status = "ACCEPTABLE"
else:
    coverage_status = "POOR"
print(f"   Coverage Assessment: {coverage_status}")

# ============================================================================
# PLOT: TEST STATISTICS COMPARISON
# ============================================================================
print("\n7. Creating test statistics comparison plots...")
fig, axes = plt.subplots(2, 5, figsize=(18, 8))
axes = axes.flatten()

stats_order = ['mean', 'std', 'min', 'max', 'range', 'q25', 'median', 'q75', 'skewness', 'kurtosis']

for idx, stat_name in enumerate(stats_order):
    ax = axes[idx]
    ax.hist(rep_stats[stat_name], bins=30, density=True, alpha=0.6,
            color='steelblue', edgecolor='black')
    ax.axvline(obs_stats[stat_name], color='darkred', linewidth=3,
               label='Observed', linestyle='--')
    ax.set_xlabel(stat_name.capitalize(), fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title(f'p = {pp_pvalues[stat_name]:.3f}', fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

plt.suptitle('Test Statistics: Observed vs Posterior Predictive Distribution',
             fontsize=14, fontweight='bold', y=1.00)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/plots/ppc_statistics.png', dpi=300, bbox_inches='tight')
print(f"   Saved: {OUTPUT_DIR}/plots/ppc_statistics.png")
plt.close()

# ============================================================================
# RESIDUAL ANALYSIS
# ============================================================================
print("\n" + "=" * 80)
print("RESIDUAL ANALYSIS")
print("=" * 80)

print("\n8. Analyzing residuals from fitted model...")
residuals = residuals_df['residual'].values
fitted_values = residuals_df['mu_mean'].values

# Normality tests
shapiro_stat, shapiro_p = shapiro(residuals)
ks_stat, ks_p = kstest(residuals, 'norm', args=(np.mean(residuals), np.std(residuals)))

print(f"\n   Normality Tests:")
print(f"   Shapiro-Wilk: W = {shapiro_stat:.4f}, p = {shapiro_p:.4f}")
print(f"   Kolmogorov-Smirnov: D = {ks_stat:.4f}, p = {ks_p:.4f}")
if shapiro_p > 0.05:
    print(f"   Assessment: Residuals appear NORMAL (p > 0.05)")
else:
    print(f"   Assessment: Residuals show deviation from normality (p < 0.05)")

# Independence test (Durbin-Watson)
residuals_sorted = residuals[np.argsort(x_obs)]
dw_stat = np.sum(np.diff(residuals_sorted)**2) / np.sum(residuals_sorted**2)
print(f"\n   Independence Test:")
print(f"   Durbin-Watson: {dw_stat:.4f} (target: ~2.0)")
if 1.5 <= dw_stat <= 2.5:
    print(f"   Assessment: No strong autocorrelation detected")
else:
    print(f"   Assessment: Potential autocorrelation present")

# Homoscedasticity test
from scipy.stats import pearsonr
abs_residuals = np.abs(residuals)
corr_fitted, p_fitted = pearsonr(fitted_values, abs_residuals)
corr_x, p_x = pearsonr(x_obs, abs_residuals)

print(f"\n   Homoscedasticity Tests:")
print(f"   Correlation(|residuals|, fitted): r = {corr_fitted:.4f}, p = {p_fitted:.4f}")
print(f"   Correlation(|residuals|, x): r = {corr_x:.4f}, p = {p_x:.4f}")
if p_fitted > 0.05 and p_x > 0.05:
    print(f"   Assessment: Homoscedasticity assumption SATISFIED")
else:
    print(f"   Assessment: Potential heteroscedasticity detected")

# ============================================================================
# PLOT: COMPREHENSIVE RESIDUAL DIAGNOSTICS
# ============================================================================
print("\n9. Creating comprehensive residual diagnostic plots...")
fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# Panel 1: Residuals vs Fitted
ax1 = fig.add_subplot(gs[0, 0])
ax1.scatter(fitted_values, residuals, s=80, alpha=0.7, color='darkred', edgecolor='black')
ax1.axhline(0, color='black', linestyle='--', linewidth=1)
z = np.polyfit(fitted_values, residuals, 1)
p = np.poly1d(z)
ax1.plot(sorted(fitted_values), p(sorted(fitted_values)), "b--", alpha=0.8, linewidth=2)
ax1.set_xlabel('Fitted Values', fontsize=11)
ax1.set_ylabel('Residuals', fontsize=11)
ax1.set_title('Residuals vs Fitted Values', fontsize=12, fontweight='bold')
ax1.grid(alpha=0.3)

# Panel 2: Residuals vs x
ax2 = fig.add_subplot(gs[0, 1])
ax2.scatter(x_obs, residuals, s=80, alpha=0.7, color='darkred', edgecolor='black')
ax2.axhline(0, color='black', linestyle='--', linewidth=1)
z = np.polyfit(x_obs, residuals, 1)
p = np.poly1d(z)
ax2.plot(sorted(x_obs), p(sorted(x_obs)), "b--", alpha=0.8, linewidth=2)
ax2.set_xlabel('x', fontsize=11)
ax2.set_ylabel('Residuals', fontsize=11)
ax2.set_title('Residuals vs x', fontsize=12, fontweight='bold')
ax2.grid(alpha=0.3)

# Panel 3: Histogram of residuals
ax3 = fig.add_subplot(gs[0, 2])
ax3.hist(residuals, bins=12, density=True, alpha=0.6, color='darkred', edgecolor='black')
x_range = np.linspace(residuals.min(), residuals.max(), 100)
ax3.plot(x_range, stats.norm.pdf(x_range, np.mean(residuals), np.std(residuals)),
         'b-', linewidth=2, label='Normal fit')
ax3.set_xlabel('Residuals', fontsize=11)
ax3.set_ylabel('Density', fontsize=11)
ax3.set_title('Residual Distribution', fontsize=12, fontweight='bold')
ax3.legend(fontsize=9)
ax3.grid(alpha=0.3)

# Panel 4: Q-Q Plot
ax4 = fig.add_subplot(gs[1, 0])
stats.probplot(residuals, dist="norm", plot=ax4)
ax4.set_title('Normal Q-Q Plot', fontsize=12, fontweight='bold')
ax4.grid(alpha=0.3)

# Panel 5: Scale-Location Plot
ax5 = fig.add_subplot(gs[1, 1])
standardized_residuals = residuals / np.std(residuals)
sqrt_abs_std_residuals = np.sqrt(np.abs(standardized_residuals))
ax5.scatter(fitted_values, sqrt_abs_std_residuals, s=80, alpha=0.7,
            color='darkred', edgecolor='black')
z = np.polyfit(fitted_values, sqrt_abs_std_residuals, 1)
p = np.poly1d(z)
ax5.plot(sorted(fitted_values), p(sorted(fitted_values)), "b--", alpha=0.8, linewidth=2)
ax5.set_xlabel('Fitted Values', fontsize=11)
ax5.set_ylabel('√|Standardized Residuals|', fontsize=11)
ax5.set_title('Scale-Location Plot', fontsize=12, fontweight='bold')
ax5.grid(alpha=0.3)

# Panel 6: ACF Plot
ax6 = fig.add_subplot(gs[1, 2])
# Manual ACF plot
lags = 15
acf_values = [1.0]
for lag in range(1, lags + 1):
    acf_val = np.corrcoef(residuals_sorted[:-lag], residuals_sorted[lag:])[0, 1]
    acf_values.append(acf_val)
ax6.stem(range(len(acf_values)), acf_values, linefmt="C0-", markerfmt="C0o", basefmt=" ")
confidence_interval = 1.96 / np.sqrt(len(residuals_sorted))
ax6.axhline(confidence_interval, color="red", linestyle="--", linewidth=1, alpha=0.7)
ax6.axhline(-confidence_interval, color="red", linestyle="--", linewidth=1, alpha=0.7)
ax6.axhline(0, color="black", linestyle="-", linewidth=1)
ax6.set_xlabel("Lag", fontsize=11)
ax6.set_ylabel("ACF", fontsize=11)
ax6.set_title('Autocorrelation Function', fontsize=12, fontweight='bold')
ax6.grid(alpha=0.3)

# Panel 7: Absolute residuals vs fitted
ax7 = fig.add_subplot(gs[2, 0])
ax7.scatter(fitted_values, abs_residuals, s=80, alpha=0.7, color='darkred', edgecolor='black')
z = np.polyfit(fitted_values, abs_residuals, 1)
p = np.poly1d(z)
ax7.plot(sorted(fitted_values), p(sorted(fitted_values)), "b--", alpha=0.8, linewidth=2)
ax7.set_xlabel('Fitted Values', fontsize=11)
ax7.set_ylabel('|Residuals|', fontsize=11)
ax7.set_title('Absolute Residuals vs Fitted', fontsize=12, fontweight='bold')
ax7.grid(alpha=0.3)

# Panel 8: Absolute residuals vs x
ax8 = fig.add_subplot(gs[2, 1])
ax8.scatter(x_obs, abs_residuals, s=80, alpha=0.7, color='darkred', edgecolor='black')
z = np.polyfit(x_obs, abs_residuals, 1)
p = np.poly1d(z)
ax8.plot(sorted(x_obs), p(sorted(x_obs)), "b--", alpha=0.8, linewidth=2)
ax8.set_xlabel('x', fontsize=11)
ax8.set_ylabel('|Residuals|', fontsize=11)
ax8.set_title('Absolute Residuals vs x', fontsize=12, fontweight='bold')
ax8.grid(alpha=0.3)

# Panel 9: Cook's distance approximation
ax9 = fig.add_subplot(gs[2, 2])
# Approximate Cook's distance
leverage = 1/n_obs + (x_obs - np.mean(x_obs))**2 / np.sum((x_obs - np.mean(x_obs))**2)
cooks_d = (standardized_residuals**2 * leverage) / (2 * (1 - leverage))
ax9.stem(range(n_obs), cooks_d, linefmt='darkred', markerfmt='o', basefmt=' ')
ax9.axhline(4/n_obs, color='blue', linestyle='--', linewidth=2, label='Threshold (4/n)')
ax9.set_xlabel('Observation Index', fontsize=11)
ax9.set_ylabel("Cook's Distance", fontsize=11)
ax9.set_title("Influence Measures", fontsize=12, fontweight='bold')
ax9.legend(fontsize=9)
ax9.grid(alpha=0.3)

plt.suptitle('Comprehensive Residual Diagnostics', fontsize=14, fontweight='bold', y=0.995)
plt.savefig(f'{OUTPUT_DIR}/plots/residual_diagnostics.png', dpi=300, bbox_inches='tight')
print(f"   Saved: {OUTPUT_DIR}/plots/residual_diagnostics.png")
plt.close()

# ============================================================================
# LOO-PIT ANALYSIS
# ============================================================================
print("\n" + "=" * 80)
print("LOO-PIT ANALYSIS")
print("=" * 80)

print("\n10. Computing LOO-PIT (Leave-One-Out Probability Integral Transform)...")
# ArviZ LOO-PIT
fig, ax = plt.subplots(figsize=(10, 6))
az.plot_loo_pit(idata, y='Y', legend=True, ax=ax)
ax.set_title('LOO-PIT Calibration Check', fontsize=14, fontweight='bold')
ax.set_xlabel('LOO-PIT Value', fontsize=12)
ax.set_ylabel('Density', fontsize=12)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/plots/loo_pit.png', dpi=300, bbox_inches='tight')
print(f"   Saved: {OUTPUT_DIR}/plots/loo_pit.png")
print("   LOO-PIT should be approximately uniform if model is well-calibrated")
plt.close()

# ============================================================================
# COVERAGE ASSESSMENT
# ============================================================================
print("\n11. Creating detailed coverage assessment plot...")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Panel A: Coverage by observation
ax = axes[0, 0]
coverage_by_obs = []
for i in range(n_obs):
    lower = np.percentile(y_rep[:, i], 2.5)
    upper = np.percentile(y_rep[:, i], 97.5)
    in_interval = (y_obs[i] >= lower) and (y_obs[i] <= upper)
    coverage_by_obs.append(in_interval)

colors = ['green' if c else 'red' for c in coverage_by_obs]
ax.scatter(x_obs, y_obs, c=colors, s=100, alpha=0.7, edgecolor='black', linewidth=1.5)
ax.set_xlabel('x', fontsize=12)
ax.set_ylabel('Y', fontsize=12)
ax.set_title('A. Coverage Status by Observation', fontsize=13, fontweight='bold')
ax.grid(alpha=0.3)
# Add legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='green', label='Inside 95% PI'),
                   Patch(facecolor='red', label='Outside 95% PI')]
ax.legend(handles=legend_elements, fontsize=10)

# Panel B: Interval widths
ax = axes[0, 1]
interval_widths = []
for i in range(n_obs):
    lower = np.percentile(y_rep[:, i], 2.5)
    upper = np.percentile(y_rep[:, i], 97.5)
    interval_widths.append(upper - lower)

ax.scatter(x_obs, interval_widths, s=80, alpha=0.7, color='steelblue', edgecolor='black')
ax.set_xlabel('x', fontsize=12)
ax.set_ylabel('95% PI Width', fontsize=12)
ax.set_title('B. Predictive Interval Width vs x', fontsize=13, fontweight='bold')
ax.grid(alpha=0.3)

# Panel C: PIT values (approximate)
ax = axes[1, 0]
pit_values = []
for i in range(n_obs):
    pit = np.mean(y_rep[:, i] <= y_obs[i])
    pit_values.append(pit)

ax.hist(pit_values, bins=15, density=True, alpha=0.6, color='darkred', edgecolor='black')
ax.axhline(1.0, color='blue', linestyle='--', linewidth=2, label='Uniform (ideal)')
ax.set_xlabel('PIT Value', fontsize=12)
ax.set_ylabel('Density', fontsize=12)
ax.set_title('C. Probability Integral Transform', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(alpha=0.3)

# Panel D: Coverage summary
ax = axes[1, 1]
coverage_summary = {
    '50% PI': coverage_50,
    '95% PI': coverage_95,
    'Target 50%': 50,
    'Target 95%': 95
}
bars = ax.bar(['50% PI\n(Observed)', '50% PI\n(Target)', '95% PI\n(Observed)', '95% PI\n(Target)'],
              [coverage_50, 50, coverage_95, 95],
              color=['steelblue', 'lightgray', 'darkred', 'lightgray'],
              alpha=0.7, edgecolor='black', linewidth=1.5)
ax.set_ylabel('Coverage (%)', fontsize=12)
ax.set_title('D. Coverage Summary', fontsize=13, fontweight='bold')
ax.axhline(50, color='blue', linestyle='--', linewidth=1, alpha=0.5)
ax.axhline(95, color='red', linestyle='--', linewidth=1, alpha=0.5)
ax.grid(alpha=0.3, axis='y')
# Add values on bars
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.suptitle('Coverage Assessment and Calibration', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/plots/coverage_assessment.png', dpi=300, bbox_inches='tight')
print(f"   Saved: {OUTPUT_DIR}/plots/coverage_assessment.png")
plt.close()

# ============================================================================
# IDENTIFY MODEL WEAKNESSES
# ============================================================================
print("\n" + "=" * 80)
print("MODEL WEAKNESSES ASSESSMENT")
print("=" * 80)

print("\n12. Identifying potential model weaknesses...")

weaknesses = []
has_weaknesses = False

# Check 1: Extreme observations
outside_95 = [i for i, c in enumerate(coverage_by_obs) if not c]
if len(outside_95) > 0:
    has_weaknesses = True
    weaknesses.append(f"Observations outside 95% PI: {len(outside_95)} (indices: {outside_95})")
    print(f"   WARNING: {len(outside_95)} observations outside 95% predictive interval")
    for idx in outside_95:
        print(f"      - Observation {idx}: x={x_obs[idx]:.1f}, Y={y_obs[idx]:.3f}")

# Check 2: Systematic residual patterns
if abs(corr_fitted) > 0.3 or abs(corr_x) > 0.3:
    has_weaknesses = True
    weaknesses.append("Systematic patterns in residuals detected")
    print(f"   WARNING: Systematic patterns in residuals")
    print(f"      - Correlation with fitted: {corr_fitted:.3f}")
    print(f"      - Correlation with x: {corr_x:.3f}")

# Check 3: Non-normal residuals
if shapiro_p < 0.05:
    has_weaknesses = True
    weaknesses.append(f"Non-normal residuals (Shapiro p={shapiro_p:.4f})")
    print(f"   WARNING: Residuals deviate from normality (p={shapiro_p:.4f})")

# Check 4: Autocorrelation
if dw_stat < 1.5 or dw_stat > 2.5:
    has_weaknesses = True
    weaknesses.append(f"Potential autocorrelation (DW={dw_stat:.3f})")
    print(f"   WARNING: Potential autocorrelation in residuals (DW={dw_stat:.3f})")

# Check 5: Poor coverage
if coverage_95 < 85:
    has_weaknesses = True
    weaknesses.append(f"Low predictive coverage ({coverage_95:.1f}%)")
    print(f"   WARNING: Low predictive interval coverage ({coverage_95:.1f}%)")

# Check 6: Extreme test statistics
extreme_stats = [k for k, v in pp_pvalues.items() if v < 0.05 or v > 0.95]
if len(extreme_stats) > 0:
    has_weaknesses = True
    weaknesses.append(f"Extreme test statistics: {', '.join(extreme_stats)}")
    print(f"   WARNING: Extreme posterior predictive p-values for: {', '.join(extreme_stats)}")

if not has_weaknesses:
    print("   No major weaknesses detected - model appears well-calibrated")

# ============================================================================
# PLOT: MODEL WEAKNESSES (if any)
# ============================================================================
if has_weaknesses:
    print("\n13. Creating model weaknesses visualization...")
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)

    # Panel 1: Highlight problematic observations
    ax1 = fig.add_subplot(gs[0, :])
    sort_idx = np.argsort(x_obs)
    x_sorted = x_obs[sort_idx]
    y_sorted = y_obs[sort_idx]
    y_rep_sorted = y_rep[:, sort_idx]
    lower_95 = np.percentile(y_rep_sorted, 2.5, axis=0)
    upper_95 = np.percentile(y_rep_sorted, 97.5, axis=0)

    ax1.fill_between(x_sorted, lower_95, upper_95, alpha=0.3, color='steelblue',
                     label='95% Predictive Interval')

    # Plot points
    coverage_sorted = [coverage_by_obs[i] for i in sort_idx]
    for i, (x, y, c) in enumerate(zip(x_sorted, y_sorted, coverage_sorted)):
        color = 'green' if c else 'red'
        marker = 'o' if c else 'X'
        size = 80 if c else 150
        ax1.scatter(x, y, c=color, marker=marker, s=size, alpha=0.7,
                   edgecolor='black', linewidth=1.5, zorder=10)

    ax1.set_xlabel('x', fontsize=12)
    ax1.set_ylabel('Y', fontsize=12)
    ax1.set_title('Observations Outside Predictive Intervals', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(alpha=0.3)

    # Panel 2: Residual pattern emphasis
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.scatter(fitted_values, residuals, s=100, alpha=0.7,
               c=coverage_by_obs, cmap='RdYlGn', edgecolor='black', linewidth=1.5)
    ax2.axhline(0, color='black', linestyle='--', linewidth=2)
    z = np.polyfit(fitted_values, residuals, 2)  # Quadratic fit to show curvature
    p = np.poly1d(z)
    x_smooth = np.linspace(fitted_values.min(), fitted_values.max(), 100)
    ax2.plot(x_smooth, p(x_smooth), "b-", alpha=0.8, linewidth=3, label='Quadratic trend')
    ax2.set_xlabel('Fitted Values', fontsize=12)
    ax2.set_ylabel('Residuals', fontsize=12)
    ax2.set_title('Residual Patterns', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(alpha=0.3)

    # Panel 3: Extreme observations
    ax3 = fig.add_subplot(gs[1, 1])
    z_scores = np.abs((y_obs - np.mean(y_obs)) / np.std(y_obs))
    colors_z = ['red' if z > 2 else 'orange' if z > 1.5 else 'green' for z in z_scores]
    ax3.bar(range(n_obs), z_scores, color=colors_z, alpha=0.7, edgecolor='black')
    ax3.axhline(2, color='red', linestyle='--', linewidth=2, label='|z| = 2')
    ax3.axhline(1.5, color='orange', linestyle='--', linewidth=2, label='|z| = 1.5')
    ax3.set_xlabel('Observation Index', fontsize=12)
    ax3.set_ylabel('|Z-score|', fontsize=12)
    ax3.set_title('Extreme Value Detection', fontsize=13, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(alpha=0.3, axis='y')

    # Panel 4: Test statistic extremeness
    ax4 = fig.add_subplot(gs[1, 2])
    stat_names = list(pp_pvalues.keys())
    p_vals = [pp_pvalues[k] for k in stat_names]
    colors_p = ['red' if p < 0.05 or p > 0.95 else 'green' for p in p_vals]
    ax4.barh(stat_names, p_vals, color=colors_p, alpha=0.7, edgecolor='black')
    ax4.axvline(0.05, color='red', linestyle='--', linewidth=2)
    ax4.axvline(0.95, color='red', linestyle='--', linewidth=2)
    ax4.axvline(0.5, color='blue', linestyle='-', linewidth=1, alpha=0.5)
    ax4.set_xlabel('Posterior Predictive P-value', fontsize=12)
    ax4.set_title('Test Statistic Calibration', fontsize=13, fontweight='bold')
    ax4.set_xlim(0, 1)
    ax4.grid(alpha=0.3, axis='x')

    # Panel 5: Heteroscedasticity evidence
    ax5 = fig.add_subplot(gs[2, 0])
    ax5.scatter(fitted_values, abs_residuals, s=100, alpha=0.7,
               color='darkred', edgecolor='black')
    z = np.polyfit(fitted_values, abs_residuals, 1)
    p = np.poly1d(z)
    ax5.plot(sorted(fitted_values), p(sorted(fitted_values)), "b-",
            alpha=0.8, linewidth=3, label=f'Linear fit (slope={z[0]:.4f})')
    ax5.set_xlabel('Fitted Values', fontsize=12)
    ax5.set_ylabel('|Residuals|', fontsize=12)
    ax5.set_title('Heteroscedasticity Check', fontsize=13, fontweight='bold')
    ax5.legend(fontsize=10)
    ax5.grid(alpha=0.3)

    # Panel 6: Non-normality evidence
    ax6 = fig.add_subplot(gs[2, 1])
    stats.probplot(residuals, dist="norm", plot=ax6)
    ax6.set_title(f'Q-Q Plot (Shapiro p={shapiro_p:.4f})', fontsize=13, fontweight='bold')
    ax6.grid(alpha=0.3)

    # Panel 7: Summary table
    ax7 = fig.add_subplot(gs[2, 2])
    ax7.axis('off')

    # Create summary text
    summary_text = "MODEL WEAKNESSES SUMMARY\n" + "="*40 + "\n\n"
    for i, weakness in enumerate(weaknesses, 1):
        summary_text += f"{i}. {weakness}\n\n"

    summary_text += "\nOVERALL ASSESSMENT:\n"
    if len(weaknesses) >= 4:
        summary_text += "MULTIPLE ISSUES DETECTED\n"
        summary_text += "Consider model revision"
    elif len(weaknesses) >= 2:
        summary_text += "SOME ISSUES DETECTED\n"
        summary_text += "Model may need refinement"
    else:
        summary_text += "MINOR ISSUES DETECTED\n"
        summary_text += "Model is acceptable"

    ax7.text(0.1, 0.9, summary_text, transform=ax7.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.suptitle('Model Weaknesses and Deficiencies', fontsize=14, fontweight='bold')
    plt.savefig(f'{OUTPUT_DIR}/plots/model_weaknesses.png', dpi=300, bbox_inches='tight')
    print(f"   Saved: {OUTPUT_DIR}/plots/model_weaknesses.png")
    plt.close()
else:
    print("   No weaknesses plot needed - model is well-calibrated")

# ============================================================================
# SUMMARY REPORT
# ============================================================================
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

print(f"\nPosterior Predictive Check Results:")
print(f"  - Posterior samples used: {n_samples}")
print(f"  - PPC replications: {n_ppc_samples}")
print(f"  - Observations: {n_obs}")
print(f"\nCoverage:")
print(f"  - 50% PI: {coverage_50:.1f}% (target: ~50%)")
print(f"  - 95% PI: {coverage_95:.1f}% [{coverage_status}] (target: 90-98%)")
print(f"\nTest Statistics:")
print(f"  - Extreme p-values: {len(extreme_stats)} out of {len(pp_pvalues)}")
print(f"\nResiduals:")
print(f"  - Normality (Shapiro p): {shapiro_p:.4f}")
print(f"  - Independence (DW): {dw_stat:.4f}")
print(f"  - Homoscedasticity: {'OK' if p_fitted > 0.05 else 'Concern'}")
print(f"\nWeaknesses Identified: {len(weaknesses)}")
if weaknesses:
    for weakness in weaknesses:
        print(f"  - {weakness}")

print("\n" + "=" * 80)
print("POSTERIOR PREDICTIVE CHECKS COMPLETE")
print("=" * 80)
