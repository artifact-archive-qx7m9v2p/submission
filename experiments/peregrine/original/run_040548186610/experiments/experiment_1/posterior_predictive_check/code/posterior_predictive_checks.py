"""
Comprehensive Posterior Predictive Checks for Experiment 1: Negative Binomial Quadratic Model

This script performs extensive posterior predictive validation including:
- Visual comparisons (coverage, trajectories, distributions)
- Residual diagnostics (ACF analysis critical for temporal decision)
- Quantitative test statistics with Bayesian p-values
- Coverage assessment
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import arviz as az
from scipy import stats
from scipy.stats import skew, kurtosis
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Paths
DATA_PATH = "/workspace/data/data.csv"
INFERENCE_PATH = "/workspace/experiments/experiment_1/posterior_inference/diagnostics/posterior_inference.netcdf"
PLOTS_DIR = "/workspace/experiments/experiment_1/posterior_predictive_check/plots"

print("="*80)
print("POSTERIOR PREDICTIVE CHECKS: Negative Binomial Quadratic Model")
print("="*80)

# Load data
print("\n1. Loading observed data...")
data = pd.read_csv(DATA_PATH)
y_obs = data['C'].values
x = data['year'].values
n_obs = len(y_obs)
print(f"   - Observations: {n_obs}")
print(f"   - Count range: [{y_obs.min()}, {y_obs.max()}]")
print(f"   - Mean: {y_obs.mean():.1f}, Std: {y_obs.std():.1f}")

# Load InferenceData
print("\n2. Loading posterior samples...")
idata = az.from_netcdf(INFERENCE_PATH)
print(f"   - Chains: {idata.posterior.dims['chain']}")
print(f"   - Draws per chain: {idata.posterior.dims['draw']}")

# Extract posterior predictive samples
print("\n3. Extracting posterior predictive samples...")
if 'posterior_predictive' in idata.groups():
    y_rep = idata.posterior_predictive['C_obs'].values
    # Flatten chains and draws
    y_rep = y_rep.reshape(-1, n_obs)
    print(f"   - Posterior predictive samples shape: {y_rep.shape}")
    print(f"   - Total replications: {y_rep.shape[0]}")
else:
    print("   ERROR: No posterior_predictive group found!")
    print("   Available groups:", list(idata.groups()))
    raise ValueError("Posterior predictive samples not found in InferenceData")

# Extract posterior means for predictions
print("\n4. Computing posterior summaries...")
y_pred_mean = y_rep.mean(axis=0)
y_pred_std = y_rep.std(axis=0)
y_pred_median = np.median(y_rep, axis=0)

# Compute prediction intervals
y_pred_q025 = np.percentile(y_rep, 2.5, axis=0)
y_pred_q975 = np.percentile(y_rep, 97.5, axis=0)
y_pred_q10 = np.percentile(y_rep, 10, axis=0)
y_pred_q90 = np.percentile(y_rep, 90, axis=0)
y_pred_q25 = np.percentile(y_rep, 25, axis=0)
y_pred_q75 = np.percentile(y_rep, 75, axis=0)

print(f"   - Mean prediction: {y_pred_mean.mean():.1f}")
print(f"   - Prediction SD: {y_pred_std.mean():.1f}")

# ============================================================================
# COVERAGE ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("COVERAGE ANALYSIS")
print("="*80)

# Calculate coverage at different levels
in_50 = np.sum((y_obs >= y_pred_q25) & (y_obs <= y_pred_q75))
in_80 = np.sum((y_obs >= y_pred_q10) & (y_obs <= y_pred_q90))
in_95 = np.sum((y_obs >= y_pred_q025) & (y_obs <= y_pred_q975))

coverage_50 = 100 * in_50 / n_obs
coverage_80 = 100 * in_80 / n_obs
coverage_95 = 100 * in_95 / n_obs

print(f"\nEmpirical Coverage:")
print(f"   50% interval: {coverage_50:.1f}% ({in_50}/{n_obs}) [Expected: 50%]")
print(f"   80% interval: {coverage_80:.1f}% ({in_80}/{n_obs}) [Expected: 80%]")
print(f"   95% interval: {coverage_95:.1f}% ({in_95}/{n_obs}) [Expected: 95%]")

# Assess coverage quality
if 90 <= coverage_95 <= 98:
    coverage_quality = "GOOD"
elif 85 <= coverage_95 <= 100:
    coverage_quality = "ACCEPTABLE"
else:
    coverage_quality = "POOR"
print(f"\nCoverage Quality: {coverage_quality}")

# ============================================================================
# RESIDUAL ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("RESIDUAL ANALYSIS")
print("="*80)

# Compute residuals (observed - predicted mean)
residuals = y_obs - y_pred_mean
pearson_residuals = residuals / y_pred_std

print(f"\nResidual Statistics:")
print(f"   Mean: {residuals.mean():.2f}")
print(f"   Std: {residuals.std():.2f}")
print(f"   Min: {residuals.min():.2f}")
print(f"   Max: {residuals.max():.2f}")

# Compute autocorrelation of residuals
import sys; sys.path.insert(0, "/workspace/experiments/experiment_1/posterior_predictive_check/code"); from acf_util import acf

max_lag = min(20, n_obs // 3)
residual_acf = acf(residuals, nlags=max_lag, fft=False)

print(f"\nResidual Autocorrelation (CRITICAL for Phase 2 decision):")
print(f"   ACF(1): {residual_acf[1]:.3f}")
print(f"   ACF(2): {residual_acf[2]:.3f}")
print(f"   ACF(3): {residual_acf[3]:.3f}")

# Decision threshold
if residual_acf[1] > 0.5:
    temporal_decision = "TRIGGERS PHASE 2 (temporal models needed)"
elif residual_acf[1] > 0.3:
    temporal_decision = "BORDERLINE (consider temporal models)"
else:
    temporal_decision = "No temporal structure detected"

print(f"\nTemporal Structure Decision:")
print(f"   {temporal_decision}")

# ============================================================================
# TEST STATISTICS AND BAYESIAN P-VALUES
# ============================================================================
print("\n" + "="*80)
print("TEST STATISTICS & BAYESIAN P-VALUES")
print("="*80)

def compute_test_statistics(y):
    """Compute various test statistics for PPC"""
    return {
        'mean': np.mean(y),
        'variance': np.var(y, ddof=1),
        'std': np.std(y, ddof=1),
        'min': np.min(y),
        'max': np.max(y),
        'range': np.max(y) - np.min(y),
        'skewness': skew(y),
        'kurtosis': kurtosis(y),
        'q25': np.percentile(y, 25),
        'q50': np.percentile(y, 50),
        'q75': np.percentile(y, 75),
        'iqr': np.percentile(y, 75) - np.percentile(y, 25),
    }

# Add autocorrelation as test statistic
def compute_acf1(y):
    """Compute lag-1 autocorrelation"""
    acf_vals = acf(y, nlags=1, fft=False)
    return acf_vals[1]

# Observed statistics
obs_stats = compute_test_statistics(y_obs)
obs_acf1 = compute_acf1(y_obs)

print("\nObserved Test Statistics:")
for key, val in obs_stats.items():
    print(f"   {key:12s}: {val:8.2f}")
print(f"   {'ACF(1)':12s}: {obs_acf1:8.3f}")

# Replicated statistics
print("\nComputing test statistics for replicated datasets...")
rep_stats = {key: [] for key in obs_stats.keys()}
rep_stats['acf1'] = []

for i in range(y_rep.shape[0]):
    stats_i = compute_test_statistics(y_rep[i])
    for key, val in stats_i.items():
        rep_stats[key].append(val)
    rep_stats['acf1'].append(compute_acf1(y_rep[i]))

# Convert to arrays
for key in rep_stats.keys():
    rep_stats[key] = np.array(rep_stats[key])

# Compute Bayesian p-values
print("\nBayesian p-values [P(T_rep > T_obs)]:")
print("   (Extreme values <0.05 or >0.95 indicate poor fit)")
print()

bayesian_pvals = {}
for key in obs_stats.keys():
    pval = np.mean(rep_stats[key] >= obs_stats[key])
    bayesian_pvals[key] = pval
    flag = "***" if pval < 0.05 or pval > 0.95 else ""
    print(f"   {key:12s}: {pval:.3f} {flag}")

# ACF p-value
pval_acf1 = np.mean(rep_stats['acf1'] >= obs_acf1)
bayesian_pvals['acf1'] = pval_acf1
flag = "***" if pval_acf1 < 0.05 or pval_acf1 > 0.95 else ""
print(f"   {'ACF(1)':12s}: {pval_acf1:.3f} {flag}")

# Identify problematic statistics
problematic_stats = [k for k, v in bayesian_pvals.items() if v < 0.05 or v > 0.95]
if problematic_stats:
    print(f"\nProblematic statistics (p < 0.05 or p > 0.95):")
    for stat in problematic_stats:
        print(f"   - {stat}")
else:
    print("\nNo statistics with extreme p-values detected.")

# ============================================================================
# VISUALIZATIONS
# ============================================================================
print("\n" + "="*80)
print("GENERATING VISUALIZATIONS")
print("="*80)

# 1. COMPREHENSIVE PPC DASHBOARD
# ============================================================================
print("\n1. Creating comprehensive PPC dashboard...")

fig = plt.figure(figsize=(20, 12))
gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)

# Panel A: Observed vs Predicted
ax1 = fig.add_subplot(gs[0, 0])
ax1.scatter(y_pred_mean, y_obs, alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
lim_min = min(y_obs.min(), y_pred_mean.min())
lim_max = max(y_obs.max(), y_pred_mean.max())
ax1.plot([lim_min, lim_max], [lim_min, lim_max], 'r--', lw=2, label='Perfect fit')
ax1.set_xlabel('Posterior Mean Prediction', fontsize=11)
ax1.set_ylabel('Observed Count', fontsize=11)
ax1.set_title('A. Observed vs Predicted', fontsize=12, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Add R-squared
r2 = np.corrcoef(y_obs, y_pred_mean)[0, 1]**2
ax1.text(0.05, 0.95, f'R² = {r2:.3f}', transform=ax1.transAxes,
         fontsize=10, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Panel B: Coverage Plot
ax2 = fig.add_subplot(gs[0, 1])
time_idx = np.arange(n_obs)
ax2.fill_between(time_idx, y_pred_q025, y_pred_q975, alpha=0.2, label='95% PI', color='blue')
ax2.fill_between(time_idx, y_pred_q10, y_pred_q90, alpha=0.3, label='80% PI', color='blue')
ax2.fill_between(time_idx, y_pred_q25, y_pred_q75, alpha=0.4, label='50% PI', color='blue')
ax2.plot(time_idx, y_pred_mean, 'b-', lw=2, label='Posterior mean')
ax2.scatter(time_idx, y_obs, color='red', s=40, zorder=5, label='Observed', edgecolors='black', linewidth=0.5)
ax2.set_xlabel('Time Index', fontsize=11)
ax2.set_ylabel('Count', fontsize=11)
ax2.set_title(f'B. Coverage Plot (95% coverage: {coverage_95:.1f}%)', fontsize=12, fontweight='bold')
ax2.legend(loc='upper left', fontsize=8)
ax2.grid(True, alpha=0.3)

# Panel C: Trajectory Plot (Spaghetti)
ax3 = fig.add_subplot(gs[0, 2])
n_trajectories = min(100, y_rep.shape[0])
for i in np.random.choice(y_rep.shape[0], n_trajectories, replace=False):
    ax3.plot(time_idx, y_rep[i], alpha=0.05, color='blue', lw=0.5)
ax3.plot(time_idx, y_obs, 'ro-', lw=2, markersize=4, label='Observed', zorder=10)
ax3.set_xlabel('Time Index', fontsize=11)
ax3.set_ylabel('Count', fontsize=11)
ax3.set_title('C. Posterior Predictive Trajectories', fontsize=12, fontweight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Panel D: Distribution Comparison
ax4 = fig.add_subplot(gs[0, 3])
ax4.hist(y_obs, bins=20, alpha=0.6, density=True, label='Observed', color='red', edgecolor='black')
# Sample some replicated datasets
for i in np.random.choice(y_rep.shape[0], 50, replace=False):
    ax4.hist(y_rep[i], bins=20, alpha=0.02, density=True, color='blue', edgecolor='none')
ax4.set_xlabel('Count', fontsize=11)
ax4.set_ylabel('Density', fontsize=11)
ax4.set_title('D. Distribution: Observed vs Replicated', fontsize=12, fontweight='bold')
ax4.legend()
ax4.grid(True, alpha=0.3)

# Panel E: Residuals vs Fitted
ax5 = fig.add_subplot(gs[1, 0])
ax5.scatter(y_pred_mean, residuals, alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
ax5.axhline(0, color='red', linestyle='--', lw=2)
ax5.set_xlabel('Fitted Values', fontsize=11)
ax5.set_ylabel('Residuals', fontsize=11)
ax5.set_title('E. Residuals vs Fitted Values', fontsize=12, fontweight='bold')
ax5.grid(True, alpha=0.3)

# Panel F: Residuals vs Time
ax6 = fig.add_subplot(gs[1, 1])
ax6.scatter(time_idx, residuals, alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
ax6.axhline(0, color='red', linestyle='--', lw=2)
ax6.plot(time_idx, residuals, 'b-', alpha=0.3, lw=1)
ax6.set_xlabel('Time Index', fontsize=11)
ax6.set_ylabel('Residuals', fontsize=11)
ax6.set_title('F. Residuals vs Time', fontsize=12, fontweight='bold')
ax6.grid(True, alpha=0.3)

# Panel G: Residual ACF (CRITICAL)
ax7 = fig.add_subplot(gs[1, 2])
lags = np.arange(len(residual_acf))
ax7.bar(lags, residual_acf, width=0.3, alpha=0.7)
ax7.axhline(0, color='black', lw=1)
# Confidence bands (approximate)
conf_level = 1.96 / np.sqrt(n_obs)
ax7.axhline(conf_level, color='red', linestyle='--', lw=1, label='95% CI')
ax7.axhline(-conf_level, color='red', linestyle='--', lw=1)
# Highlight ACF(1)
ax7.axhline(0.5, color='orange', linestyle=':', lw=2, label='Phase 2 threshold')
ax7.set_xlabel('Lag', fontsize=11)
ax7.set_ylabel('ACF', fontsize=11)
ax7.set_title(f'G. Residual ACF (Lag-1: {residual_acf[1]:.3f})', fontsize=12, fontweight='bold')
ax7.legend(fontsize=8)
ax7.grid(True, alpha=0.3)

# Panel H: QQ Plot
ax8 = fig.add_subplot(gs[1, 3])
stats.probplot(pearson_residuals, dist="norm", plot=ax8)
ax8.set_title('H. Q-Q Plot (Pearson Residuals)', fontsize=12, fontweight='bold')
ax8.grid(True, alpha=0.3)

# Panel I: Test Statistic - Mean
ax9 = fig.add_subplot(gs[2, 0])
ax9.hist(rep_stats['mean'], bins=50, alpha=0.6, density=True, edgecolor='black')
ax9.axvline(obs_stats['mean'], color='red', linestyle='--', lw=2, label='Observed')
ax9.set_xlabel('Mean', fontsize=11)
ax9.set_ylabel('Density', fontsize=11)
ax9.set_title(f'I. Mean (p={bayesian_pvals["mean"]:.3f})', fontsize=12, fontweight='bold')
ax9.legend()
ax9.grid(True, alpha=0.3)

# Panel J: Test Statistic - Variance
ax10 = fig.add_subplot(gs[2, 1])
ax10.hist(rep_stats['variance'], bins=50, alpha=0.6, density=True, edgecolor='black')
ax10.axvline(obs_stats['variance'], color='red', linestyle='--', lw=2, label='Observed')
ax10.set_xlabel('Variance', fontsize=11)
ax10.set_ylabel('Density', fontsize=11)
ax10.set_title(f'J. Variance (p={bayesian_pvals["variance"]:.3f})', fontsize=12, fontweight='bold')
ax10.legend()
ax10.grid(True, alpha=0.3)

# Panel K: Test Statistic - ACF(1)
ax11 = fig.add_subplot(gs[2, 2])
ax11.hist(rep_stats['acf1'], bins=50, alpha=0.6, density=True, edgecolor='black')
ax11.axvline(obs_acf1, color='red', linestyle='--', lw=2, label='Observed')
ax11.axvline(0.5, color='orange', linestyle=':', lw=2, label='Phase 2 threshold')
ax11.set_xlabel('ACF(1)', fontsize=11)
ax11.set_ylabel('Density', fontsize=11)
ax11.set_title(f'K. ACF(1) (p={bayesian_pvals["acf1"]:.3f})', fontsize=12, fontweight='bold')
ax11.legend()
ax11.grid(True, alpha=0.3)

# Panel L: Test Statistic - Max
ax12 = fig.add_subplot(gs[2, 3])
ax12.hist(rep_stats['max'], bins=50, alpha=0.6, density=True, edgecolor='black')
ax12.axvline(obs_stats['max'], color='red', linestyle='--', lw=2, label='Observed')
ax12.set_xlabel('Maximum', fontsize=11)
ax12.set_ylabel('Density', fontsize=11)
ax12.set_title(f'L. Maximum (p={bayesian_pvals["max"]:.3f})', fontsize=12, fontweight='bold')
ax12.legend()
ax12.grid(True, alpha=0.3)

plt.suptitle('Posterior Predictive Check Dashboard: NegBinomial Quadratic Model',
             fontsize=16, fontweight='bold', y=0.995)

plt.savefig(f'{PLOTS_DIR}/ppc_dashboard.png', dpi=300, bbox_inches='tight')
print(f"   Saved: ppc_dashboard.png")
plt.close()

# 2. DETAILED COVERAGE PLOT
# ============================================================================
print("2. Creating detailed coverage plot...")

fig, ax = plt.subplots(figsize=(14, 6))
time_idx = np.arange(n_obs)

# Plot prediction intervals
ax.fill_between(time_idx, y_pred_q025, y_pred_q975, alpha=0.15, label='95% PI', color='skyblue')
ax.fill_between(time_idx, y_pred_q10, y_pred_q90, alpha=0.25, label='80% PI', color='steelblue')
ax.fill_between(time_idx, y_pred_q25, y_pred_q75, alpha=0.35, label='50% PI', color='royalblue')

# Plot means
ax.plot(time_idx, y_pred_mean, 'b-', lw=2.5, label='Posterior mean', zorder=4)
ax.plot(time_idx, y_pred_median, 'b--', lw=1.5, label='Posterior median', alpha=0.6, zorder=3)

# Plot observed data
ax.scatter(time_idx, y_obs, color='red', s=60, zorder=5, label='Observed',
           edgecolors='darkred', linewidth=1.5, marker='o')

# Highlight points outside 95% interval
outside_95 = (y_obs < y_pred_q025) | (y_obs > y_pred_q975)
if np.any(outside_95):
    ax.scatter(time_idx[outside_95], y_obs[outside_95], color='orange', s=120,
               zorder=6, label='Outside 95% PI', marker='X', edgecolors='darkorange', linewidth=2)

ax.set_xlabel('Time Index', fontsize=13)
ax.set_ylabel('Count', fontsize=13)
ax.set_title(f'Posterior Predictive Coverage\n95% Coverage: {coverage_95:.1f}%, 80% Coverage: {coverage_80:.1f}%, 50% Coverage: {coverage_50:.1f}%',
             fontsize=14, fontweight='bold')
ax.legend(loc='upper left', fontsize=10, ncol=2)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{PLOTS_DIR}/coverage_detailed.png', dpi=300, bbox_inches='tight')
print(f"   Saved: coverage_detailed.png")
plt.close()

# 3. RESIDUAL DIAGNOSTICS SUITE
# ============================================================================
print("3. Creating residual diagnostics suite...")

fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# A: Residuals vs Fitted
ax = axes[0, 0]
ax.scatter(y_pred_mean, residuals, alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
ax.axhline(0, color='red', linestyle='--', lw=2)
# Add lowess smooth
from scipy.signal import savgol_filter
sorted_idx = np.argsort(y_pred_mean)
try:
    smooth = savgol_filter(residuals[sorted_idx], window_length=min(11, n_obs//3*2-1), polyorder=2)
    ax.plot(y_pred_mean[sorted_idx], smooth, 'g-', lw=2, label='Smooth trend')
except:
    pass
ax.set_xlabel('Fitted Values', fontsize=11)
ax.set_ylabel('Residuals', fontsize=11)
ax.set_title('A. Residuals vs Fitted', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# B: Residuals vs Time
ax = axes[0, 1]
ax.scatter(time_idx, residuals, alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
ax.plot(time_idx, residuals, 'b-', alpha=0.3, lw=1)
ax.axhline(0, color='red', linestyle='--', lw=2)
# Add smooth
try:
    smooth = savgol_filter(residuals, window_length=min(11, n_obs//3*2-1), polyorder=2)
    ax.plot(time_idx, smooth, 'g-', lw=2, label='Smooth trend')
except:
    pass
ax.set_xlabel('Time Index', fontsize=11)
ax.set_ylabel('Residuals', fontsize=11)
ax.set_title('B. Residuals vs Time', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# C: Residual ACF
ax = axes[0, 2]
lags = np.arange(len(residual_acf))
ax.bar(lags, residual_acf, width=0.5, alpha=0.7, edgecolor='black')
ax.axhline(0, color='black', lw=1)
conf_level = 1.96 / np.sqrt(n_obs)
ax.axhline(conf_level, color='red', linestyle='--', lw=1.5, label='95% CI')
ax.axhline(-conf_level, color='red', linestyle='--', lw=1.5)
ax.axhline(0.5, color='orange', linestyle=':', lw=2.5, label='Phase 2 threshold (0.5)')
ax.axhline(0.3, color='yellow', linestyle=':', lw=2, label='Borderline (0.3)')
ax.set_xlabel('Lag', fontsize=11)
ax.set_ylabel('ACF', fontsize=11)
ax.set_title(f'C. Residual ACF (ACF(1)={residual_acf[1]:.3f})', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# D: Histogram of residuals
ax = axes[1, 0]
ax.hist(residuals, bins=20, alpha=0.6, density=True, edgecolor='black', label='Residuals')
# Overlay normal
x_norm = np.linspace(residuals.min(), residuals.max(), 100)
ax.plot(x_norm, stats.norm.pdf(x_norm, residuals.mean(), residuals.std()),
        'r-', lw=2, label='Normal fit')
ax.set_xlabel('Residuals', fontsize=11)
ax.set_ylabel('Density', fontsize=11)
ax.set_title('D. Residual Distribution', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# E: QQ Plot
ax = axes[1, 1]
stats.probplot(pearson_residuals, dist="norm", plot=ax)
ax.set_title('E. Q-Q Plot (Pearson Residuals)', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)

# F: Scale-Location Plot
ax = axes[1, 2]
sqrt_abs_resid = np.sqrt(np.abs(pearson_residuals))
ax.scatter(y_pred_mean, sqrt_abs_resid, alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
sorted_idx = np.argsort(y_pred_mean)
try:
    smooth = savgol_filter(sqrt_abs_resid[sorted_idx], window_length=min(11, n_obs//3*2-1), polyorder=2)
    ax.plot(y_pred_mean[sorted_idx], smooth, 'r-', lw=2, label='Smooth trend')
except:
    pass
ax.set_xlabel('Fitted Values', fontsize=11)
ax.set_ylabel('√|Standardized Residuals|', fontsize=11)
ax.set_title('F. Scale-Location Plot', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

plt.suptitle('Residual Diagnostics Suite', fontsize=14, fontweight='bold', y=0.998)
plt.tight_layout()
plt.savefig(f'{PLOTS_DIR}/residual_diagnostics.png', dpi=300, bbox_inches='tight')
print(f"   Saved: residual_diagnostics.png")
plt.close()

# 4. TEST STATISTICS COMPARISON
# ============================================================================
print("4. Creating test statistics comparison plot...")

# Select key statistics to plot
key_stats = ['mean', 'variance', 'max', 'acf1', 'skewness', 'kurtosis']
n_stats = len(key_stats)

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.flatten()

for idx, stat in enumerate(key_stats):
    ax = axes[idx]

    if stat == 'acf1':
        obs_val = obs_acf1
    else:
        obs_val = obs_stats[stat]

    rep_vals = rep_stats[stat]
    pval = bayesian_pvals[stat]

    # Histogram
    ax.hist(rep_vals, bins=50, alpha=0.6, density=True, edgecolor='black', label='Replicated')

    # Observed value
    ax.axvline(obs_val, color='red', linestyle='--', lw=2.5, label='Observed')

    # Add percentile info
    percentile = 100 * np.mean(rep_vals <= obs_val)

    # Color based on p-value
    if pval < 0.05 or pval > 0.95:
        color = 'red'
        result = 'POOR'
    else:
        color = 'green'
        result = 'GOOD'

    ax.set_xlabel(stat.capitalize(), fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title(f'{stat.capitalize()}\np-value: {pval:.3f} ({result})',
                 fontsize=12, fontweight='bold', color=color)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Add text box with percentile
    ax.text(0.02, 0.98, f'Obs at {percentile:.1f}th percentile',
            transform=ax.transAxes, fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.suptitle('Test Statistics: Observed vs Posterior Predictive Distribution',
             fontsize=14, fontweight='bold', y=0.998)
plt.tight_layout()
plt.savefig(f'{PLOTS_DIR}/test_statistics.png', dpi=300, bbox_inches='tight')
print(f"   Saved: test_statistics.png")
plt.close()

# 5. ArviZ PPC plots
# ============================================================================
print("5. Creating ArviZ PPC plots...")

# Standard PPC plot
fig, ax = plt.subplots(figsize=(12, 6))
az.plot_ppc(idata, num_pp_samples=100, ax=ax)
ax.set_title('ArviZ Posterior Predictive Check', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{PLOTS_DIR}/arviz_ppc.png', dpi=300, bbox_inches='tight')
print(f"   Saved: arviz_ppc.png")
plt.close()

# LOO-PIT plot for calibration
print("6. Creating LOO-PIT calibration plot...")
try:
    fig, ax = plt.subplots(figsize=(10, 6))
    az.plot_loo_pit(idata, y='C_obs', ecdf=True, ax=ax)
    ax.set_title('LOO-PIT Calibration Check', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{PLOTS_DIR}/loo_pit.png', dpi=300, bbox_inches='tight')
    print(f"   Saved: loo_pit.png")
    plt.close()
except Exception as e:
    print(f"   LOO-PIT plot failed: {e}")

# ============================================================================
# SAVE RESULTS
# ============================================================================
print("\n" + "="*80)
print("SAVING RESULTS")
print("="*80)

results = {
    'coverage_50': coverage_50,
    'coverage_80': coverage_80,
    'coverage_95': coverage_95,
    'coverage_quality': coverage_quality,
    'residual_acf1': residual_acf[1],
    'residual_acf2': residual_acf[2],
    'residual_acf3': residual_acf[3],
    'temporal_decision': temporal_decision,
    'bayesian_pvals': bayesian_pvals,
    'problematic_stats': problematic_stats,
    'obs_stats': obs_stats,
    'obs_acf1': obs_acf1,
}

# Save as numpy archive
np.savez(f'{PLOTS_DIR}/../code/ppc_results.npz',
         y_obs=y_obs,
         y_pred_mean=y_pred_mean,
         y_pred_std=y_pred_std,
         residuals=residuals,
         residual_acf=residual_acf,
         **results)

print("   Saved: ppc_results.npz")

# ============================================================================
# OVERALL ASSESSMENT
# ============================================================================
print("\n" + "="*80)
print("OVERALL MODEL FIT ASSESSMENT")
print("="*80)

# Determine overall fit quality
fit_criteria = []

# Coverage
if 90 <= coverage_95 <= 98:
    fit_criteria.append(('Coverage', 'GOOD', f'{coverage_95:.1f}% (target: 90-98%)'))
elif 85 <= coverage_95 <= 100:
    fit_criteria.append(('Coverage', 'ACCEPTABLE', f'{coverage_95:.1f}% (target: 85-100%)'))
else:
    fit_criteria.append(('Coverage', 'POOR', f'{coverage_95:.1f}%'))

# Residual ACF
if residual_acf[1] < 0.3:
    fit_criteria.append(('Residual ACF(1)', 'GOOD', f'{residual_acf[1]:.3f} (< 0.3)'))
elif residual_acf[1] < 0.5:
    fit_criteria.append(('Residual ACF(1)', 'ACCEPTABLE', f'{residual_acf[1]:.3f} (0.3-0.5)'))
else:
    fit_criteria.append(('Residual ACF(1)', 'POOR', f'{residual_acf[1]:.3f} (> 0.5)'))

# P-values
n_extreme_pvals = len(problematic_stats)
if n_extreme_pvals == 0:
    fit_criteria.append(('P-values', 'GOOD', 'No extreme p-values'))
elif n_extreme_pvals <= 2:
    fit_criteria.append(('P-values', 'ACCEPTABLE', f'{n_extreme_pvals} extreme p-values'))
else:
    fit_criteria.append(('P-values', 'POOR', f'{n_extreme_pvals} extreme p-values'))

# Overall
print("\nFit Criteria Summary:")
for criterion, quality, detail in fit_criteria:
    print(f"   {criterion:20s}: {quality:12s} - {detail}")

# Final decision
good_count = sum(1 for _, q, _ in fit_criteria if q == 'GOOD')
acceptable_count = sum(1 for _, q, _ in fit_criteria if q == 'ACCEPTABLE')
poor_count = sum(1 for _, q, _ in fit_criteria if q == 'POOR')

if poor_count == 0 and good_count >= 2:
    overall_fit = 'GOOD'
elif poor_count <= 1:
    overall_fit = 'ACCEPTABLE'
else:
    overall_fit = 'POOR'

print(f"\nOVERALL FIT QUALITY: {overall_fit}")

# Next steps
print("\n" + "="*80)
print("NEXT STEPS RECOMMENDATION")
print("="*80)

if residual_acf[1] > 0.5:
    print("\n*** PHASE 2 TRIGGERED: Temporal Models Recommended ***")
    print(f"   Residual ACF(1) = {residual_acf[1]:.3f} exceeds threshold of 0.5")
    print("   Models to explore: AR, ARMA, Random Walk, State Space")
elif residual_acf[1] > 0.3:
    print("\n*** BORDERLINE: Consider Temporal Models ***")
    print(f"   Residual ACF(1) = {residual_acf[1]:.3f} in borderline range (0.3-0.5)")
    print("   Temporal models may improve fit")
else:
    print("\n*** No strong temporal structure detected ***")
    print(f"   Residual ACF(1) = {residual_acf[1]:.3f} < 0.3")
    print("   Temporal models may not be necessary")

if coverage_95 < 85:
    print("\n*** COVERAGE WARNING: Model underpredicting uncertainty ***")
    print(f"   95% coverage = {coverage_95:.1f}% (target: 85-98%)")
    print("   Consider: overdispersion, mixture models, robust distributions")

if problematic_stats:
    print(f"\n*** {len(problematic_stats)} test statistics show extreme p-values ***")
    print("   Specific deficiencies to address:")
    for stat in problematic_stats:
        pval = bayesian_pvals[stat]
        if pval < 0.05:
            print(f"   - {stat}: observed value in lower tail (p={pval:.3f})")
        else:
            print(f"   - {stat}: observed value in upper tail (p={pval:.3f})")

print("\n" + "="*80)
print("POSTERIOR PREDICTIVE CHECKS COMPLETE")
print("="*80)
print(f"\nResults saved to: {PLOTS_DIR}/../")
print("Generated plots:")
print("   - ppc_dashboard.png (comprehensive 12-panel overview)")
print("   - coverage_detailed.png (detailed coverage assessment)")
print("   - residual_diagnostics.png (6-panel residual suite)")
print("   - test_statistics.png (6 key test statistics)")
print("   - arviz_ppc.png (ArviZ standard PPC)")
print("   - loo_pit.png (calibration check)")
