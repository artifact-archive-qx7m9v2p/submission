"""
Comprehensive Posterior Predictive Checks for Experiment 3: Latent AR(1) Negative Binomial Model

This script performs extensive posterior predictive validation with emphasis on:
- Temporal autocorrelation (KEY: must be < 0.3, was 0.686 in Exp 1)
- Coverage assessment (TARGET: 90-98%, was 100% in Exp 1)
- Comparison to Experiment 1 baseline
- Residual diagnostics after AR(1) structure accounted for

Critical Success Metrics:
1. Residual ACF(1) < 0.3 (was 0.686)
2. Coverage 90-98% (was 100%)
3. No systematic temporal patterns
4. Bayesian p-values in 0.1-0.9 range
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import arviz as az
from scipy import stats
from scipy.stats import skew, kurtosis
from scipy.signal import savgol_filter
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Paths
DATA_PATH = "/workspace/data/data.csv"
INFERENCE_PATH = "/workspace/experiments/experiment_3/posterior_inference/diagnostics/posterior_inference.netcdf"
PLOTS_DIR = "/workspace/experiments/experiment_3/posterior_predictive_check/plots"
EXP1_RESULTS = "/workspace/experiments/experiment_1/posterior_predictive_check/code/ppc_results.npz"

print("="*80)
print("POSTERIOR PREDICTIVE CHECKS: Latent AR(1) Negative Binomial Model")
print("="*80)
print("\nExperiment 3: Testing if AR(1) structure solves temporal autocorrelation")
print("Baseline (Exp 1): Residual ACF(1) = 0.686, Coverage = 100%")
print("Target: Residual ACF(1) < 0.3, Coverage = 90-98%")

# ============================================================================
# LOAD DATA AND POSTERIOR
# ============================================================================

print("\n" + "="*80)
print("1. LOADING DATA AND POSTERIOR SAMPLES")
print("="*80)

# Load data
data = pd.read_csv(DATA_PATH)
y_obs = data['C'].values
x = data['year'].values
n_obs = len(y_obs)
print(f"\nObserved data:")
print(f"   - Observations: {n_obs}")
print(f"   - Count range: [{y_obs.min()}, {y_obs.max()}]")
print(f"   - Mean: {y_obs.mean():.1f}, Std: {y_obs.std():.1f}")

# Load InferenceData
print(f"\nLoading posterior from: {INFERENCE_PATH}")
idata = az.from_netcdf(INFERENCE_PATH)
print(f"   - Chains: {idata.posterior.dims['chain']}")
print(f"   - Draws per chain: {idata.posterior.dims['draw']}")
print(f"   - Available groups: {list(idata.groups())}")

# Load Experiment 1 results for comparison
print(f"\nLoading Experiment 1 baseline results...")
exp1_results = np.load(EXP1_RESULTS, allow_pickle=True)
exp1_acf1 = float(exp1_results['residual_acf'][1])
exp1_coverage_95 = float(exp1_results['coverage_95'])
print(f"   - Exp 1 Residual ACF(1): {exp1_acf1:.3f}")
print(f"   - Exp 1 Coverage (95%): {exp1_coverage_95:.1f}%")

# ============================================================================
# GENERATE POSTERIOR PREDICTIVE SAMPLES
# ============================================================================

print("\n" + "="*80)
print("2. GENERATING POSTERIOR PREDICTIVE SAMPLES")
print("="*80)

# Check if posterior predictive exists
if 'posterior_predictive' not in idata.groups():
    print("\nGenerating posterior predictive samples from model...")

    # Extract posterior samples
    posterior = idata.posterior
    n_chains = posterior.dims['chain']
    n_draws = posterior.dims['draw']

    # Extract parameters
    beta_0 = posterior['beta_0'].values.reshape(-1)
    beta_1 = posterior['beta_1'].values.reshape(-1)
    beta_2 = posterior['beta_2'].values.reshape(-1)
    rho = posterior['rho'].values.reshape(-1)
    sigma_eta = posterior['sigma_eta'].values.reshape(-1)
    phi = posterior['phi'].values.reshape(-1)

    n_samples = len(beta_0)
    print(f"   - Total posterior samples: {n_samples}")

    # Generate posterior predictive samples
    print("   - Generating replicated datasets with AR(1) structure...")
    y_rep = np.zeros((n_samples, n_obs))

    for i in range(n_samples):
        if i % 500 == 0:
            print(f"     Progress: {i}/{n_samples} samples...")

        # Generate AR(1) latent process
        epsilon = np.zeros(n_obs)
        sigma_0 = sigma_eta[i] / np.sqrt(1 - rho[i]**2)
        epsilon[0] = np.random.normal(0, sigma_0)

        for t in range(1, n_obs):
            epsilon[t] = rho[i] * epsilon[t-1] + np.random.normal(0, sigma_eta[i])

        # Compute latent state
        alpha = beta_0[i] + beta_1[i] * x + beta_2[i] * x**2 + epsilon
        mu = np.exp(alpha)

        # Generate negative binomial observations
        # Stan uses NegBinomial2 parameterization: Var = μ + μ²/φ
        # Convert to scipy's n, p parameterization
        for t in range(n_obs):
            p = phi[i] / (mu[t] + phi[i])
            n = phi[i]
            y_rep[i, t] = np.random.negative_binomial(n, p)

    print(f"   - Generated {n_samples} replicated datasets")

else:
    print("\nUsing pre-generated posterior predictive samples...")
    # Extract from InferenceData - need to check variable name
    if 'C' in idata.posterior_predictive:
        y_rep = idata.posterior_predictive['C'].values
    elif 'C_obs' in idata.posterior_predictive:
        y_rep = idata.posterior_predictive['C_obs'].values
    else:
        print(f"   Available variables: {list(idata.posterior_predictive.data_vars)}")
        raise ValueError("Could not find posterior predictive variable")

    # Flatten chains and draws
    y_rep = y_rep.reshape(-1, n_obs)
    print(f"   - Posterior predictive samples shape: {y_rep.shape}")

print(f"\nPosterior predictive summary:")
print(f"   - Total replications: {y_rep.shape[0]}")
print(f"   - Mean replication count: {y_rep.mean():.1f}")
print(f"   - Observed mean count: {y_obs.mean():.1f}")

# ============================================================================
# COMPUTE POSTERIOR SUMMARIES
# ============================================================================

print("\n" + "="*80)
print("3. COMPUTING POSTERIOR SUMMARIES")
print("="*80)

y_pred_mean = y_rep.mean(axis=0)
y_pred_std = y_rep.std(axis=0)
y_pred_median = np.median(y_rep, axis=0)

# Prediction intervals
y_pred_q025 = np.percentile(y_rep, 2.5, axis=0)
y_pred_q975 = np.percentile(y_rep, 97.5, axis=0)
y_pred_q10 = np.percentile(y_rep, 10, axis=0)
y_pred_q90 = np.percentile(y_rep, 90, axis=0)
y_pred_q25 = np.percentile(y_rep, 25, axis=0)
y_pred_q75 = np.percentile(y_rep, 75, axis=0)

print(f"   - Mean prediction: {y_pred_mean.mean():.1f}")
print(f"   - Prediction SD: {y_pred_std.mean():.1f}")
print(f"   - Median prediction: {y_pred_median.mean():.1f}")

# ============================================================================
# COVERAGE ANALYSIS
# ============================================================================

print("\n" + "="*80)
print("4. COVERAGE ANALYSIS")
print("="*80)

in_50 = np.sum((y_obs >= y_pred_q25) & (y_obs <= y_pred_q75))
in_80 = np.sum((y_obs >= y_pred_q10) & (y_obs <= y_pred_q90))
in_95 = np.sum((y_obs >= y_pred_q025) & (y_obs <= y_pred_q975))

coverage_50 = 100 * in_50 / n_obs
coverage_80 = 100 * in_80 / n_obs
coverage_95 = 100 * in_95 / n_obs

print(f"\nEmpirical Coverage:")
print(f"   50% interval: {coverage_50:.1f}% ({in_50}/{n_obs}) [Expected: 50%, Exp1: {exp1_results['coverage_50']:.1f}%]")
print(f"   80% interval: {coverage_80:.1f}% ({in_80}/{n_obs}) [Expected: 80%, Exp1: {exp1_results['coverage_80']:.1f}%]")
print(f"   95% interval: {coverage_95:.1f}% ({in_95}/{n_obs}) [Expected: 95%, Exp1: {exp1_coverage_95:.1f}%]")

# Assess coverage quality
if 92 <= coverage_95 <= 96:
    coverage_quality = "GOOD"
elif 90 <= coverage_95 <= 98:
    coverage_quality = "ACCEPTABLE"
else:
    coverage_quality = "POOR"

print(f"\nCoverage Quality: {coverage_quality}")
print(f"Improvement from Exp 1: {exp1_coverage_95:.1f}% → {coverage_95:.1f}%")

# ============================================================================
# RESIDUAL ANALYSIS (CRITICAL)
# ============================================================================

print("\n" + "="*80)
print("5. RESIDUAL ANALYSIS (CRITICAL TEST)")
print("="*80)

# Compute residuals
residuals = y_obs - y_pred_mean
pearson_residuals = residuals / y_pred_std

print(f"\nResidual Statistics:")
print(f"   Mean: {residuals.mean():.2f}")
print(f"   Std: {residuals.std():.2f}")
print(f"   Min: {residuals.min():.2f}")
print(f"   Max: {residuals.max():.2f}")

# ACF calculation (custom implementation)
def acf(x, nlags=40, fft=False):
    """Calculate autocorrelation function"""
    x = np.asarray(x).squeeze()
    x = x - np.mean(x)

    c0 = np.dot(x, x) / len(x)

    acf_result = np.ones(nlags + 1)
    for k in range(1, nlags + 1):
        c_k = np.dot(x[:-k], x[k:]) / len(x)
        acf_result[k] = c_k / c0

    return acf_result

# Compute residual ACF
max_lag = min(15, n_obs // 3)
residual_acf = acf(residuals, nlags=max_lag)

print(f"\nResidual Autocorrelation (PRIMARY SUCCESS METRIC):")
print(f"   ACF(1): {residual_acf[1]:.3f} [Exp1: {exp1_acf1:.3f}, Target: < 0.3]")
print(f"   ACF(2): {residual_acf[2]:.3f}")
print(f"   ACF(3): {residual_acf[3]:.3f}")

# Improvement calculation
acf_improvement = exp1_acf1 - residual_acf[1]
acf_pct_reduction = 100 * acf_improvement / exp1_acf1

print(f"\nImprovement from Experiment 1:")
print(f"   ACF(1) reduction: {acf_improvement:.3f} ({acf_pct_reduction:.1f}% reduction)")

# Decision threshold
if residual_acf[1] < 0.2:
    temporal_decision = "EXCELLENT - AR(1) fully resolved temporal structure"
    temporal_quality = "GOOD"
elif residual_acf[1] < 0.3:
    temporal_decision = "GOOD - Residual correlation acceptable"
    temporal_quality = "ACCEPTABLE"
elif residual_acf[1] < 0.5:
    temporal_decision = "BORDERLINE - Some temporal structure remains"
    temporal_quality = "ACCEPTABLE"
else:
    temporal_decision = "POOR - AR(1) insufficient, need higher order or different structure"
    temporal_quality = "POOR"

print(f"\nTemporal Structure Assessment:")
print(f"   {temporal_decision}")
print(f"   Quality: {temporal_quality}")

# ============================================================================
# TEST STATISTICS AND BAYESIAN P-VALUES
# ============================================================================

print("\n" + "="*80)
print("6. TEST STATISTICS & BAYESIAN P-VALUES")
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

def compute_acf1(y):
    """Compute lag-1 autocorrelation"""
    acf_vals = acf(y, nlags=1)
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

n_rep = y_rep.shape[0]
for i in range(n_rep):
    if i % 500 == 0:
        print(f"   Progress: {i}/{n_rep} samples...")

    stats_i = compute_test_statistics(y_rep[i])
    for key, val in stats_i.items():
        rep_stats[key].append(val)
    rep_stats['acf1'].append(compute_acf1(y_rep[i]))

# Convert to arrays
for key in rep_stats.keys():
    rep_stats[key] = np.array(rep_stats[key])

# Compute Bayesian p-values
print("\nBayesian p-values [P(T_rep >= T_obs)]:")
print("   (Healthy range: 0.1 - 0.9)")
print()

bayesian_pvals = {}
for key in obs_stats.keys():
    pval = np.mean(rep_stats[key] >= obs_stats[key])
    bayesian_pvals[key] = pval
    flag = "***" if pval < 0.05 or pval > 0.95 else ""
    print(f"   {key:12s}: {pval:.3f} {flag}")

# ACF p-value (CRITICAL)
pval_acf1 = np.mean(rep_stats['acf1'] >= obs_acf1)
bayesian_pvals['acf1'] = pval_acf1
flag = "***" if pval_acf1 < 0.05 or pval_acf1 > 0.95 else ""
print(f"   {'ACF(1)':12s}: {pval_acf1:.3f} {flag} [Exp1: 0.000***]")

# Identify problematic statistics
problematic_stats = [k for k, v in bayesian_pvals.items() if v < 0.05 or v > 0.95]
exp1_problematic = 7  # From Exp 1 findings

print(f"\nProblematic statistics (p < 0.05 or p > 0.95): {len(problematic_stats)} [Exp1: {exp1_problematic}]")
if problematic_stats:
    for stat in problematic_stats:
        print(f"   - {stat}: p = {bayesian_pvals[stat]:.3f}")
else:
    print("   None! All test statistics in healthy range.")

# ============================================================================
# VISUALIZATIONS
# ============================================================================

print("\n" + "="*80)
print("7. GENERATING VISUALIZATIONS")
print("="*80)

# 1. COMPREHENSIVE PPC DASHBOARD
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
ax2.set_title(f'B. Coverage (95%: {coverage_95:.1f}% vs Exp1: {exp1_coverage_95:.1f}%)', fontsize=12, fontweight='bold')
ax2.legend(loc='upper left', fontsize=8)
ax2.grid(True, alpha=0.3)

# Panel C: Trajectory Plot
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

# Panel G: Residual ACF (CRITICAL COMPARISON)
ax7 = fig.add_subplot(gs[1, 2])
lags = np.arange(len(residual_acf))
ax7.bar(lags, residual_acf, width=0.3, alpha=0.7, label='Exp 3 (AR(1))')
ax7.axhline(0, color='black', lw=1)
conf_level = 1.96 / np.sqrt(n_obs)
ax7.axhline(conf_level, color='red', linestyle='--', lw=1, label='95% CI')
ax7.axhline(-conf_level, color='red', linestyle='--', lw=1)
ax7.axhline(exp1_acf1, color='orange', linestyle=':', lw=2, label=f'Exp1 ACF(1)={exp1_acf1:.3f}')
ax7.axhline(0.3, color='green', linestyle=':', lw=2, label='Target threshold')
ax7.set_xlabel('Lag', fontsize=11)
ax7.set_ylabel('ACF', fontsize=11)
ax7.set_title(f'G. Residual ACF (Exp3: {residual_acf[1]:.3f} vs Exp1: {exp1_acf1:.3f})', fontsize=12, fontweight='bold')
ax7.legend(fontsize=7)
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

# Panel K: Test Statistic - ACF(1) (CRITICAL)
ax11 = fig.add_subplot(gs[2, 2])
ax11.hist(rep_stats['acf1'], bins=50, alpha=0.6, density=True, edgecolor='black', label='Replicated')
ax11.axvline(obs_acf1, color='red', linestyle='--', lw=2, label='Observed')
ax11.axvline(0.3, color='green', linestyle=':', lw=2, label='Target threshold')
ax11.set_xlabel('ACF(1)', fontsize=11)
ax11.set_ylabel('Density', fontsize=11)
ax11.set_title(f'K. ACF(1) (p={pval_acf1:.3f}, Exp1: 0.000***)', fontsize=12, fontweight='bold')
ax11.legend(fontsize=8)
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

plt.suptitle('Posterior Predictive Check Dashboard: Latent AR(1) NegBinomial Model',
             fontsize=16, fontweight='bold', y=0.995)

plt.savefig(f'{PLOTS_DIR}/ppc_dashboard.png', dpi=300, bbox_inches='tight')
print(f"   Saved: ppc_dashboard.png")
plt.close()

# 2. ACF COMPARISON PLOT (Exp 3 vs Exp 1)
print("2. Creating ACF comparison plot (Exp 3 vs Exp 1)...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Exp 1 ACF
ax = axes[0]
exp1_residual_acf = exp1_results['residual_acf']
lags = np.arange(len(exp1_residual_acf))
ax.bar(lags, exp1_residual_acf, width=0.4, alpha=0.7, color='orange', edgecolor='black')
ax.axhline(0, color='black', lw=1)
ax.axhline(conf_level, color='red', linestyle='--', lw=1.5, label='95% CI')
ax.axhline(-conf_level, color='red', linestyle='--', lw=1.5)
ax.axhline(0.5, color='red', linestyle=':', lw=2.5, label='Phase 2 trigger (0.5)')
ax.axhline(0.3, color='orange', linestyle=':', lw=2, label='Target threshold (0.3)')
ax.set_xlabel('Lag', fontsize=12)
ax.set_ylabel('ACF', fontsize=12)
ax.set_title(f'Experiment 1: No Temporal Structure\nResidual ACF(1) = {exp1_acf1:.3f}',
             fontsize=13, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_ylim(-0.3, 1.0)

# Exp 3 ACF
ax = axes[1]
lags = np.arange(len(residual_acf))
ax.bar(lags, residual_acf, width=0.4, alpha=0.7, color='green', edgecolor='black')
ax.axhline(0, color='black', lw=1)
ax.axhline(conf_level, color='red', linestyle='--', lw=1.5, label='95% CI')
ax.axhline(-conf_level, color='red', linestyle='--', lw=1.5)
ax.axhline(0.5, color='red', linestyle=':', lw=2.5, label='Phase 2 trigger (0.5)')
ax.axhline(0.3, color='orange', linestyle=':', lw=2, label='Target threshold (0.3)')
ax.set_xlabel('Lag', fontsize=12)
ax.set_ylabel('ACF', fontsize=12)
ax.set_title(f'Experiment 3: Latent AR(1) Structure\nResidual ACF(1) = {residual_acf[1]:.3f} ({acf_pct_reduction:.1f}% reduction)',
             fontsize=13, fontweight='bold', color='green' if residual_acf[1] < 0.3 else 'orange')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_ylim(-0.3, 1.0)

plt.suptitle(f'Critical Improvement: Residual Autocorrelation\nACF(1) Improvement: {exp1_acf1:.3f} → {residual_acf[1]:.3f}',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{PLOTS_DIR}/acf_comparison_exp1_vs_exp3.png', dpi=300, bbox_inches='tight')
print(f"   Saved: acf_comparison_exp1_vs_exp3.png")
plt.close()

# 3. DETAILED COVERAGE PLOT
print("3. Creating detailed coverage plot...")

fig, ax = plt.subplots(figsize=(14, 6))
time_idx = np.arange(n_obs)

ax.fill_between(time_idx, y_pred_q025, y_pred_q975, alpha=0.15, label='95% PI', color='skyblue')
ax.fill_between(time_idx, y_pred_q10, y_pred_q90, alpha=0.25, label='80% PI', color='steelblue')
ax.fill_between(time_idx, y_pred_q25, y_pred_q75, alpha=0.35, label='50% PI', color='royalblue')

ax.plot(time_idx, y_pred_mean, 'b-', lw=2.5, label='Posterior mean', zorder=4)
ax.plot(time_idx, y_pred_median, 'b--', lw=1.5, label='Posterior median', alpha=0.6, zorder=3)

ax.scatter(time_idx, y_obs, color='red', s=60, zorder=5, label='Observed',
           edgecolors='darkred', linewidth=1.5, marker='o')

# Highlight points outside 95% interval
outside_95 = (y_obs < y_pred_q025) | (y_obs > y_pred_q975)
if np.any(outside_95):
    ax.scatter(time_idx[outside_95], y_obs[outside_95], color='orange', s=120,
               zorder=6, label='Outside 95% PI', marker='X', edgecolors='darkorange', linewidth=2)

ax.set_xlabel('Time Index', fontsize=13)
ax.set_ylabel('Count', fontsize=13)
ax.set_title(f'Posterior Predictive Coverage: AR(1) Model\n95%: {coverage_95:.1f}% (Exp1: {exp1_coverage_95:.1f}%), 80%: {coverage_80:.1f}%, 50%: {coverage_50:.1f}%',
             fontsize=14, fontweight='bold')
ax.legend(loc='upper left', fontsize=10, ncol=2)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{PLOTS_DIR}/coverage_detailed.png', dpi=300, bbox_inches='tight')
print(f"   Saved: coverage_detailed.png")
plt.close()

# 4. RESIDUAL DIAGNOSTICS SUITE
print("4. Creating residual diagnostics suite...")

fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# A: Residuals vs Fitted
ax = axes[0, 0]
ax.scatter(y_pred_mean, residuals, alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
ax.axhline(0, color='red', linestyle='--', lw=2)
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
try:
    smooth = savgol_filter(residuals, window_length=min(11, n_obs//3*2-1), polyorder=2)
    ax.plot(time_idx, smooth, 'g-', lw=2, label='Smooth trend')
except:
    pass
ax.set_xlabel('Time Index', fontsize=11)
ax.set_ylabel('Residuals', fontsize=11)
ax.set_title('B. Residuals vs Time (Should be flat)', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# C: Residual ACF
ax = axes[0, 2]
lags = np.arange(len(residual_acf))
ax.bar(lags, residual_acf, width=0.5, alpha=0.7, edgecolor='black')
ax.axhline(0, color='black', lw=1)
ax.axhline(conf_level, color='red', linestyle='--', lw=1.5, label='95% CI')
ax.axhline(-conf_level, color='red', linestyle='--', lw=1.5)
ax.axhline(0.3, color='green', linestyle=':', lw=2, label='Target (0.3)')
ax.axhline(exp1_acf1, color='orange', linestyle=':', lw=2, label=f'Exp1 ({exp1_acf1:.3f})')
ax.set_xlabel('Lag', fontsize=11)
ax.set_ylabel('ACF', fontsize=11)
ax.set_title(f'C. Residual ACF (Improvement: {acf_pct_reduction:.1f}%)', fontsize=12, fontweight='bold')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# D: Histogram of residuals
ax = axes[1, 0]
ax.hist(residuals, bins=20, alpha=0.6, density=True, edgecolor='black', label='Residuals')
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
ax.set_ylabel('sqrt|Standardized Residuals|', fontsize=11)
ax.set_title('F. Scale-Location Plot', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

plt.suptitle('Residual Diagnostics Suite: AR(1) Model', fontsize=14, fontweight='bold', y=0.998)
plt.tight_layout()
plt.savefig(f'{PLOTS_DIR}/residual_diagnostics.png', dpi=300, bbox_inches='tight')
print(f"   Saved: residual_diagnostics.png")
plt.close()

# 5. TEST STATISTICS COMPARISON
print("5. Creating test statistics comparison plot...")

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

    ax.hist(rep_vals, bins=50, alpha=0.6, density=True, edgecolor='black', label='Replicated')
    ax.axvline(obs_val, color='red', linestyle='--', lw=2.5, label='Observed')

    percentile = 100 * np.mean(rep_vals <= obs_val)

    if pval < 0.05 or pval > 0.95:
        color = 'red'
        result = 'POOR'
    elif pval < 0.1 or pval > 0.9:
        color = 'orange'
        result = 'OK'
    else:
        color = 'green'
        result = 'GOOD'

    ax.set_xlabel(stat.capitalize(), fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title(f'{stat.capitalize()}\np-value: {pval:.3f} ({result})',
                 fontsize=12, fontweight='bold', color=color)
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax.text(0.02, 0.98, f'Obs at {percentile:.1f}th percentile',
            transform=ax.transAxes, fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.suptitle('Test Statistics: Observed vs Posterior Predictive Distribution',
             fontsize=14, fontweight='bold', y=0.998)
plt.tight_layout()
plt.savefig(f'{PLOTS_DIR}/test_statistics.png', dpi=300, bbox_inches='tight')
print(f"   Saved: test_statistics.png")
plt.close()

# ============================================================================
# SAVE RESULTS
# ============================================================================

print("\n" + "="*80)
print("8. SAVING RESULTS")
print("="*80)

results = {
    'coverage_50': coverage_50,
    'coverage_80': coverage_80,
    'coverage_95': coverage_95,
    'coverage_quality': coverage_quality,
    'residual_acf1': residual_acf[1],
    'residual_acf': residual_acf,
    'temporal_decision': temporal_decision,
    'temporal_quality': temporal_quality,
    'bayesian_pvals': bayesian_pvals,
    'problematic_stats': problematic_stats,
    'obs_stats': obs_stats,
    'obs_acf1': obs_acf1,
    'exp1_acf1': exp1_acf1,
    'exp1_coverage_95': exp1_coverage_95,
    'acf_improvement': acf_improvement,
    'acf_pct_reduction': acf_pct_reduction,
}

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

fit_criteria = []

# Coverage
if 92 <= coverage_95 <= 96:
    fit_criteria.append(('Coverage (95%)', 'GOOD', f'{coverage_95:.1f}% (optimal: 92-96%)'))
elif 90 <= coverage_95 <= 98:
    fit_criteria.append(('Coverage (95%)', 'ACCEPTABLE', f'{coverage_95:.1f}% (target: 90-98%)'))
else:
    fit_criteria.append(('Coverage (95%)', 'POOR', f'{coverage_95:.1f}%'))

# Residual ACF (PRIMARY METRIC)
if residual_acf[1] < 0.2:
    fit_criteria.append(('Residual ACF(1)', 'GOOD', f'{residual_acf[1]:.3f} (< 0.2, excellent)'))
elif residual_acf[1] < 0.3:
    fit_criteria.append(('Residual ACF(1)', 'ACCEPTABLE', f'{residual_acf[1]:.3f} (< 0.3, target met)'))
elif residual_acf[1] < 0.5:
    fit_criteria.append(('Residual ACF(1)', 'ACCEPTABLE', f'{residual_acf[1]:.3f} (borderline)'))
else:
    fit_criteria.append(('Residual ACF(1)', 'POOR', f'{residual_acf[1]:.3f} (> 0.5, needs work)'))

# P-values
n_extreme_pvals = len(problematic_stats)
if n_extreme_pvals == 0:
    fit_criteria.append(('P-values', 'GOOD', 'No extreme p-values'))
elif n_extreme_pvals <= 2:
    fit_criteria.append(('P-values', 'ACCEPTABLE', f'{n_extreme_pvals} extreme (≤2 ok)'))
else:
    fit_criteria.append(('P-values', 'POOR', f'{n_extreme_pvals} extreme (>2)'))

print("\nFit Criteria Summary:")
print(f"{'Criterion':<25} {'Quality':<15} {'Detail'}")
print("-" * 80)
for criterion, quality, detail in fit_criteria:
    print(f"{criterion:<25} {quality:<15} {detail}")

# Improvement from Exp 1
print("\n" + "-" * 80)
print("IMPROVEMENT FROM EXPERIMENT 1:")
print("-" * 80)
print(f"  Residual ACF(1):  {exp1_acf1:.3f} → {residual_acf[1]:.3f} ({acf_pct_reduction:.1f}% reduction)")
print(f"  Coverage (95%):   {exp1_coverage_95:.1f}% → {coverage_95:.1f}%")
print(f"  Extreme p-values: {exp1_problematic} → {n_extreme_pvals}")

# Final decision
good_count = sum(1 for _, q, _ in fit_criteria if q == 'GOOD')
acceptable_count = sum(1 for _, q, _ in fit_criteria if q == 'ACCEPTABLE')
poor_count = sum(1 for _, q, _ in fit_criteria if q == 'POOR')

if residual_acf[1] < 0.2 and 90 <= coverage_95 <= 98:
    overall_fit = 'GOOD'
elif residual_acf[1] < 0.3 and 88 <= coverage_95 <= 100:
    overall_fit = 'ACCEPTABLE'
else:
    overall_fit = 'POOR'

print("\n" + "="*80)
print(f"OVERALL FIT QUALITY: {overall_fit}")
print("="*80)

print(f"\nCriteria met: {good_count} GOOD, {acceptable_count} ACCEPTABLE, {poor_count} POOR")

# Success assessment
print("\n" + "="*80)
print("SUCCESS METRICS ASSESSMENT")
print("="*80)

success_metrics = [
    ("Residual ACF(1) < 0.3", residual_acf[1] < 0.3, f"{residual_acf[1]:.3f}"),
    ("Coverage 90-98%", 90 <= coverage_95 <= 98, f"{coverage_95:.1f}%"),
    ("No extreme p-values (≤2)", n_extreme_pvals <= 2, f"{n_extreme_pvals}"),
    ("Improvement vs Exp 1", acf_pct_reduction > 50, f"{acf_pct_reduction:.1f}% reduction"),
]

print(f"\n{'Metric':<35} {'Status':<10} {'Value'}")
print("-" * 80)
for metric, passed, value in success_metrics:
    status = "PASS" if passed else "FAIL"
    print(f"{metric:<35} {status:<10} {value}")

all_passed = all(passed for _, passed, _ in success_metrics)
print("\n" + "="*80)
if all_passed:
    print("SUCCESS: All critical metrics met! AR(1) model successfully addresses")
    print("         temporal autocorrelation from Experiment 1.")
else:
    print("PARTIAL SUCCESS: Some metrics not met. Review specific deficiencies.")
print("="*80)

print("\n" + "="*80)
print("POSTERIOR PREDICTIVE CHECKS COMPLETE")
print("="*80)
print(f"\nResults saved to: {PLOTS_DIR}/../")
print("\nGenerated plots:")
print("   - ppc_dashboard.png (comprehensive 12-panel overview)")
print("   - acf_comparison_exp1_vs_exp3.png (CRITICAL: shows improvement)")
print("   - coverage_detailed.png (detailed coverage assessment)")
print("   - residual_diagnostics.png (6-panel residual suite)")
print("   - test_statistics.png (6 key test statistics)")
print("\nKey findings:")
print(f"   - Residual ACF(1): {exp1_acf1:.3f} → {residual_acf[1]:.3f} ({acf_pct_reduction:.1f}% reduction)")
print(f"   - Coverage: {exp1_coverage_95:.1f}% → {coverage_95:.1f}%")
print(f"   - Overall fit: {overall_fit}")
print(f"   - Temporal quality: {temporal_quality}")
