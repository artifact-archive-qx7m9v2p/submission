"""
Simulation-Based Validation for Experiment 1: Logarithmic Model

Test if optimization can recover known parameters when truth is known.
Uses maximum likelihood estimation with bootstrap for uncertainty.

This is a critical safety check before fitting real data.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.optimize import minimize
from pathlib import Path
import json

# Set random seed for reproducibility
np.random.seed(42)

# Setup paths
BASE_DIR = Path("/workspace/experiments/experiment_1/simulation_based_validation")
CODE_DIR = BASE_DIR / "code"
PLOTS_DIR = BASE_DIR / "plots"
DATA_PATH = Path("/workspace/data/data.csv")

# Ensure plots directory exists
PLOTS_DIR.mkdir(exist_ok=True, parents=True)

print("=" * 80)
print("SIMULATION-BASED VALIDATION: Parameter Recovery Test")
print("=" * 80)

# ============================================================================
# STEP 1: Load real x values and set true parameters
# ============================================================================

print("\n[1] Loading real x values and setting true parameters...")

# Load real data to get x values
real_data = pd.read_csv(DATA_PATH)
x_values = real_data['x'].values
N = len(x_values)

print(f"   - Sample size: N = {N}")
print(f"   - x range: [{x_values.min():.1f}, {x_values.max():.1f}]")

# Set TRUE parameters (close to EDA estimates for realism)
beta_0_true = 2.3
beta_1_true = 0.29
sigma_true = 0.09

true_params = {
    'beta_0': beta_0_true,
    'beta_1': beta_1_true,
    'sigma': sigma_true
}

print(f"\n   TRUE PARAMETERS:")
print(f"   - β₀ (intercept):     {beta_0_true}")
print(f"   - β₁ (log slope):     {beta_1_true}")
print(f"   - σ (residual SD):    {sigma_true}")

# ============================================================================
# STEP 2: Generate synthetic data from the model
# ============================================================================

print("\n[2] Generating synthetic data from known model...")

# Generate synthetic Y values
mu_true = beta_0_true + beta_1_true * np.log(x_values)
Y_synthetic = np.random.normal(mu_true, sigma_true)

print(f"   - Generated {N} synthetic observations")
print(f"   - Y range: [{Y_synthetic.min():.3f}, {Y_synthetic.max():.3f}]")
print(f"   - Mean Y: {Y_synthetic.mean():.3f}")
print(f"   - SD Y: {Y_synthetic.std():.3f}")

# Save synthetic data for inspection
synthetic_data = pd.DataFrame({
    'x': x_values,
    'Y_synthetic': Y_synthetic,
    'mu_true': mu_true
})
synthetic_data.to_csv(CODE_DIR / "synthetic_data.csv", index=False)
print(f"   - Saved to: {CODE_DIR / 'synthetic_data.csv'}")

# ============================================================================
# STEP 3: Define negative log-likelihood for optimization
# ============================================================================

def neg_log_likelihood(params, x, y):
    """Negative log-likelihood for logarithmic model with Normal errors."""
    beta_0, beta_1, log_sigma = params
    sigma = np.exp(log_sigma)  # Ensure positive

    # Predicted values
    mu = beta_0 + beta_1 * np.log(x)

    # Negative log-likelihood (Normal distribution)
    nll = -np.sum(stats.norm.logpdf(y, loc=mu, scale=sigma))

    return nll

# ============================================================================
# STEP 4: Fit model to synthetic data using MLE
# ============================================================================

print("\n[3] Fitting model to synthetic data using Maximum Likelihood...")

# Initial guess (close to prior means)
initial_params = [2.3, 0.29, np.log(0.1)]

# Optimize
result = minimize(
    neg_log_likelihood,
    initial_params,
    args=(x_values, Y_synthetic),
    method='L-BFGS-B',
    bounds=[(-5, 5), (-2, 2), (np.log(0.001), np.log(1))]
)

# Extract MLE estimates
beta_0_mle = result.x[0]
beta_1_mle = result.x[1]
sigma_mle = np.exp(result.x[2])

print(f"   - Optimization successful: {result.success}")
print(f"   - MLE estimates:")
print(f"     β₀ = {beta_0_mle:.4f}")
print(f"     β₁ = {beta_1_mle:.4f}")
print(f"     σ  = {sigma_mle:.4f}")

# ============================================================================
# STEP 5: Bootstrap for uncertainty quantification
# ============================================================================

print("\n[4] Running bootstrap for uncertainty quantification...")

n_bootstrap = 1000
bootstrap_results = np.zeros((n_bootstrap, 3))

print(f"   - Running {n_bootstrap} bootstrap iterations...")

for i in range(n_bootstrap):
    if (i + 1) % 200 == 0:
        print(f"     Iteration {i + 1}/{n_bootstrap}")

    # Resample indices
    boot_indices = np.random.choice(N, size=N, replace=True)
    x_boot = x_values[boot_indices]
    y_boot = Y_synthetic[boot_indices]

    # Fit to bootstrap sample
    try:
        boot_result = minimize(
            neg_log_likelihood,
            initial_params,
            args=(x_boot, y_boot),
            method='L-BFGS-B',
            bounds=[(-5, 5), (-2, 2), (np.log(0.001), np.log(1))],
            options={'maxiter': 100}
        )

        bootstrap_results[i, 0] = boot_result.x[0]
        bootstrap_results[i, 1] = boot_result.x[1]
        bootstrap_results[i, 2] = np.exp(boot_result.x[2])
    except:
        # If optimization fails, use previous values
        bootstrap_results[i] = bootstrap_results[i-1] if i > 0 else [beta_0_mle, beta_1_mle, sigma_mle]

print("   - Bootstrap complete!")

# ============================================================================
# STEP 6: Check parameter recovery
# ============================================================================

print("\n" + "=" * 80)
print("PARAMETER RECOVERY ASSESSMENT")
print("=" * 80)

param_names = ['beta_0', 'beta_1', 'sigma']
recovery_results = {}

for idx, param_name in enumerate(param_names):
    true_value = true_params[param_name]
    mle_value = [beta_0_mle, beta_1_mle, sigma_mle][idx]
    boot_samples = bootstrap_results[:, idx]

    # Remove any NaN or extreme values
    boot_samples = boot_samples[np.isfinite(boot_samples)]
    boot_samples = boot_samples[(boot_samples > np.percentile(boot_samples, 0.5)) &
                                (boot_samples < np.percentile(boot_samples, 99.5))]

    mean_est = boot_samples.mean()
    median_est = np.median(boot_samples)
    sd_est = boot_samples.std()

    # 95% confidence interval from bootstrap
    ci_lower = np.percentile(boot_samples, 2.5)
    ci_upper = np.percentile(boot_samples, 97.5)

    # Coverage: is true value in 95% CI?
    in_ci = ci_lower <= true_value <= ci_upper

    # Standardized error: |mean - true| / SD
    z_score = abs(mean_est - true_value) / sd_est if sd_est > 0 else 0

    # Bias
    bias = mean_est - true_value
    relative_bias = bias / true_value * 100 if true_value != 0 else 0

    recovery_results[param_name] = {
        'true': true_value,
        'mle': mle_value,
        'mean': mean_est,
        'median': median_est,
        'sd': sd_est,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'in_ci': in_ci,
        'z_score': z_score,
        'bias': bias,
        'relative_bias_pct': relative_bias,
        'samples': boot_samples
    }

    print(f"\n{param_name}:")
    print(f"   True value:       {true_value:.4f}")
    print(f"   MLE estimate:     {mle_value:.4f}")
    print(f"   Bootstrap mean:   {mean_est:.4f}")
    print(f"   Bootstrap median: {median_est:.4f}")
    print(f"   Bootstrap SD:     {sd_est:.4f}")
    print(f"   95% CI:           [{ci_lower:.4f}, {ci_upper:.4f}]")
    print(f"   In CI?            {in_ci} {'✓' if in_ci else '✗ FAIL'}")
    print(f"   Bias:             {bias:.4f} ({relative_bias:.2f}%)")
    print(f"   |z-score|:        {z_score:.2f} {'✓' if z_score < 2 else '✗ WARNING'}")

# ============================================================================
# STEP 7: Compute model diagnostics
# ============================================================================

print("\n" + "=" * 80)
print("MODEL DIAGNOSTICS")
print("=" * 80)

# Compute fitted values and residuals
mu_fitted = beta_0_mle + beta_1_mle * np.log(x_values)
residuals = Y_synthetic - mu_fitted
rmse = np.sqrt(np.mean(residuals**2))
r_squared = 1 - np.sum(residuals**2) / np.sum((Y_synthetic - Y_synthetic.mean())**2)

print(f"\nGoodness of Fit:")
print(f"   RMSE:      {rmse:.4f}")
print(f"   R²:        {r_squared:.4f}")
print(f"   True σ:    {sigma_true:.4f}")
print(f"   MLE σ:     {sigma_mle:.4f}")

# Check residual normality
shapiro_stat, shapiro_p = stats.shapiro(residuals)
print(f"\nResidual Diagnostics:")
print(f"   Shapiro-Wilk test: p = {shapiro_p:.4f} {'(Normal)' if shapiro_p > 0.05 else '(Non-normal)'}")

# ============================================================================
# STEP 8: Visualizations
# ============================================================================

print("\n[5] Creating diagnostic visualizations...")

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.facecolor'] = 'white'

# -------------------------------------------------------
# Plot 1: Parameter Recovery (True vs Bootstrap Distribution)
# -------------------------------------------------------

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

param_labels = {
    'beta_0': r'$\beta_0$ (Intercept)',
    'beta_1': r'$\beta_1$ (Log Slope)',
    'sigma': r'$\sigma$ (Residual SD)'
}

for idx, param_name in enumerate(param_names):
    ax = axes[idx]
    samples = recovery_results[param_name]['samples']
    true_value = recovery_results[param_name]['true']

    # Histogram of bootstrap distribution
    ax.hist(samples, bins=50, density=True, alpha=0.6, color='steelblue',
            edgecolor='black', linewidth=0.5)

    # KDE
    kde = stats.gaussian_kde(samples)
    x_range = np.linspace(samples.min(), samples.max(), 200)
    ax.plot(x_range, kde(x_range), 'b-', linewidth=2, label='Bootstrap')

    # True value
    ax.axvline(true_value, color='red', linestyle='--', linewidth=2.5,
               label='True Value', zorder=10)

    # 95% CI
    ci_lower = recovery_results[param_name]['ci_lower']
    ci_upper = recovery_results[param_name]['ci_upper']
    ax.axvline(ci_lower, color='gray', linestyle=':', linewidth=1.5, alpha=0.7)
    ax.axvline(ci_upper, color='gray', linestyle=':', linewidth=1.5, alpha=0.7)

    # MLE estimate
    mle_est = recovery_results[param_name]['mle']
    ax.axvline(mle_est, color='blue', linestyle='-', linewidth=2, alpha=0.7,
               label='MLE')

    ax.set_xlabel(param_labels[param_name], fontsize=12, fontweight='bold')
    ax.set_ylabel('Density', fontsize=11)
    ax.legend(fontsize=9, loc='best')

    # Color code title based on recovery success
    title_color = 'green' if recovery_results[param_name]['in_ci'] else 'red'
    recovery_status = 'RECOVERED' if recovery_results[param_name]['in_ci'] else 'FAILED'
    ax.set_title(f"{recovery_status}", fontsize=11, fontweight='bold', color=title_color)

plt.suptitle('Parameter Recovery Assessment (Bootstrap Distribution)',
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "parameter_recovery.png", dpi=300, bbox_inches='tight')
print(f"   - Saved: {PLOTS_DIR / 'parameter_recovery.png'}")
plt.close()

# -------------------------------------------------------
# Plot 2: Prior vs Bootstrap Distribution
# -------------------------------------------------------

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Prior specifications
priors = {
    'beta_0': ('normal', [2.3, 0.3]),
    'beta_1': ('normal', [0.29, 0.15]),
    'sigma': ('exponential', [10])  # rate parameter
}

for idx, param_name in enumerate(param_names):
    ax = axes[idx]
    samples = recovery_results[param_name]['samples']
    dist_type, params = priors[param_name]

    # Bootstrap distribution
    ax.hist(samples, bins=50, density=True, alpha=0.5, color='blue',
            label='Bootstrap', edgecolor='black', linewidth=0.5)

    # Prior
    if dist_type == 'normal':
        x_range = np.linspace(params[0] - 4*params[1], params[0] + 4*params[1], 200)
        prior_density = stats.norm.pdf(x_range, params[0], params[1])
        ax.plot(x_range, prior_density, 'r--', linewidth=2, label='Prior')
    elif dist_type == 'exponential':
        x_range = np.linspace(0, 0.5, 200)
        prior_density = stats.expon.pdf(x_range, scale=1/params[0])
        ax.plot(x_range, prior_density, 'r--', linewidth=2, label='Prior')

    # True value
    true_value = true_params[param_name]
    ax.axvline(true_value, color='green', linestyle='-.', linewidth=2.5,
               label='True Value', zorder=10)

    ax.set_xlabel(param_labels[param_name], fontsize=12, fontweight='bold')
    ax.set_ylabel('Density', fontsize=11)
    ax.legend(fontsize=10, loc='best')
    ax.set_title('Data Concentrates Uncertainty', fontsize=11, fontweight='bold')

plt.suptitle('Prior vs Data: Learning from Observations',
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "prior_posterior_comparison.png", dpi=300, bbox_inches='tight')
print(f"   - Saved: {PLOTS_DIR / 'prior_posterior_comparison.png'}")
plt.close()

# -------------------------------------------------------
# Plot 3: Bootstrap Convergence
# -------------------------------------------------------

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for idx, param_name in enumerate(param_names):
    ax = axes[idx]

    # Cumulative mean
    samples = bootstrap_results[:, idx]
    cumulative_mean = np.cumsum(samples) / np.arange(1, len(samples) + 1)

    ax.plot(cumulative_mean, 'b-', linewidth=1.5, alpha=0.7)
    ax.axhline(true_params[param_name], color='red', linestyle='--',
               linewidth=2, label='True Value')
    ax.axhline(recovery_results[param_name]['mle'], color='green', linestyle=':',
               linewidth=2, label='MLE')

    ax.set_xlabel('Bootstrap Iteration', fontsize=11, fontweight='bold')
    ax.set_ylabel(param_labels[param_name], fontsize=11, fontweight='bold')
    ax.legend(fontsize=9, loc='best')
    ax.set_title('Bootstrap Convergence', fontsize=10, fontweight='bold')
    ax.grid(True, alpha=0.3)

plt.suptitle('Bootstrap Stability Check', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "convergence_diagnostics.png", dpi=300, bbox_inches='tight')
print(f"   - Saved: {PLOTS_DIR / 'convergence_diagnostics.png'}")
plt.close()

# -------------------------------------------------------
# Plot 4: Correlation Structure (Bivariate Scatter)
# -------------------------------------------------------

fig, axes = plt.subplots(2, 2, figsize=(12, 12))

# Beta_0 vs Beta_1
ax = axes[0, 0]
ax.scatter(bootstrap_results[:, 0], bootstrap_results[:, 1],
          alpha=0.3, s=20, color='steelblue')
ax.axvline(beta_0_true, color='red', linestyle='--', linewidth=2, alpha=0.7)
ax.axhline(beta_1_true, color='red', linestyle='--', linewidth=2, alpha=0.7)
ax.plot(beta_0_true, beta_1_true, 'r*', markersize=20, label='True Values')
ax.set_xlabel(r'$\beta_0$', fontsize=12, fontweight='bold')
ax.set_ylabel(r'$\beta_1$', fontsize=12, fontweight='bold')
corr_01 = np.corrcoef(bootstrap_results[:, 0], bootstrap_results[:, 1])[0, 1]
ax.set_title(f'Correlation = {corr_01:.3f}', fontsize=11, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# Beta_0 vs Sigma
ax = axes[0, 1]
ax.scatter(bootstrap_results[:, 0], bootstrap_results[:, 2],
          alpha=0.3, s=20, color='steelblue')
ax.axvline(beta_0_true, color='red', linestyle='--', linewidth=2, alpha=0.7)
ax.axhline(sigma_true, color='red', linestyle='--', linewidth=2, alpha=0.7)
ax.plot(beta_0_true, sigma_true, 'r*', markersize=20, label='True Values')
ax.set_xlabel(r'$\beta_0$', fontsize=12, fontweight='bold')
ax.set_ylabel(r'$\sigma$', fontsize=12, fontweight='bold')
corr_02 = np.corrcoef(bootstrap_results[:, 0], bootstrap_results[:, 2])[0, 1]
ax.set_title(f'Correlation = {corr_02:.3f}', fontsize=11, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# Beta_1 vs Sigma
ax = axes[1, 0]
ax.scatter(bootstrap_results[:, 1], bootstrap_results[:, 2],
          alpha=0.3, s=20, color='steelblue')
ax.axvline(beta_1_true, color='red', linestyle='--', linewidth=2, alpha=0.7)
ax.axhline(sigma_true, color='red', linestyle='--', linewidth=2, alpha=0.7)
ax.plot(beta_1_true, sigma_true, 'r*', markersize=20, label='True Values')
ax.set_xlabel(r'$\beta_1$', fontsize=12, fontweight='bold')
ax.set_ylabel(r'$\sigma$', fontsize=12, fontweight='bold')
corr_12 = np.corrcoef(bootstrap_results[:, 1], bootstrap_results[:, 2])[0, 1]
ax.set_title(f'Correlation = {corr_12:.3f}', fontsize=11, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# Marginal distributions
ax = axes[1, 1]
ax.text(0.5, 0.7, 'Parameter Correlations', ha='center', fontsize=14, fontweight='bold',
        transform=ax.transAxes)
ax.text(0.5, 0.5, f'ρ(β₀, β₁) = {corr_01:.3f}', ha='center', fontsize=12,
        transform=ax.transAxes)
ax.text(0.5, 0.4, f'ρ(β₀, σ) = {corr_02:.3f}', ha='center', fontsize=12,
        transform=ax.transAxes)
ax.text(0.5, 0.3, f'ρ(β₁, σ) = {corr_12:.3f}', ha='center', fontsize=12,
        transform=ax.transAxes)
ax.axis('off')

plt.suptitle('Parameter Correlation Structure', fontsize=14, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "posterior_pairs.png", dpi=300, bbox_inches='tight')
print(f"   - Saved: {PLOTS_DIR / 'posterior_pairs.png'}")
plt.close()

# -------------------------------------------------------
# Plot 5: Fit to Synthetic Data
# -------------------------------------------------------

fig, ax = plt.subplots(figsize=(10, 6))

# Create smooth x range for plotting
x_plot = np.linspace(x_values.min(), x_values.max(), 200)

# Plot bootstrap draws (sample 100)
sample_indices = np.random.choice(n_bootstrap, size=100, replace=False)
for i in sample_indices:
    mu_draw = bootstrap_results[i, 0] + bootstrap_results[i, 1] * np.log(x_plot)
    ax.plot(x_plot, mu_draw, 'b-', alpha=0.02, linewidth=0.5)

# Plot true mean function
mu_true_plot = beta_0_true + beta_1_true * np.log(x_plot)
ax.plot(x_plot, mu_true_plot, 'r--', linewidth=3, label='True Mean Function', zorder=10)

# Plot synthetic data
ax.scatter(x_values, Y_synthetic, c='black', s=80, alpha=0.7,
          edgecolors='white', linewidth=1.5, label='Synthetic Data', zorder=5)

# Plot MLE mean function
mu_mle = beta_0_mle + beta_1_mle * np.log(x_plot)
ax.plot(x_plot, mu_mle, 'b-', linewidth=3, label='MLE Fit', zorder=8)

# Add uncertainty band (±1.96 σ)
ax.fill_between(x_plot, mu_mle - 1.96*sigma_mle, mu_mle + 1.96*sigma_mle,
                alpha=0.2, color='blue', label='95% Prediction Interval')

ax.set_xlabel('x', fontsize=13, fontweight='bold')
ax.set_ylabel('Y', fontsize=13, fontweight='bold')
ax.set_title('Model Fit to Synthetic Data\n(Blue: Bootstrap Uncertainty, Red: True Function)',
            fontsize=12, fontweight='bold')
ax.legend(fontsize=11, loc='best')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(PLOTS_DIR / "synthetic_data_fit.png", dpi=300, bbox_inches='tight')
print(f"   - Saved: {PLOTS_DIR / 'synthetic_data_fit.png'}")
plt.close()

# -------------------------------------------------------
# Plot 6: Residual Diagnostics
# -------------------------------------------------------

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Residuals vs Fitted
ax = axes[0, 0]
ax.scatter(mu_fitted, residuals, alpha=0.7, s=60, edgecolors='black', linewidth=0.5)
ax.axhline(0, color='red', linestyle='--', linewidth=2)
ax.axhline(2*sigma_mle, color='gray', linestyle=':', linewidth=1)
ax.axhline(-2*sigma_mle, color='gray', linestyle=':', linewidth=1)
ax.set_xlabel('Fitted Values', fontsize=11, fontweight='bold')
ax.set_ylabel('Residuals', fontsize=11, fontweight='bold')
ax.set_title('Residuals vs Fitted', fontsize=11, fontweight='bold')
ax.grid(True, alpha=0.3)

# QQ Plot
ax = axes[0, 1]
stats.probplot(residuals, dist="norm", plot=ax)
ax.set_title('Normal Q-Q Plot', fontsize=11, fontweight='bold')
ax.grid(True, alpha=0.3)

# Histogram of Residuals
ax = axes[1, 0]
ax.hist(residuals, bins=15, density=True, alpha=0.6, color='steelblue',
        edgecolor='black', linewidth=0.5)
x_range = np.linspace(residuals.min(), residuals.max(), 100)
ax.plot(x_range, stats.norm.pdf(x_range, 0, sigma_mle), 'r-', linewidth=2,
        label='N(0, σ_MLE)')
ax.set_xlabel('Residuals', fontsize=11, fontweight='bold')
ax.set_ylabel('Density', fontsize=11)
ax.set_title(f'Residual Distribution (Shapiro p={shapiro_p:.3f})',
             fontsize=11, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# Residuals vs x
ax = axes[1, 1]
ax.scatter(x_values, residuals, alpha=0.7, s=60, edgecolors='black', linewidth=0.5)
ax.axhline(0, color='red', linestyle='--', linewidth=2)
ax.axhline(2*sigma_mle, color='gray', linestyle=':', linewidth=1)
ax.axhline(-2*sigma_mle, color='gray', linestyle=':', linewidth=1)
ax.set_xlabel('x', fontsize=11, fontweight='bold')
ax.set_ylabel('Residuals', fontsize=11, fontweight='bold')
ax.set_title('Residuals vs x', fontsize=11, fontweight='bold')
ax.grid(True, alpha=0.3)

plt.suptitle('Residual Diagnostics', fontsize=14, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "residual_diagnostics.png", dpi=300, bbox_inches='tight')
print(f"   - Saved: {PLOTS_DIR / 'residual_diagnostics.png'}")
plt.close()

# ============================================================================
# STEP 9: Overall Assessment and Decision
# ============================================================================

print("\n" + "=" * 80)
print("OVERALL VALIDATION ASSESSMENT")
print("=" * 80)

# Check all criteria
all_in_ci = all(recovery_results[p]['in_ci'] for p in param_names)
all_z_scores_ok = all(recovery_results[p]['z_score'] < 2 for p in param_names)
residuals_normal = shapiro_p > 0.05
fit_quality_ok = r_squared > 0.8 and rmse < 0.15

# Overall decision
validation_pass = all_in_ci and all_z_scores_ok and residuals_normal and fit_quality_ok

print(f"\nParameter Recovery Tests:")
print(f"   - All true values in 95% CIs:  {all_in_ci} {'✓' if all_in_ci else '✗'}")
print(f"   - All |z-scores| < 2:         {all_z_scores_ok} {'✓' if all_z_scores_ok else '✗'}")

print(f"\nModel Quality Tests:")
print(f"   - Residuals normally distributed: {residuals_normal} {'✓' if residuals_normal else '✗'}")
print(f"   - Good fit (R² > 0.8, RMSE < 0.15): {fit_quality_ok} {'✓' if fit_quality_ok else '✗'}")

print(f"\n" + "=" * 80)
if validation_pass:
    print("VALIDATION RESULT: PASS")
    print("=" * 80)
    print("\nThe model successfully recovered all true parameters with proper")
    print("uncertainty quantification and no computational issues.")
    print("\nRECOMMENDATION: PROCEED TO REAL DATA FITTING")
else:
    print("VALIDATION RESULT: FAIL")
    print("=" * 80)
    print("\nThe model failed to recover known parameters. Issues detected:")
    if not all_in_ci:
        print("   - True values outside 95% confidence intervals")
    if not all_z_scores_ok:
        print("   - Large standardized errors (|z| > 2)")
    if not residuals_normal:
        print("   - Residuals not normally distributed")
    if not fit_quality_ok:
        print("   - Poor model fit quality")
    print("\nRECOMMENDATION: DO NOT PROCEED - REVISE MODEL OR SAMPLING STRATEGY")

# Save results to JSON
results_dict = {
    'validation_pass': validation_pass,
    'true_parameters': true_params,
    'recovery_results': {k: {kk: float(vv) if isinstance(vv, (np.floating, np.integer))
                            else (vv.tolist() if isinstance(vv, np.ndarray) else vv)
                            for kk, vv in v.items() if kk != 'samples'}
                        for k, v in recovery_results.items()},
    'fit_diagnostics': {
        'r_squared': float(r_squared),
        'rmse': float(rmse),
        'shapiro_p_value': float(shapiro_p),
        'n_bootstrap': n_bootstrap
    }
}

with open(CODE_DIR / "validation_results.json", 'w') as f:
    json.dump(results_dict, f, indent=2)

print(f"\nResults saved to: {CODE_DIR / 'validation_results.json'}")
print(f"All plots saved to: {PLOTS_DIR}/")

print("\n" + "=" * 80)
print("SIMULATION-BASED VALIDATION COMPLETE")
print("=" * 80)
