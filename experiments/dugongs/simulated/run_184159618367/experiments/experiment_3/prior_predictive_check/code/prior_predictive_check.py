"""
Prior Predictive Check for Log-Log Power Law Model (Experiment 3)

Model:
    log(Y) ~ Normal(α + β*log(x), σ)
    Equivalently: Y = exp(α) * x^β * exp(ε), where ε ~ Normal(0, σ)

Priors:
    α ~ Normal(0.6, 0.3)
    β ~ Normal(0.12, 0.1)
    σ ~ Half-Cauchy(0, 0.1)

Validation Goals:
    1. Check if prior predictions are scientifically plausible
    2. Verify Y stays mostly in [0.5, 5.0] (observed range with buffer)
    3. Ensure monotonic increasing behavior (since β > 0 likely)
    4. Check for computational issues (extreme values, NaNs)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings

# Set random seed for reproducibility
np.random.seed(42)

# Configuration
N_PRIOR_DRAWS = 2000  # Number of prior samples
N_PRED_POINTS = 100   # Number of x values for prediction
X_MIN, X_MAX = 1.0, 35.0  # Prediction range (slightly beyond observed)

# Load observed data for context
data = pd.read_csv('/workspace/data/data.csv')
x_obs = data['x'].values
y_obs = data['Y'].values

print("="*70)
print("PRIOR PREDICTIVE CHECK: Log-Log Power Law Model")
print("="*70)
print(f"\nObserved data summary:")
print(f"  x range: [{x_obs.min():.2f}, {x_obs.max():.2f}]")
print(f"  Y range: [{y_obs.min():.2f}, {y_obs.max():.2f}]")
print(f"  N observations: {len(x_obs)}")

# ============================================================================
# STEP 1: Sample from priors
# ============================================================================
print(f"\n{'='*70}")
print(f"STEP 1: Sampling from priors (N={N_PRIOR_DRAWS})")
print(f"{'='*70}")

# Sample prior parameters
alpha_prior = np.random.normal(0.6, 0.3, N_PRIOR_DRAWS)
beta_prior = np.random.normal(0.12, 0.1, N_PRIOR_DRAWS)

# Half-Cauchy: sample from Cauchy and take absolute value
# Using scipy's implementation: scale parameter = 0.1
sigma_prior = np.abs(stats.cauchy.rvs(loc=0, scale=0.1, size=N_PRIOR_DRAWS))

print("\nPrior parameter statistics:")
print(f"\nα (intercept on log scale):")
print(f"  Mean: {alpha_prior.mean():.3f}, SD: {alpha_prior.std():.3f}")
print(f"  Range: [{alpha_prior.min():.3f}, {alpha_prior.max():.3f}]")
print(f"  5th-95th percentile: [{np.percentile(alpha_prior, 5):.3f}, {np.percentile(alpha_prior, 95):.3f}]")

print(f"\nβ (power law exponent):")
print(f"  Mean: {beta_prior.mean():.3f}, SD: {beta_prior.std():.3f}")
print(f"  Range: [{beta_prior.min():.3f}, {beta_prior.max():.3f}]")
print(f"  5th-95th percentile: [{np.percentile(beta_prior, 5):.3f}, {np.percentile(beta_prior, 95):.3f}]")
print(f"  % negative: {100*(beta_prior < 0).mean():.1f}%")

print(f"\nσ (residual SD on log scale):")
print(f"  Mean: {sigma_prior.mean():.3f}, SD: {sigma_prior.std():.3f}")
print(f"  Range: [{sigma_prior.min():.3f}, {sigma_prior.max():.3f}]")
print(f"  95th percentile: {np.percentile(sigma_prior, 95):.3f}")

# Check for computational issues in priors
print("\nComputational checks on priors:")
print(f"  Any NaN in α: {np.any(np.isnan(alpha_prior))}")
print(f"  Any NaN in β: {np.any(np.isnan(beta_prior))}")
print(f"  Any NaN in σ: {np.any(np.isnan(sigma_prior))}")
print(f"  Any σ > 1.0: {np.any(sigma_prior > 1.0)} (count: {(sigma_prior > 1.0).sum()})")

# ============================================================================
# STEP 2: Generate prior predictive datasets
# ============================================================================
print(f"\n{'='*70}")
print(f"STEP 2: Generating prior predictive datasets")
print(f"{'='*70}")

# Create x values for prediction (log-spaced for better coverage)
x_pred = np.linspace(X_MIN, X_MAX, N_PRED_POINTS)
log_x_pred = np.log(x_pred)

# Storage for prior predictions
y_pred_samples = np.zeros((N_PRIOR_DRAWS, N_PRED_POINTS))
log_y_pred_samples = np.zeros((N_PRIOR_DRAWS, N_PRED_POINTS))

print(f"\nGenerating {N_PRIOR_DRAWS} prior predictive trajectories...")
print(f"Prediction range: x ∈ [{X_MIN}, {X_MAX}]")

for i in range(N_PRIOR_DRAWS):
    # Mean on log scale
    mu_log = alpha_prior[i] + beta_prior[i] * log_x_pred

    # Generate predictions on log scale (with noise)
    log_y_pred = np.random.normal(mu_log, sigma_prior[i], N_PRED_POINTS)

    # Transform back to original scale
    y_pred = np.exp(log_y_pred)

    # Store
    log_y_pred_samples[i, :] = log_y_pred
    y_pred_samples[i, :] = y_pred

print("Prior predictive generation complete!")

# ============================================================================
# STEP 3: Assess plausibility
# ============================================================================
print(f"\n{'='*70}")
print(f"STEP 3: Assessing plausibility of prior predictions")
print(f"{'='*70}")

# Define plausibility bounds
PLAUSIBLE_MIN = 0.5  # Lower bound (with buffer)
PLAUSIBLE_MAX = 5.0  # Upper bound (with buffer)

print(f"\nPlausibility criteria:")
print(f"  Y should mostly fall in [{PLAUSIBLE_MIN}, {PLAUSIBLE_MAX}]")
print(f"  (Observed range: [{y_obs.min():.2f}, {y_obs.max():.2f}])")

# Check each trajectory
trajectory_checks = []
for i in range(N_PRIOR_DRAWS):
    y_traj = y_pred_samples[i, :]

    # Check if all values in plausible range
    all_plausible = np.all((y_traj >= PLAUSIBLE_MIN) & (y_traj <= PLAUSIBLE_MAX))

    # Check for extreme values
    has_extreme = np.any(y_traj > 100) or np.any(y_traj < 0.01)

    # Check for NaNs or Infs
    has_numerical_issues = np.any(np.isnan(y_traj)) or np.any(np.isinf(y_traj))

    # Check monotonicity (increasing)
    is_monotonic_increasing = np.all(np.diff(y_traj) >= 0)

    trajectory_checks.append({
        'all_plausible': all_plausible,
        'has_extreme': has_extreme,
        'has_numerical_issues': has_numerical_issues,
        'is_monotonic_increasing': is_monotonic_increasing,
        'min_y': y_traj.min(),
        'max_y': y_traj.max(),
        'range_y': y_traj.max() - y_traj.min()
    })

trajectory_df = pd.DataFrame(trajectory_checks)

print("\nTrajectory-level assessment:")
print(f"  % trajectories fully in plausible range: {100*trajectory_df['all_plausible'].mean():.1f}%")
print(f"  % trajectories with extreme values: {100*trajectory_df['has_extreme'].mean():.1f}%")
print(f"  % trajectories with numerical issues: {100*trajectory_df['has_numerical_issues'].mean():.1f}%")
print(f"  % trajectories monotonically increasing: {100*trajectory_df['is_monotonic_increasing'].mean():.1f}%")

# Point-wise assessment at key x values
key_x_values = [1.0, 5.0, 10.0, 15.0, 20.0, 30.0]
print(f"\nPoint-wise assessment at key x values:")
for x_key in key_x_values:
    idx = np.argmin(np.abs(x_pred - x_key))
    y_at_x = y_pred_samples[:, idx]

    pct_plausible = 100 * np.mean((y_at_x >= PLAUSIBLE_MIN) & (y_at_x <= PLAUSIBLE_MAX))

    print(f"\n  x = {x_key:.1f}:")
    print(f"    Median Y: {np.median(y_at_x):.2f}")
    print(f"    95% interval: [{np.percentile(y_at_x, 2.5):.2f}, {np.percentile(y_at_x, 97.5):.2f}]")
    print(f"    % plausible: {pct_plausible:.1f}%")

# Overall assessment
print(f"\n{'='*70}")
print(f"OVERALL PLAUSIBILITY ASSESSMENT")
print(f"{'='*70}")

# Calculate overall plausibility
all_y_values = y_pred_samples.flatten()
pct_all_plausible = 100 * np.mean((all_y_values >= PLAUSIBLE_MIN) & (all_y_values <= PLAUSIBLE_MAX))

print(f"\nOverall statistics (all {len(all_y_values)} predicted values):")
print(f"  % in plausible range [{PLAUSIBLE_MIN}, {PLAUSIBLE_MAX}]: {pct_all_plausible:.1f}%")
print(f"  Median: {np.median(all_y_values):.2f}")
print(f"  95% interval: [{np.percentile(all_y_values, 2.5):.2f}, {np.percentile(all_y_values, 97.5):.2f}]")
print(f"  Min: {all_y_values.min():.3f}")
print(f"  Max: {all_y_values.max():.3f}")

# Check for domain violations
print(f"\nDomain violation checks:")
print(f"  % values > 100: {100*np.mean(all_y_values > 100):.2f}%")
print(f"  % values < 0.01: {100*np.mean(all_y_values < 0.01):.2f}%")
print(f"  Any NaN: {np.any(np.isnan(all_y_values))}")
print(f"  Any Inf: {np.any(np.isinf(all_y_values))}")

# ============================================================================
# STEP 4: Compute additional diagnostics
# ============================================================================
print(f"\n{'='*70}")
print(f"STEP 4: Additional diagnostics")
print(f"{'='*70}")

# Power law interpretation: Y = exp(α) * x^β
scaling_constant = np.exp(alpha_prior)
print(f"\nScaling constant (exp(α)):")
print(f"  Median: {np.median(scaling_constant):.2f}")
print(f"  95% interval: [{np.percentile(scaling_constant, 2.5):.2f}, {np.percentile(scaling_constant, 97.5):.2f}]")

# At x=1, Y should be close to exp(α)
print(f"\nAt x=1, Y ≈ exp(α) * 1^β = exp(α):")
idx_x1 = np.argmin(np.abs(x_pred - 1.0))
y_at_x1 = y_pred_samples[:, idx_x1]
print(f"  Median Y at x=1: {np.median(y_at_x1):.2f}")
print(f"  Median exp(α): {np.median(scaling_constant):.2f}")

# Growth factor from x=1 to x=30
idx_x30 = np.argmin(np.abs(x_pred - 30.0))
y_at_x30 = y_pred_samples[:, idx_x30]
growth_factor = y_at_x30 / y_at_x1

print(f"\nGrowth from x=1 to x=30:")
print(f"  Median growth factor: {np.median(growth_factor):.2f}x")
print(f"  95% interval: [{np.percentile(growth_factor, 2.5):.2f}x, {np.percentile(growth_factor, 97.5):.2f}x]")

# Expected growth based on power law: (30/1)^β = 30^β
expected_growth = 30**beta_prior
print(f"\nExpected growth (30^β):")
print(f"  Median: {np.median(expected_growth):.2f}x")
print(f"  95% interval: [{np.percentile(expected_growth, 2.5):.2f}x, {np.percentile(expected_growth, 97.5):.2f}x]")

# ============================================================================
# Save results for visualization
# ============================================================================
print(f"\n{'='*70}")
print(f"Saving results for visualization...")
print(f"{'='*70}")

# Save prior samples
prior_samples = pd.DataFrame({
    'alpha': alpha_prior,
    'beta': beta_prior,
    'sigma': sigma_prior
})
prior_samples.to_csv('/workspace/experiments/experiment_3/prior_predictive_check/code/prior_samples.csv', index=False)

# Save prior predictions at key points for plotting
np.savez('/workspace/experiments/experiment_3/prior_predictive_check/code/prior_predictions.npz',
         x_pred=x_pred,
         y_pred_samples=y_pred_samples,
         log_y_pred_samples=log_y_pred_samples,
         x_obs=x_obs,
         y_obs=y_obs)

print("\nResults saved:")
print("  - prior_samples.csv")
print("  - prior_predictions.npz")

# ============================================================================
# Final recommendation
# ============================================================================
print(f"\n{'='*70}")
print(f"RECOMMENDATION")
print(f"{'='*70}")

# Decision criteria
trajectory_pass_rate = trajectory_df['all_plausible'].mean()
monotonic_rate = trajectory_df['is_monotonic_increasing'].mean()
numerical_issue_rate = trajectory_df['has_numerical_issues'].mean()

print(f"\nKey metrics:")
print(f"  Trajectory pass rate: {100*trajectory_pass_rate:.1f}%")
print(f"  Monotonic increasing: {100*monotonic_rate:.1f}%")
print(f"  Numerical issues: {100*numerical_issue_rate:.1f}%")

if trajectory_pass_rate >= 0.80 and numerical_issue_rate < 0.01:
    recommendation = "PASS"
    message = "Priors are well-calibrated. Proceed to SBC."
elif trajectory_pass_rate >= 0.60:
    recommendation = "CONDITIONAL PASS"
    message = "Priors are reasonable but could be tightened. Consider proceeding with caution."
else:
    recommendation = "REVISE"
    message = "Priors generate too many implausible values. Revision recommended."

print(f"\n>>> RECOMMENDATION: {recommendation}")
print(f">>> {message}")
print(f"\n{'='*70}\n")
