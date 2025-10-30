"""
Simulation-Based Calibration for Experiment 1: Negative Binomial GLM
Tests if model can recover known parameters from synthetic data
Using PyMC (since CmdStan requires make which is not available)
"""

import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import time
import warnings

warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# Define paths
BASE_DIR = Path("/workspace/experiments/experiment_1/simulation_based_validation")
CODE_DIR = BASE_DIR / "code"
PLOT_DIR = BASE_DIR / "plots"
DATA_PATH = Path("/workspace/data/data.csv")

# Load real data to get year structure
data = pd.read_csv(DATA_PATH)
year = data['year'].values
N = len(year)

print("="*70)
print("SIMULATION-BASED VALIDATION: EXPERIMENT 1")
print("="*70)
print(f"Model: Negative Binomial GLM with Quadratic Trend")
print(f"PPL: PyMC (CmdStan unavailable - requires make)")
print(f"N observations: {N}")
print(f"Year range: [{year.min():.2f}, {year.max():.2f}]")
print("="*70)

# ============================================================================
# STEP 1: Define True Parameters (Realistic Values)
# ============================================================================
print("\n[1] TRUE PARAMETERS")
print("-" * 70)

TRUE_PARAMS = {
    'beta_0': 4.5,   # Intercept (log scale) - exp(4.5) ≈ 90
    'beta_1': 0.85,  # Linear growth
    'beta_2': 0.1,   # Quadratic acceleration
    'phi': 15.0      # Dispersion (moderate overdispersion)
}

for param, value in TRUE_PARAMS.items():
    print(f"  {param:8s} = {value:8.3f}")

# ============================================================================
# STEP 2: Generate Synthetic Data
# ============================================================================
print("\n[2] GENERATING SYNTHETIC DATA")
print("-" * 70)

# Compute true mean on log scale
log_mu_true = TRUE_PARAMS['beta_0'] + TRUE_PARAMS['beta_1'] * year + TRUE_PARAMS['beta_2'] * year**2
mu_true = np.exp(log_mu_true)

# Generate counts from Negative Binomial
# NB2 parameterization: mean = mu, variance = mu + mu^2 / phi
# Convert to (n, p) parameterization for numpy
# n = phi, p = phi / (phi + mu)
phi_true = TRUE_PARAMS['phi']
n_param = phi_true
p_param = phi_true / (phi_true + mu_true)
C_synthetic = np.random.negative_binomial(n_param, p_param)

print(f"  Mean count (synthetic): {C_synthetic.mean():.1f}")
print(f"  Std count (synthetic): {C_synthetic.std():.1f}")
print(f"  Variance/Mean ratio: {C_synthetic.var() / C_synthetic.mean():.2f}")
print(f"  Range: [{C_synthetic.min()}, {C_synthetic.max()}]")

# Save synthetic data
synthetic_data_path = CODE_DIR / "synthetic_data.csv"
synthetic_df = pd.DataFrame({
    'year': year,
    'C': C_synthetic,
    'mu_true': mu_true
})
synthetic_df.to_csv(synthetic_data_path, index=False)
print(f"\n  Saved: {synthetic_data_path}")

# ============================================================================
# STEP 3: Build PyMC Model
# ============================================================================
print("\n[3] BUILDING PYMC MODEL")
print("-" * 70)

with pm.Model() as model:
    # Priors
    beta_0 = pm.Normal('beta_0', mu=4.5, sigma=1.0)
    beta_1 = pm.Normal('beta_1', mu=0.9, sigma=0.5)
    beta_2 = pm.Normal('beta_2', mu=0, sigma=0.3)
    phi = pm.Gamma('phi', alpha=2, beta=0.1)

    # Linear predictor on log scale
    log_mu = beta_0 + beta_1 * year + beta_2 * year**2
    mu = pm.math.exp(log_mu)

    # Likelihood: Negative Binomial
    # PyMC uses (mu, alpha) parameterization where alpha = phi
    obs = pm.NegativeBinomial('obs', mu=mu, alpha=phi, observed=C_synthetic)

print(f"  ✓ Model built successfully")
print(f"  Parameters: beta_0, beta_1, beta_2, phi")
print(f"  Likelihood: NegativeBinomial(mu=exp(beta_0 + beta_1*year + beta_2*year^2), alpha=phi)")

# ============================================================================
# STEP 4: Fit Model to Synthetic Data
# ============================================================================
print("\n[4] FITTING MODEL TO SYNTHETIC DATA")
print("-" * 70)

print(f"  Chains: 4")
print(f"  Draws: 1000 (500 tune)")
print(f"  Sampling...")

start_sample = time.time()
with model:
    trace = pm.sample(
        draws=1000,
        tune=500,
        chains=4,
        cores=4,
        random_seed=42,
        return_inferencedata=True,
        progressbar=True
    )
sample_time = time.time() - start_sample
print(f"  ✓ Sampling completed ({sample_time:.1f}s)")

# ============================================================================
# STEP 5: Convergence Diagnostics
# ============================================================================
print("\n[5] CONVERGENCE DIAGNOSTICS")
print("-" * 70)

params_of_interest = ['beta_0', 'beta_1', 'beta_2', 'phi']
summary = az.summary(trace, var_names=params_of_interest)

print(f"\n  {'Parameter':<10} {'R-hat':<10} {'ESS_bulk':<12} {'ESS_tail':<12}")
print("  " + "-" * 46)

convergence_pass = True
for param in params_of_interest:
    rhat = summary.loc[param, 'r_hat']
    ess_bulk = summary.loc[param, 'ess_bulk']
    ess_tail = summary.loc[param, 'ess_tail']

    rhat_flag = "✓" if rhat < 1.01 else "✗"
    ess_flag = "✓" if ess_bulk > 400 and ess_tail > 400 else "✗"

    print(f"  {param:<10} {rhat:<9.4f} {rhat_flag}  {ess_bulk:<11.0f} {ess_flag}  {ess_tail:<11.0f} {ess_flag}")

    if rhat >= 1.01 or ess_bulk <= 400 or ess_tail <= 400:
        convergence_pass = False

# Check for divergences
divergences = trace.sample_stats.diverging.sum().values
if divergences == 0:
    print(f"\n  ✓ No divergent transitions")
else:
    print(f"\n  ✗ {divergences} divergent transitions detected")
    convergence_pass = False

print(f"\n  Overall convergence: {'PASS ✓' if convergence_pass else 'FAIL ✗'}")

# ============================================================================
# STEP 6: Parameter Recovery Assessment
# ============================================================================
print("\n[6] PARAMETER RECOVERY ASSESSMENT")
print("-" * 70)

print(f"\n  {'Parameter':<10} {'True':<10} {'Post.Mean':<12} {'Post.SD':<10} {'Rel.Error':<12} {'Status':<8}")
print("  " + "-" * 64)

recovery_results = {}
recovery_pass = True

for param in params_of_interest:
    true_val = TRUE_PARAMS[param]
    post_samples = trace.posterior[param].values.flatten()
    post_mean = post_samples.mean()
    post_sd = post_samples.std()

    # Relative error
    rel_error = abs(post_mean - true_val) / abs(true_val) * 100

    # Check if true value is in 90% credible interval
    ci_lower = np.percentile(post_samples, 5)
    ci_upper = np.percentile(post_samples, 95)
    in_ci = ci_lower <= true_val <= ci_upper

    # Pass criteria: <20% error is good, <30% is acceptable
    if rel_error < 20:
        status = "PASS ✓"
    elif rel_error < 30:
        status = "MARGINAL"
        recovery_pass = False
    else:
        status = "FAIL ✗"
        recovery_pass = False

    print(f"  {param:<10} {true_val:<10.3f} {post_mean:<12.3f} {post_sd:<10.3f} {rel_error:<11.1f}% {status:<8}")

    recovery_results[param] = {
        'true': true_val,
        'post_mean': post_mean,
        'post_sd': post_sd,
        'rel_error': rel_error,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'in_ci': in_ci,
        'post_samples': post_samples
    }

print(f"\n  Overall recovery: {'PASS ✓' if recovery_pass else 'FAIL ✗'}")

# ============================================================================
# STEP 7: Credible Interval Coverage
# ============================================================================
print("\n[7] CREDIBLE INTERVAL COVERAGE (90%)")
print("-" * 70)

coverage_pass = True
for param in params_of_interest:
    true_val = TRUE_PARAMS[param]
    ci_lower = recovery_results[param]['ci_lower']
    ci_upper = recovery_results[param]['ci_upper']
    in_ci = recovery_results[param]['in_ci']

    status = "✓" if in_ci else "✗"
    print(f"  {param:<10} [{ci_lower:8.3f}, {ci_upper:8.3f}] contains {true_val:8.3f}: {status}")

    if not in_ci:
        coverage_pass = False

print(f"\n  Overall coverage: {'PASS ✓' if coverage_pass else 'FAIL ✗'}")

# ============================================================================
# STEP 8: Parameter Correlation Analysis
# ============================================================================
print("\n[8] PARAMETER IDENTIFIABILITY")
print("-" * 70)

# Compute posterior correlations
param_matrix = np.column_stack([recovery_results[p]['post_samples'] for p in params_of_interest])
correlations = np.corrcoef(param_matrix.T)

print(f"\n  Posterior correlations:")
print(f"  {'':>10}", end="")
for param in params_of_interest:
    print(f" {param:>9}", end="")
print()

identifiability_pass = True
for i, param1 in enumerate(params_of_interest):
    print(f"  {param1:>10}", end="")
    for j, param2 in enumerate(params_of_interest):
        corr = correlations[i, j]
        print(f" {corr:9.3f}", end="")

        # Check for problematic correlations (>0.95, excluding diagonal)
        if i != j and abs(corr) > 0.95:
            identifiability_pass = False
    print()

# Check specific correlations
beta1_beta2_corr = correlations[1, 2]  # beta_1 vs beta_2
print(f"\n  Critical correlation (beta_1 vs beta_2): {beta1_beta2_corr:.3f}")
if abs(beta1_beta2_corr) > 0.95:
    print(f"  ✗ Severe multicollinearity detected")
    identifiability_pass = False
elif abs(beta1_beta2_corr) > 0.8:
    print(f"  ⚠ Moderate correlation (expected for quadratic model)")
else:
    print(f"  ✓ Acceptable correlation")

print(f"\n  Overall identifiability: {'PASS ✓' if identifiability_pass else 'FAIL ✗'}")

# ============================================================================
# STEP 9: Create Visualizations
# ============================================================================
print("\n[9] GENERATING DIAGNOSTIC PLOTS")
print("-" * 70)

# Plot 1: Parameter Recovery
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('Parameter Recovery Assessment', fontsize=14, fontweight='bold')

for idx, param in enumerate(params_of_interest):
    ax = axes[idx // 2, idx % 2]

    post_samples = recovery_results[param]['post_samples']
    true_val = TRUE_PARAMS[param]

    # Histogram of posterior
    ax.hist(post_samples, bins=40, alpha=0.6, color='steelblue', edgecolor='black', density=True)

    # True value line
    ax.axvline(true_val, color='red', linestyle='--', linewidth=2, label='True value')

    # Posterior mean
    post_mean = post_samples.mean()
    ax.axvline(post_mean, color='green', linestyle='-', linewidth=2, label='Posterior mean')

    # 90% CI
    ci_lower = recovery_results[param]['ci_lower']
    ci_upper = recovery_results[param]['ci_upper']
    ax.axvspan(ci_lower, ci_upper, alpha=0.2, color='green', label='90% CI')

    # Labels and title
    ax.set_xlabel(f'{param}', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)

    rel_error = recovery_results[param]['rel_error']
    in_ci = recovery_results[param]['in_ci']
    ci_status = "✓" if in_ci else "✗"

    ax.set_title(f'{param}: Rel. Error = {rel_error:.1f}% {ci_status}', fontsize=11)
    ax.legend(fontsize=9, loc='best')
    ax.grid(alpha=0.3)

plt.tight_layout()
plot1_path = PLOT_DIR / "parameter_recovery.png"
plt.savefig(plot1_path, dpi=300, bbox_inches='tight')
print(f"  ✓ Saved: {plot1_path}")
plt.close()

# Plot 2: True vs Recovered (scatter)
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Parameter Recovery: True vs Posterior', fontsize=14, fontweight='bold')

# Panel A: Mean recovery
ax = axes[0]
true_vals = [TRUE_PARAMS[p] for p in params_of_interest]
post_means = [recovery_results[p]['post_mean'] for p in params_of_interest]
post_sds = [recovery_results[p]['post_sd'] for p in params_of_interest]

ax.errorbar(true_vals, post_means, yerr=[1.96*sd for sd in post_sds],
            fmt='o', markersize=8, capsize=5, color='steelblue', label='Posterior mean ± 1.96 SD')

# Identity line
val_range = [min(true_vals + post_means), max(true_vals + post_means)]
ax.plot(val_range, val_range, 'k--', linewidth=1, label='Perfect recovery')

# Annotate points
for i, param in enumerate(params_of_interest):
    ax.annotate(param, (true_vals[i], post_means[i]),
                xytext=(5, 5), textcoords='offset points', fontsize=9)

ax.set_xlabel('True Parameter Value', fontsize=11)
ax.set_ylabel('Posterior Mean', fontsize=11)
ax.set_title('Recovery Accuracy', fontsize=12)
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

# Panel B: Relative error
ax = axes[1]
rel_errors = [recovery_results[p]['rel_error'] for p in params_of_interest]
colors = ['green' if e < 20 else 'orange' if e < 30 else 'red' for e in rel_errors]

bars = ax.bar(params_of_interest, rel_errors, color=colors, alpha=0.7, edgecolor='black')
ax.axhline(20, color='green', linestyle='--', linewidth=1, label='20% threshold (PASS)')
ax.axhline(30, color='orange', linestyle='--', linewidth=1, label='30% threshold (FAIL)')

ax.set_ylabel('Relative Error (%)', fontsize=11)
ax.set_title('Recovery Error by Parameter', fontsize=12)
ax.legend(fontsize=9)
ax.grid(axis='y', alpha=0.3)

# Add values on bars
for i, (bar, val) in enumerate(zip(bars, rel_errors)):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
            f'{val:.1f}%', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plot2_path = PLOT_DIR / "recovery_accuracy.png"
plt.savefig(plot2_path, dpi=300, bbox_inches='tight')
print(f"  ✓ Saved: {plot2_path}")
plt.close()

# Plot 3: Convergence Diagnostics (Trace plots)
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('MCMC Convergence Diagnostics (Trace Plots)', fontsize=14, fontweight='bold')

for idx, param in enumerate(params_of_interest):
    ax = axes[idx // 2, idx % 2]

    rhat = summary.loc[param, 'r_hat']
    ess_bulk = summary.loc[param, 'ess_bulk']

    # Plot trace for each chain
    for chain in range(4):
        chain_samples = trace.posterior[param].sel(chain=chain).values
        ax.plot(chain_samples, alpha=0.6, linewidth=0.5, label=f'Chain {chain+1}')

    ax.set_xlabel('Iteration (post-warmup)', fontsize=10)
    ax.set_ylabel(param, fontsize=10)
    ax.set_title(f'{param}: R̂={rhat:.4f}, ESS_bulk={ess_bulk:.0f}', fontsize=11)
    ax.legend(fontsize=8, loc='best', ncol=2)
    ax.grid(alpha=0.3)

plt.tight_layout()
plot3_path = PLOT_DIR / "convergence_diagnostics.png"
plt.savefig(plot3_path, dpi=300, bbox_inches='tight')
print(f"  ✓ Saved: {plot3_path}")
plt.close()

# Plot 4: Parameter Correlations
fig, ax = plt.subplots(1, 1, figsize=(10, 8))
im = ax.imshow(correlations, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')

# Add colorbar
cbar = plt.colorbar(im, ax=ax)
cbar.set_label('Correlation', rotation=270, labelpad=20, fontsize=11)

# Set ticks and labels
ax.set_xticks(range(len(params_of_interest)))
ax.set_yticks(range(len(params_of_interest)))
ax.set_xticklabels(params_of_interest, fontsize=11)
ax.set_yticklabels(params_of_interest, fontsize=11)

# Add correlation values
for i in range(len(params_of_interest)):
    for j in range(len(params_of_interest)):
        text = ax.text(j, i, f'{correlations[i, j]:.2f}',
                      ha="center", va="center", color="black", fontsize=10)

ax.set_title('Posterior Parameter Correlations\n(Identifiability Check)',
             fontsize=13, fontweight='bold', pad=15)

plt.tight_layout()
plot4_path = PLOT_DIR / "parameter_correlations.png"
plt.savefig(plot4_path, dpi=300, bbox_inches='tight')
print(f"  ✓ Saved: {plot4_path}")
plt.close()

# Plot 5: Data Fit Visualization
fig, ax = plt.subplots(1, 1, figsize=(12, 6))

# Plot synthetic data
ax.scatter(year, C_synthetic, color='black', s=50, zorder=3, label='Synthetic data', alpha=0.7)

# Plot true mean
ax.plot(year, mu_true, color='red', linewidth=2, linestyle='--', label='True mean', zorder=2)

# Plot posterior mean predictions
with model:
    post_pred = pm.sample_posterior_predictive(trace, random_seed=42)

# Extract mu (compute from parameters)
mu_samples_list = []
for i in range(len(trace.posterior.chain) * len(trace.posterior.draw)):
    chain_idx = i // len(trace.posterior.draw)
    draw_idx = i % len(trace.posterior.draw)

    b0 = trace.posterior['beta_0'].sel(chain=chain_idx, draw=draw_idx).values
    b1 = trace.posterior['beta_1'].sel(chain=chain_idx, draw=draw_idx).values
    b2 = trace.posterior['beta_2'].sel(chain=chain_idx, draw=draw_idx).values

    mu_sample = np.exp(b0 + b1 * year + b2 * year**2)
    mu_samples_list.append(mu_sample)

mu_samples = np.array(mu_samples_list)
mu_post_mean = mu_samples.mean(axis=0)
mu_post_lower = np.percentile(mu_samples, 5, axis=0)
mu_post_upper = np.percentile(mu_samples, 95, axis=0)

ax.plot(year, mu_post_mean, color='steelblue', linewidth=2, label='Recovered mean', zorder=2)
ax.fill_between(year, mu_post_lower, mu_post_upper, color='steelblue', alpha=0.2, label='90% CI')

ax.set_xlabel('Standardized Year', fontsize=12)
ax.set_ylabel('Count', fontsize=12)
ax.set_title('Synthetic Data Recovery: True vs Fitted Mean', fontsize=13, fontweight='bold')
ax.legend(fontsize=10, loc='upper left')
ax.grid(alpha=0.3)

plt.tight_layout()
plot5_path = PLOT_DIR / "data_fit.png"
plt.savefig(plot5_path, dpi=300, bbox_inches='tight')
print(f"  ✓ Saved: {plot5_path}")
plt.close()

# ============================================================================
# STEP 10: Final Decision
# ============================================================================
print("\n" + "="*70)
print("FINAL VALIDATION DECISION")
print("="*70)

all_checks = {
    'Convergence (R-hat < 1.01, ESS > 400)': convergence_pass,
    'Parameter Recovery (<20% rel. error)': recovery_pass,
    'Credible Interval Coverage (90%)': coverage_pass,
    'Parameter Identifiability (|corr| < 0.95)': identifiability_pass
}

print("\nCriteria checklist:")
for criterion, passed in all_checks.items():
    status = "✓ PASS" if passed else "✗ FAIL"
    print(f"  [{status}] {criterion}")

final_pass = all(all_checks.values())

print("\n" + "-"*70)
if final_pass:
    print("  ✓✓✓ VALIDATION PASSED ✓✓✓")
    print("\n  Model successfully recovers known parameters from synthetic data.")
    print("  Safe to proceed with fitting to real data.")
else:
    print("  ✗✗✗ VALIDATION FAILED ✗✗✗")
    print("\n  Model cannot reliably recover known parameters.")
    print("  DO NOT proceed to real data fitting without addressing issues.")

print("="*70)

# Save results (exclude post_samples from JSON)
results = {
    'validation_status': 'PASS' if final_pass else 'FAIL',
    'ppl': 'PyMC',
    'true_parameters': TRUE_PARAMS,
    'recovery_results': {k: {sk: float(sv) if not isinstance(sv, (bool, np.ndarray)) else (sv if isinstance(sv, bool) else None)
                             for sk, sv in v.items() if sk != 'post_samples'}
                        for k, v in recovery_results.items()},
    'convergence_pass': convergence_pass,
    'recovery_pass': recovery_pass,
    'coverage_pass': coverage_pass,
    'identifiability_pass': identifiability_pass,
    'correlations': correlations.tolist(),
    'computation_time': {
        'sample_seconds': sample_time
    },
    'divergences': int(divergences)
}

results_path = BASE_DIR / "validation_results.json"
with open(results_path, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\nResults saved to: {results_path}")
print(f"Plots saved to: {PLOT_DIR}/")
print("\nSimulation-based validation complete!")
