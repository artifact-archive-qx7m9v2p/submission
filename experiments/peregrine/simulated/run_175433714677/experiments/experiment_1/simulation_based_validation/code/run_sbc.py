"""
Simulation-Based Calibration for Negative Binomial Regression Model

This script:
1. Draws parameters from priors
2. Simulates data from those parameters
3. Fits the model to simulated data
4. Checks if posteriors recover true parameters
"""

import numpy as np
import pandas as pd
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from cmdstanpy import CmdStanModel
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# Paths
BASE_DIR = Path("/workspace/experiments/experiment_1/simulation_based_validation")
CODE_DIR = BASE_DIR / "code"
PLOTS_DIR = BASE_DIR / "plots"
STAN_FILE = CODE_DIR / "negbinom_model.stan"

# Load real data to get year covariate structure
with open("/workspace/data/data.csv", "r") as f:
    real_data = json.load(f)

n = real_data["n"]
year = np.array(real_data["year"])

# SBC Configuration
N_SIMS = 50  # Number of simulations
WARMUP = 500
SAMPLES = 500
CHAINS = 4
THIN = 1

# Prior specifications
PRIOR_BETA_0_MEAN = 4.3
PRIOR_BETA_0_SD = 1.0
PRIOR_BETA_1_MEAN = 0.85
PRIOR_BETA_1_SD = 0.5
PRIOR_PHI_RATE = 0.667  # Exponential rate parameter

print("="*80)
print("SIMULATION-BASED CALIBRATION")
print("="*80)
print(f"\nConfiguration:")
print(f"  Simulations: {N_SIMS}")
print(f"  Chains: {CHAINS}, Warmup: {WARMUP}, Samples: {SAMPLES}")
print(f"  Total posterior draws per simulation: {CHAINS * SAMPLES}")
print(f"\nPrior specifications:")
print(f"  β₀ ~ Normal({PRIOR_BETA_0_MEAN}, {PRIOR_BETA_0_SD})")
print(f"  β₁ ~ Normal({PRIOR_BETA_1_MEAN}, {PRIOR_BETA_1_SD})")
print(f"  φ ~ Exponential({PRIOR_PHI_RATE}) [E[φ] = {1/PRIOR_PHI_RATE:.2f}]")

# Compile Stan model
print(f"\nCompiling Stan model...")
model = CmdStanModel(stan_file=str(STAN_FILE))
print("Model compiled successfully!")

def draw_from_prior():
    """Draw parameters from prior distributions"""
    beta_0 = np.random.normal(PRIOR_BETA_0_MEAN, PRIOR_BETA_0_SD)
    beta_1 = np.random.normal(PRIOR_BETA_1_MEAN, PRIOR_BETA_1_SD)
    phi = np.random.exponential(1 / PRIOR_PHI_RATE)
    return beta_0, beta_1, phi

def simulate_data(beta_0, beta_1, phi, year):
    """Generate synthetic count data from Negative Binomial model"""
    log_mu = beta_0 + beta_1 * year
    mu = np.exp(log_mu)

    # Generate from negative binomial
    # numpy uses different parameterization: n (phi), p
    # We need to convert: p = phi / (phi + mu)
    p = phi / (phi + mu)
    C = np.random.negative_binomial(n=phi, p=p, size=len(year))

    return C.astype(int)

def fit_model(C, year):
    """Fit Stan model to data"""
    data = {
        'n': len(C),
        'C': C.tolist(),
        'year': year.tolist()
    }

    try:
        fit = model.sample(
            data=data,
            chains=CHAINS,
            iter_warmup=WARMUP,
            iter_sampling=SAMPLES,
            thin=THIN,
            show_progress=False,
            show_console=False,
            adapt_delta=0.95
        )
        return fit
    except Exception as e:
        print(f"    ERROR during fitting: {e}")
        return None

def extract_posterior_samples(fit):
    """Extract posterior samples for parameters"""
    try:
        samples = fit.stan_variables()
        return {
            'beta_0': samples['beta_0'],
            'beta_1': samples['beta_1'],
            'phi': samples['phi']
        }
    except Exception as e:
        print(f"    ERROR extracting samples: {e}")
        return None

def compute_rank(true_value, posterior_samples):
    """
    Compute rank statistic for SBC
    Rank = number of posterior samples less than true value
    """
    return np.sum(posterior_samples < true_value)

# Run SBC
print(f"\n{'='*80}")
print("RUNNING SIMULATIONS")
print(f"{'='*80}\n")

results = []
ranks = {'beta_0': [], 'beta_1': [], 'phi': []}
failures = 0

for sim in range(N_SIMS):
    print(f"Simulation {sim+1}/{N_SIMS}...", end=" ")

    # 1. Draw true parameters from prior
    beta_0_true, beta_1_true, phi_true = draw_from_prior()

    # 2. Simulate data
    C_sim = simulate_data(beta_0_true, beta_1_true, phi_true, year)

    # 3. Fit model
    fit = fit_model(C_sim, year)

    if fit is None:
        failures += 1
        print("FAILED (fitting error)")
        continue

    # Check convergence
    try:
        summary = fit.summary()
        max_rhat = summary['R_hat'].max()
        min_ess_bulk = summary['N_Eff'].min()

        if max_rhat > 1.05 or min_ess_bulk < 100:
            failures += 1
            print(f"FAILED (convergence: R_hat={max_rhat:.3f}, ESS={min_ess_bulk:.0f})")
            continue
    except:
        failures += 1
        print("FAILED (summary error)")
        continue

    # 4. Extract posteriors
    posteriors = extract_posterior_samples(fit)

    if posteriors is None:
        failures += 1
        print("FAILED (extraction error)")
        continue

    # 5. Compute statistics
    beta_0_mean = np.mean(posteriors['beta_0'])
    beta_0_q05 = np.percentile(posteriors['beta_0'], 5)
    beta_0_q95 = np.percentile(posteriors['beta_0'], 95)

    beta_1_mean = np.mean(posteriors['beta_1'])
    beta_1_q05 = np.percentile(posteriors['beta_1'], 5)
    beta_1_q95 = np.percentile(posteriors['beta_1'], 95)

    phi_mean = np.mean(posteriors['phi'])
    phi_q05 = np.percentile(posteriors['phi'], 5)
    phi_q95 = np.percentile(posteriors['phi'], 95)

    # 6. Compute ranks for SBC
    rank_beta_0 = compute_rank(beta_0_true, posteriors['beta_0'])
    rank_beta_1 = compute_rank(beta_1_true, posteriors['beta_1'])
    rank_phi = compute_rank(phi_true, posteriors['phi'])

    ranks['beta_0'].append(rank_beta_0)
    ranks['beta_1'].append(rank_beta_1)
    ranks['phi'].append(rank_phi)

    # Store results
    results.append({
        'sim': sim + 1,
        'beta_0_true': beta_0_true,
        'beta_0_mean': beta_0_mean,
        'beta_0_q05': beta_0_q05,
        'beta_0_q95': beta_0_q95,
        'beta_0_covered': (beta_0_q05 <= beta_0_true <= beta_0_q95),
        'beta_1_true': beta_1_true,
        'beta_1_mean': beta_1_mean,
        'beta_1_q05': beta_1_q05,
        'beta_1_q95': beta_1_q95,
        'beta_1_covered': (beta_1_q05 <= beta_1_true <= beta_1_q95),
        'phi_true': phi_true,
        'phi_mean': phi_mean,
        'phi_q05': phi_q05,
        'phi_q95': phi_q95,
        'phi_covered': (phi_q05 <= phi_true <= phi_q95),
        'rank_beta_0': rank_beta_0,
        'rank_beta_1': rank_beta_1,
        'rank_phi': rank_phi
    })

    print("SUCCESS")

print(f"\n{'='*80}")
print(f"COMPLETED: {len(results)}/{N_SIMS} successful, {failures} failures")
print(f"{'='*80}\n")

# Save results
results_df = pd.DataFrame(results)
results_df.to_csv(CODE_DIR / "sbc_results.csv", index=False)
print(f"Results saved to: {CODE_DIR / 'sbc_results.csv'}")

# ============================================================================
# ANALYSIS AND METRICS
# ============================================================================

print(f"\n{'='*80}")
print("RECOVERY METRICS")
print(f"{'='*80}\n")

def compute_metrics(param_name, true_col, mean_col, q05_col, q95_col, covered_col):
    """Compute recovery metrics for a parameter"""
    true_vals = results_df[true_col].values
    mean_vals = results_df[mean_col].values

    # Bias
    errors = mean_vals - true_vals
    bias = np.mean(errors)
    bias_sd = np.std(errors) / np.sqrt(len(errors))

    # Standardized bias (relative to true parameter SD)
    true_sd = np.std(true_vals)
    standardized_bias = bias / true_sd if true_sd > 0 else 0

    # Coverage
    coverage = results_df[covered_col].mean()

    # Width
    widths = results_df[q95_col] - results_df[q05_col]
    mean_width = np.mean(widths)

    return {
        'parameter': param_name,
        'bias': bias,
        'bias_sd': bias_sd,
        'standardized_bias': standardized_bias,
        'coverage': coverage,
        'mean_width': mean_width,
        'n_sims': len(results_df)
    }

metrics = []
for param, true_col, mean_col, q05_col, q95_col, covered_col in [
    ('β₀', 'beta_0_true', 'beta_0_mean', 'beta_0_q05', 'beta_0_q95', 'beta_0_covered'),
    ('β₁', 'beta_1_true', 'beta_1_mean', 'beta_1_q05', 'beta_1_q95', 'beta_1_covered'),
    ('φ', 'phi_true', 'phi_mean', 'phi_q05', 'phi_q95', 'phi_covered')
]:
    m = compute_metrics(param, true_col, mean_col, q05_col, q95_col, covered_col)
    metrics.append(m)

    print(f"{param}:")
    print(f"  Bias: {m['bias']:.4f} ± {m['bias_sd']:.4f}")
    print(f"  Standardized bias: {m['standardized_bias']:.4f}")
    print(f"  Coverage (90% CI): {m['coverage']*100:.1f}%")
    print(f"  Mean CI width: {m['mean_width']:.4f}")
    print()

metrics_df = pd.DataFrame(metrics)
metrics_df.to_csv(CODE_DIR / "recovery_metrics.csv", index=False)

# ============================================================================
# RANK STATISTICS (SBC Diagnostic)
# ============================================================================

print(f"{'='*80}")
print("RANK STATISTICS (Uniformity Test)")
print(f"{'='*80}\n")

total_draws = CHAINS * SAMPLES

def test_uniformity(ranks, n_bins=20):
    """Test if ranks are uniformly distributed using chi-square test"""
    expected_per_bin = len(ranks) / n_bins
    hist, _ = np.histogram(ranks, bins=n_bins, range=(0, total_draws))

    chi2_stat = np.sum((hist - expected_per_bin)**2 / expected_per_bin)
    p_value = 1 - stats.chi2.cdf(chi2_stat, df=n_bins-1)

    return chi2_stat, p_value, hist

rank_tests = {}
for param in ['beta_0', 'beta_1', 'phi']:
    chi2, p_val, hist = test_uniformity(ranks[param])
    rank_tests[param] = {'chi2': chi2, 'p_value': p_val}

    param_display = {'beta_0': 'β₀', 'beta_1': 'β₁', 'phi': 'φ'}[param]
    print(f"{param_display}:")
    print(f"  χ² = {chi2:.2f}, p-value = {p_val:.4f}")
    print(f"  {'PASS' if p_val > 0.05 else 'FAIL'} (uniformity test)")
    print()

# ============================================================================
# VISUALIZATION
# ============================================================================

print(f"{'='*80}")
print("GENERATING PLOTS")
print(f"{'='*80}\n")

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100

# 1. Recovery Scatter Plot
print("Creating recovery scatter plots...")
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

params_info = [
    ('beta_0', 'β₀', 'Intercept'),
    ('beta_1', 'β₁', 'Slope'),
    ('phi', 'φ', 'Dispersion')
]

for ax, (param, symbol, name) in zip(axes, params_info):
    true_col = f'{param}_true'
    mean_col = f'{param}_mean'
    q05_col = f'{param}_q05'
    q95_col = f'{param}_q95'

    true_vals = results_df[true_col]
    mean_vals = results_df[mean_col]
    q05_vals = results_df[q05_col]
    q95_vals = results_df[q95_col]

    # Plot identity line
    min_val = min(true_vals.min(), mean_vals.min())
    max_val = max(true_vals.max(), mean_vals.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.3, linewidth=2, label='Perfect recovery')

    # Plot points with error bars
    ax.errorbar(true_vals, mean_vals,
                yerr=[mean_vals - q05_vals, q95_vals - mean_vals],
                fmt='o', alpha=0.6, capsize=3, markersize=5, label='Posterior mean ± 90% CI')

    # Find coverage
    coverage = results_df[f'{param}_covered'].mean()

    ax.set_xlabel(f'True {symbol}', fontsize=11)
    ax.set_ylabel(f'Posterior Mean {symbol}', fontsize=11)
    ax.set_title(f'{name} Recovery\nCoverage: {coverage*100:.1f}%', fontsize=12)
    ax.legend(fontsize=8, loc='best')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(PLOTS_DIR / "parameter_recovery.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: {PLOTS_DIR / 'parameter_recovery.png'}")

# 2. SBC Rank Histograms
print("Creating SBC rank histograms...")
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

n_bins = 20
expected_per_bin = len(results_df) / n_bins

for ax, (param, symbol, name) in zip(axes, params_info):
    # Plot histogram
    counts, bins, patches = ax.hist(ranks[param], bins=n_bins, range=(0, total_draws),
                                     edgecolor='black', alpha=0.7, label='Observed')

    # Expected uniform line
    ax.axhline(expected_per_bin, color='red', linestyle='--', linewidth=2,
               label=f'Expected (uniform)')

    # Confidence band for uniform (95%)
    ci_lower = expected_per_bin - 1.96 * np.sqrt(expected_per_bin)
    ci_upper = expected_per_bin + 1.96 * np.sqrt(expected_per_bin)
    ax.axhspan(ci_lower, ci_upper, alpha=0.2, color='red', label='95% CI')

    # Test results
    chi2 = rank_tests[param]['chi2']
    p_val = rank_tests[param]['p_value']
    status = 'PASS' if p_val > 0.05 else 'FAIL'

    ax.set_xlabel('Rank', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title(f'{name} ({symbol})\nχ² = {chi2:.2f}, p = {p_val:.3f} [{status}]', fontsize=12)
    ax.legend(fontsize=8, loc='best')
    ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(PLOTS_DIR / "sbc_rank_histograms.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: {PLOTS_DIR / 'sbc_rank_histograms.png'}")

# 3. Coverage Plot
print("Creating coverage diagnostic plot...")
fig, ax = plt.subplots(figsize=(10, 6))

param_names = ['β₀', 'β₁', 'φ']
coverages = [results_df['beta_0_covered'].mean(),
             results_df['beta_1_covered'].mean(),
             results_df['phi_covered'].mean()]

x_pos = np.arange(len(param_names))
bars = ax.bar(x_pos, [c * 100 for c in coverages], color=['steelblue', 'forestgreen', 'coral'],
              alpha=0.7, edgecolor='black')

# Target line
ax.axhline(90, color='red', linestyle='--', linewidth=2, label='Target (90%)')
ax.axhspan(80, 95, alpha=0.2, color='green', label='Acceptable range')

# Add percentage labels on bars
for i, (bar, cov) in enumerate(zip(bars, coverages)):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 1,
            f'{cov*100:.1f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')

ax.set_ylabel('Coverage (%)', fontsize=12)
ax.set_xlabel('Parameter', fontsize=12)
ax.set_title('90% Credible Interval Coverage', fontsize=14, fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels(param_names, fontsize=12)
ax.set_ylim(0, 105)
ax.legend(fontsize=10, loc='lower right')
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(PLOTS_DIR / "coverage_diagnostic.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: {PLOTS_DIR / 'coverage_diagnostic.png'}")

# 4. Bias diagnostic
print("Creating bias diagnostic plot...")
fig, ax = plt.subplots(figsize=(10, 6))

biases = [metrics_df.loc[metrics_df['parameter'] == p, 'standardized_bias'].values[0]
          for p in ['β₀', 'β₁', 'φ']]

bars = ax.bar(x_pos, biases, color=['steelblue', 'forestgreen', 'coral'],
              alpha=0.7, edgecolor='black')

# Reference lines
ax.axhline(0, color='black', linestyle='-', linewidth=1)
ax.axhspan(-0.2, 0.2, alpha=0.2, color='green', label='Acceptable (|bias| < 0.2 SD)')
ax.axhspan(-0.5, 0.5, alpha=0.1, color='yellow')

# Add value labels
for i, (bar, bias) in enumerate(zip(bars, biases)):
    height = bar.get_height()
    y_pos = height + 0.01 if height >= 0 else height - 0.01
    va = 'bottom' if height >= 0 else 'top'
    ax.text(bar.get_x() + bar.get_width()/2., y_pos,
            f'{bias:.3f}', ha='center', va=va, fontsize=12, fontweight='bold')

ax.set_ylabel('Standardized Bias (bias / true parameter SD)', fontsize=12)
ax.set_xlabel('Parameter', fontsize=12)
ax.set_title('Parameter Recovery Bias', fontsize=14, fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels(param_names, fontsize=12)
ax.legend(fontsize=10, loc='best')
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(PLOTS_DIR / "bias_diagnostic.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: {PLOTS_DIR / 'bias_diagnostic.png'}")

# ============================================================================
# FINAL DECISION
# ============================================================================

print(f"\n{'='*80}")
print("FINAL ASSESSMENT")
print(f"{'='*80}\n")

# Check criteria
issues = []
warnings_list = []

# Coverage checks
for param in ['β₀', 'β₁', 'φ']:
    cov = results_df[f"{param.replace('₀', '_0').replace('₁', '_1').replace('φ', 'phi')}_covered"].mean()
    if cov < 0.70:
        issues.append(f"{param} coverage critically low: {cov*100:.1f}%")
    elif cov < 0.80:
        warnings_list.append(f"{param} coverage below target: {cov*100:.1f}%")
    elif cov > 0.98:
        issues.append(f"{param} coverage suspiciously high: {cov*100:.1f}%")
    elif cov > 0.95:
        warnings_list.append(f"{param} coverage above target: {cov*100:.1f}%")

# Rank uniformity checks
for param in ['beta_0', 'beta_1', 'phi']:
    p_val = rank_tests[param]['p_value']
    param_display = {'beta_0': 'β₀', 'beta_1': 'β₁', 'phi': 'φ'}[param]
    if p_val < 0.01:
        issues.append(f"{param_display} ranks highly non-uniform (p = {p_val:.4f})")
    elif p_val < 0.05:
        warnings_list.append(f"{param_display} ranks marginally non-uniform (p = {p_val:.4f})")

# Bias checks
for i, param in enumerate(['β₀', 'β₁', 'φ']):
    std_bias = biases[i]
    if abs(std_bias) > 0.5:
        issues.append(f"{param} systematic bias |{std_bias:.3f}| > 0.5 SD")
    elif abs(std_bias) > 0.2:
        warnings_list.append(f"{param} bias {std_bias:.3f} exceeds 0.2 SD threshold")

# Computational issues
if failures > N_SIMS * 0.1:
    issues.append(f"High failure rate: {failures}/{N_SIMS} ({failures/N_SIMS*100:.1f}%)")

# Print decision
if len(issues) > 0:
    print("RESULT: FAIL")
    print("\nCritical Issues:")
    for issue in issues:
        print(f"  - {issue}")
    if len(warnings_list) > 0:
        print("\nAdditional Warnings:")
        for warning in warnings_list:
            print(f"  - {warning}")
else:
    if len(warnings_list) > 0:
        print("RESULT: PASS (with warnings)")
        print("\nWarnings:")
        for warning in warnings_list:
            print(f"  - {warning}")
    else:
        print("RESULT: PASS")
        print("\nAll validation criteria met:")
        print("  ✓ Coverage within acceptable range (80-95%)")
        print("  ✓ Rank statistics uniform (p > 0.05)")
        print("  ✓ No systematic bias (|bias| < 0.2 SD)")
        print("  ✓ Computation stable (< 10% failures)")

print(f"\n{'='*80}")
print("SBC VALIDATION COMPLETE")
print(f"{'='*80}\n")

# Save final decision
decision = {
    'result': 'FAIL' if len(issues) > 0 else ('PASS_WITH_WARNINGS' if len(warnings_list) > 0 else 'PASS'),
    'issues': issues,
    'warnings': warnings_list,
    'successful_sims': len(results_df),
    'failed_sims': failures,
    'metrics': metrics_df.to_dict('records'),
    'rank_tests': rank_tests
}

with open(CODE_DIR / "sbc_decision.json", "w") as f:
    json.dump(decision, f, indent=2)

print(f"Decision saved to: {CODE_DIR / 'sbc_decision.json'}")
