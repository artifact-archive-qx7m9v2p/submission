#!/usr/bin/env python
"""
Simulation-Based Calibration for Negative Binomial Regression
Final version with 50 simulations
"""
import sys
sys.path.insert(0, '/tmp/agent-home/.local/lib/python3.13/site-packages')

import numpy as np
import pandas as pd
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pymc as pm
import arviz as az
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

import logging
logging.getLogger("pytensor").setLevel(logging.ERROR)
logging.getLogger("pymc").setLevel(logging.ERROR)

np.random.seed(42)

# Paths
BASE_DIR = Path("/workspace/experiments/experiment_1/simulation_based_validation")
CODE_DIR = BASE_DIR / "code"
PLOTS_DIR = BASE_DIR / "plots"

# Load data
with open("/workspace/data/data.csv", "r") as f:
    real_data = json.load(f)

n = real_data["n"]
year = np.array(real_data["year"])

# Configuration
N_SIMS = 50
WARMUP = 500
SAMPLES = 500
CHAINS = 2
TARGET_ACCEPT = 0.9

# Priors
PRIOR_BETA_0_MEAN = 4.3
PRIOR_BETA_0_SD = 1.0
PRIOR_BETA_1_MEAN = 0.85
PRIOR_BETA_1_SD = 0.5
PRIOR_PHI_RATE = 0.667

print("=" * 80)
print("SIMULATION-BASED CALIBRATION")
print("=" * 80)
print(f"\nConfiguration:")
print(f"  Simulations: {N_SIMS}")
print(f"  Chains: {CHAINS}, Warmup: {WARMUP}, Samples: {SAMPLES}")
print(f"  Total draws: {CHAINS * SAMPLES}")
print(f"\nPriors:")
print(f"  β₀ ~ N({PRIOR_BETA_0_MEAN}, {PRIOR_BETA_0_SD})")
print(f"  β₁ ~ N({PRIOR_BETA_1_MEAN}, {PRIOR_BETA_1_SD})")
print(f"  φ ~ Exp({PRIOR_PHI_RATE}), E[φ]={1/PRIOR_PHI_RATE:.2f}")

def draw_from_prior():
    return (np.random.normal(PRIOR_BETA_0_MEAN, PRIOR_BETA_0_SD),
            np.random.normal(PRIOR_BETA_1_MEAN, PRIOR_BETA_1_SD),
            np.random.exponential(1 / PRIOR_PHI_RATE))

def simulate_data(beta_0, beta_1, phi, year):
    mu = np.exp(beta_0 + beta_1 * year)
    p = phi / (phi + mu)
    return np.random.negative_binomial(n=phi, p=p, size=len(year)).astype(int)

def fit_model(C_data):
    with pm.Model() as model:
        beta_0 = pm.Normal('beta_0', mu=PRIOR_BETA_0_MEAN, sigma=PRIOR_BETA_0_SD)
        beta_1 = pm.Normal('beta_1', mu=PRIOR_BETA_1_MEAN, sigma=PRIOR_BETA_1_SD)
        phi = pm.Exponential('phi', lam=PRIOR_PHI_RATE)
        mu = pm.math.exp(beta_0 + beta_1 * year)
        C = pm.NegativeBinomial('C', mu=mu, alpha=phi, observed=C_data)
        trace = pm.sample(draws=SAMPLES, tune=WARMUP, chains=CHAINS,
                         target_accept=TARGET_ACCEPT, return_inferencedata=True,
                         progressbar=False, cores=1,
                         random_seed=np.random.randint(0, 10000))
    return trace

# Run SBC
print(f"\n{'='*80}")
print("RUNNING SIMULATIONS")
print(f"{'='*80}\n")

results = []
ranks = {'beta_0': [], 'beta_1': [], 'phi': []}
failures = 0
start_time = __import__('time').time()

for sim in range(N_SIMS):
    elapsed = __import__('time').time() - start_time
    avg_time = elapsed / (sim + 1) if sim > 0 else 0
    est_remaining = avg_time * (N_SIMS - sim - 1)

    print(f"[{sim+1}/{N_SIMS}] ", end="", flush=True)
    if sim > 0:
        print(f"(~{est_remaining/60:.1f}min remaining) ", end="", flush=True)

    beta_0_true, beta_1_true, phi_true = draw_from_prior()
    C_sim = simulate_data(beta_0_true, beta_1_true, phi_true, year)

    try:
        trace = fit_model(C_sim)
        summary = az.summary(trace, var_names=['beta_0', 'beta_1', 'phi'])

        if summary['r_hat'].max() > 1.05 or summary['ess_bulk'].min() < 100:
            print(f"FAIL (convergence)")
            failures += 1
            continue

        post = trace.posterior
        beta_0_post = post['beta_0'].values.flatten()
        beta_1_post = post['beta_1'].values.flatten()
        phi_post = post['phi'].values.flatten()

        results.append({
            'sim': sim + 1,
            'beta_0_true': beta_0_true,
            'beta_0_mean': beta_0_post.mean(),
            'beta_0_q05': np.percentile(beta_0_post, 5),
            'beta_0_q95': np.percentile(beta_0_post, 95),
            'beta_0_covered': np.percentile(beta_0_post, 5) <= beta_0_true <= np.percentile(beta_0_post, 95),
            'beta_1_true': beta_1_true,
            'beta_1_mean': beta_1_post.mean(),
            'beta_1_q05': np.percentile(beta_1_post, 5),
            'beta_1_q95': np.percentile(beta_1_post, 95),
            'beta_1_covered': np.percentile(beta_1_post, 5) <= beta_1_true <= np.percentile(beta_1_post, 95),
            'phi_true': phi_true,
            'phi_mean': phi_post.mean(),
            'phi_q05': np.percentile(phi_post, 5),
            'phi_q95': np.percentile(phi_post, 95),
            'phi_covered': np.percentile(phi_post, 5) <= phi_true <= np.percentile(phi_post, 95),
            'rank_beta_0': np.sum(beta_0_post < beta_0_true),
            'rank_beta_1': np.sum(beta_1_post < beta_1_true),
            'rank_phi': np.sum(phi_post < phi_true)
        })

        ranks['beta_0'].append(np.sum(beta_0_post < beta_0_true))
        ranks['beta_1'].append(np.sum(beta_1_post < beta_1_true))
        ranks['phi'].append(np.sum(phi_post < phi_true))

        print("OK")

    except Exception as e:
        print(f"FAIL ({str(e)[:30]})")
        failures += 1

total_time = __import__('time').time() - start_time
print(f"\n{'='*80}")
print(f"COMPLETED: {len(results)}/{N_SIMS} successful, {failures} failures")
print(f"Total time: {total_time/60:.1f} minutes")
print(f"{'='*80}\n")

if len(results) == 0:
    print("ERROR: No successful simulations!")
    sys.exit(1)

results_df = pd.DataFrame(results)
results_df.to_csv(CODE_DIR / "sbc_results.csv", index=False)

# Compute metrics
print("RECOVERY METRICS\n")

metrics = []
for param, prefix in [('β₀', 'beta_0'), ('β₁', 'beta_1'), ('φ', 'phi')]:
    true_vals = results_df[f'{prefix}_true'].values
    mean_vals = results_df[f'{prefix}_mean'].values
    errors = mean_vals - true_vals
    bias = errors.mean()
    std_bias = bias / true_vals.std() if true_vals.std() > 0 else 0
    coverage = results_df[f'{prefix}_covered'].mean()
    width = (results_df[f'{prefix}_q95'] - results_df[f'{prefix}_q05']).mean()

    metrics.append({
        'parameter': param,
        'bias': bias,
        'standardized_bias': std_bias,
        'coverage': coverage,
        'mean_width': width,
        'n_sims': len(results_df)
    })

    print(f"{param}:")
    print(f"  Bias: {bias:.4f}")
    print(f"  Std bias: {std_bias:.4f}")
    print(f"  Coverage: {coverage*100:.1f}%")
    print(f"  CI width: {width:.4f}\n")

metrics_df = pd.DataFrame(metrics)
metrics_df.to_csv(CODE_DIR / "recovery_metrics.csv", index=False)

# Rank statistics
print("RANK UNIFORMITY\n")

total_draws = CHAINS * SAMPLES
def test_uniformity(ranks_list):
    expected = len(ranks_list) / 20
    hist, _ = np.histogram(ranks_list, bins=20, range=(0, total_draws))
    chi2 = np.sum((hist - expected)**2 / expected)
    p_val = 1 - stats.chi2.cdf(chi2, df=19)
    return chi2, p_val

rank_tests = {}
for param in ['beta_0', 'beta_1', 'phi']:
    chi2, p_val = test_uniformity(ranks[param])
    rank_tests[param] = {'chi2': chi2, 'p_value': p_val}

    param_name = {'beta_0': 'β₀', 'beta_1': 'β₁', 'phi': 'φ'}[param]
    print(f"{param_name}: χ²={chi2:.2f}, p={p_val:.4f} [{'PASS' if p_val > 0.05 else 'FAIL'}]")

# Generate plots
print(f"\n{'='*80}")
print("GENERATING PLOTS")
print(f"{'='*80}\n")

sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100

# 1. Recovery scatter
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, (param, symbol, name) in zip(axes, [('beta_0', 'β₀', 'Intercept'),
                                              ('beta_1', 'β₁', 'Slope'),
                                              ('phi', 'φ', 'Dispersion')]):
    true = results_df[f'{param}_true']
    mean = results_df[f'{param}_mean']
    q05 = results_df[f'{param}_q05']
    q95 = results_df[f'{param}_q95']

    ax.plot([true.min(), true.max()], [true.min(), true.max()], 'k--', alpha=0.3, linewidth=2)
    ax.errorbar(true, mean, yerr=[mean - q05, q95 - mean], fmt='o', alpha=0.6, capsize=3, markersize=5)

    cov = results_df[f'{param}_covered'].mean()
    ax.set_xlabel(f'True {symbol}', fontsize=11)
    ax.set_ylabel(f'Posterior Mean {symbol}', fontsize=11)
    ax.set_title(f'{name} Recovery\nCoverage: {cov*100:.1f}%', fontsize=12)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(PLOTS_DIR / "parameter_recovery.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved: parameter_recovery.png")

# 2. Rank histograms
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
expected = len(results_df) / 20

for ax, (param, symbol, name) in zip(axes, [('beta_0', 'β₀', 'Intercept'),
                                              ('beta_1', 'β₁', 'Slope'),
                                              ('phi', 'φ', 'Dispersion')]):
    ax.hist(ranks[param], bins=20, range=(0, total_draws), edgecolor='black', alpha=0.7)
    ax.axhline(expected, color='red', linestyle='--', linewidth=2)
    ci_low = expected - 1.96 * np.sqrt(expected)
    ci_high = expected + 1.96 * np.sqrt(expected)
    ax.axhspan(ci_low, ci_high, alpha=0.2, color='red')

    chi2 = rank_tests[param]['chi2']
    p = rank_tests[param]['p_value']
    status = 'PASS' if p > 0.05 else 'FAIL'

    ax.set_xlabel('Rank', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title(f'{name} ({symbol})\nχ²={chi2:.2f}, p={p:.3f} [{status}]', fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(PLOTS_DIR / "sbc_rank_histograms.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved: sbc_rank_histograms.png")

# 3. Coverage
fig, ax = plt.subplots(figsize=(10, 6))
param_names = ['β₀', 'β₁', 'φ']
coverages = [results_df['beta_0_covered'].mean(),
             results_df['beta_1_covered'].mean(),
             results_df['phi_covered'].mean()]

x_pos = np.arange(len(param_names))
bars = ax.bar(x_pos, [c * 100 for c in coverages],
              color=['steelblue', 'forestgreen', 'coral'],
              alpha=0.7, edgecolor='black')

ax.axhline(90, color='red', linestyle='--', linewidth=2, label='Target')
ax.axhspan(80, 95, alpha=0.2, color='green', label='Acceptable')

for bar, cov in zip(bars, coverages):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
            f'{cov*100:.1f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')

ax.set_ylabel('Coverage (%)', fontsize=12)
ax.set_xlabel('Parameter', fontsize=12)
ax.set_title('90% Credible Interval Coverage', fontsize=14, fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels(param_names, fontsize=12)
ax.set_ylim(0, 105)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(PLOTS_DIR / "coverage_diagnostic.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved: coverage_diagnostic.png")

# 4. Bias
fig, ax = plt.subplots(figsize=(10, 6))
biases = [m['standardized_bias'] for m in metrics]

bars = ax.bar(x_pos, biases, color=['steelblue', 'forestgreen', 'coral'],
              alpha=0.7, edgecolor='black')

ax.axhline(0, color='black', linestyle='-', linewidth=1)
ax.axhspan(-0.2, 0.2, alpha=0.2, color='green', label='Acceptable')

for bar, bias in zip(bars, biases):
    y = bar.get_height() + (0.01 if bar.get_height() >= 0 else -0.01)
    va = 'bottom' if bar.get_height() >= 0 else 'top'
    ax.text(bar.get_x() + bar.get_width()/2., y, f'{bias:.3f}',
            ha='center', va=va, fontsize=12, fontweight='bold')

ax.set_ylabel('Standardized Bias', fontsize=12)
ax.set_xlabel('Parameter', fontsize=12)
ax.set_title('Parameter Recovery Bias', fontsize=14, fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels(param_names, fontsize=12)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(PLOTS_DIR / "bias_diagnostic.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved: bias_diagnostic.png")

# Final assessment
print(f"\n{'='*80}")
print("FINAL ASSESSMENT")
print(f"{'='*80}\n")

issues = []
warnings_list = []

# Check coverage
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

# Check ranks
for param in ['beta_0', 'beta_1', 'phi']:
    p_val = rank_tests[param]['p_value']
    param_name = {'beta_0': 'β₀', 'beta_1': 'β₁', 'phi': 'φ'}[param]
    if p_val < 0.01:
        issues.append(f"{param_name} ranks highly non-uniform (p={p_val:.4f})")
    elif p_val < 0.05:
        warnings_list.append(f"{param_name} ranks marginally non-uniform (p={p_val:.4f})")

# Check bias
for m in metrics:
    if abs(m['standardized_bias']) > 0.5:
        issues.append(f"{m['parameter']} systematic bias |{m['standardized_bias']:.3f}| > 0.5 SD")
    elif abs(m['standardized_bias']) > 0.2:
        warnings_list.append(f"{m['parameter']} bias {m['standardized_bias']:.3f} exceeds 0.2 SD")

# Check failures
if failures > N_SIMS * 0.1:
    issues.append(f"High failure rate: {failures}/{N_SIMS} ({failures/N_SIMS*100:.1f}%)")

# Print decision
if issues:
    print("RESULT: FAIL\n")
    print("Critical Issues:")
    for issue in issues:
        print(f"  - {issue}")
    if warnings_list:
        print("\nWarnings:")
        for warning in warnings_list:
            print(f"  - {warning}")
else:
    if warnings_list:
        print("RESULT: PASS (with warnings)\n")
        print("Warnings:")
        for warning in warnings_list:
            print(f"  - {warning}")
    else:
        print("RESULT: PASS\n")
        print("All validation criteria met:")
        print("  ✓ Coverage within range (80-95%)")
        print("  ✓ Ranks uniform (p > 0.05)")
        print("  ✓ No systematic bias (|bias| < 0.2 SD)")
        print("  ✓ Computation stable (< 10% failures)")

print(f"\n{'='*80}")
print("SBC VALIDATION COMPLETE")
print(f"{'='*80}")

# Save decision
decision = {
    'result': 'FAIL' if issues else ('PASS_WITH_WARNINGS' if warnings_list else 'PASS'),
    'issues': issues,
    'warnings': warnings_list,
    'successful_sims': len(results_df),
    'failed_sims': failures,
    'metrics': metrics_df.to_dict('records'),
    'rank_tests': rank_tests
}

with open(CODE_DIR / "sbc_decision.json", "w") as f:
    json.dump(decision, f, indent=2)

print(f"\nAll files saved to: {CODE_DIR} and {PLOTS_DIR}")
