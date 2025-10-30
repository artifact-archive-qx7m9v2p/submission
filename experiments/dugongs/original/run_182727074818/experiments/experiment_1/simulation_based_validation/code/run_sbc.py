"""
Simulation-Based Calibration for Robust Logarithmic Regression
================================================================

Tests if the model can recover known parameters through:
1. Drawing parameters from priors
2. Generating synthetic data
3. Fitting model to synthetic data
4. Checking if posteriors recover true parameters

This is a critical validation step before fitting real data.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from cmdstanpy import CmdStanModel
from scipy import stats
from pathlib import Path
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# Paths
BASE_DIR = Path('/workspace/experiments/experiment_1/simulation_based_validation')
CODE_DIR = BASE_DIR / 'code'
PLOTS_DIR = BASE_DIR / 'plots'
PLOTS_DIR.mkdir(exist_ok=True)

# Load actual x values from data
data_path = Path('/workspace/data/data.csv')
data_df = pd.read_csv(data_path)
x_obs = data_df['x'].values
N = len(x_obs)

print(f"Loaded {N} observations with x values ranging from {x_obs.min():.1f} to {x_obs.max():.1f}")
print(f"Starting Simulation-Based Calibration at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

# Compile Stan model
print("Compiling Stan model...")
stan_file = CODE_DIR / 'robust_log_regression.stan'
model = CmdStanModel(stan_file=str(stan_file))
print("Model compiled successfully!\n")

# SBC Configuration
N_SIMS = 100  # Number of simulations
N_CHAINS = 4
N_ITER = 1000
N_WARMUP = 500
N_POSTERIOR_SAMPLES = (N_ITER - N_WARMUP) * N_CHAINS  # 2000 samples

# Parameter names
PARAMS = ['alpha', 'beta', 'c', 'nu', 'sigma']

# Prior sampling functions (matching Stan priors)
def sample_from_priors():
    """Draw parameter values from prior distributions"""
    return {
        'alpha': np.random.normal(2.0, 0.5),
        'beta': np.random.normal(0.3, 0.2),
        'c': np.random.gamma(2, 1/2),  # gamma(shape, scale) = gamma(shape, 1/rate)
        'nu': np.random.gamma(2, 1/0.1),
        'sigma': np.abs(np.random.normal(0, 0.15))  # half-normal
    }

def generate_data(params, x):
    """Generate synthetic data given parameters and x values"""
    alpha, beta, c, nu, sigma = params['alpha'], params['beta'], params['c'], params['nu'], params['sigma']

    # Compute mean function
    mu = alpha + beta * np.log(x + c)

    # Generate Y from Student-t distribution
    Y = mu + sigma * stats.t.rvs(df=nu, size=len(x))

    return Y

# Storage for results
results = {
    'ranks': {p: [] for p in PARAMS},
    'true_values': {p: [] for p in PARAMS},
    'posterior_means': {p: [] for p in PARAMS},
    'posterior_sds': {p: [] for p in PARAMS},
    'in_95_CI': {p: [] for p in PARAMS},
    'in_90_CI': {p: [] for p in PARAMS},
    'z_scores': {p: [] for p in PARAMS},
    'convergence': {
        'max_rhat': [],
        'min_ess_bulk': [],
        'n_divergences': [],
        'sampling_time': []
    },
    'failed_sims': []
}

# Run SBC simulations
print(f"Running {N_SIMS} SBC simulations...")
print("=" * 70)

for sim in range(N_SIMS):
    print(f"Simulation {sim+1}/{N_SIMS}...", end=" ", flush=True)

    try:
        # Step 1: Sample parameters from priors
        true_params = sample_from_priors()

        # Step 2: Generate synthetic data
        Y_sim = generate_data(true_params, x_obs)

        # Step 3: Fit model to synthetic data
        stan_data = {
            'N': N,
            'x': x_obs.tolist(),
            'Y': Y_sim.tolist()
        }

        fit = model.sample(
            data=stan_data,
            chains=N_CHAINS,
            iter_sampling=N_ITER - N_WARMUP,
            iter_warmup=N_WARMUP,
            show_progress=False,
            show_console=False,
            seed=sim  # Different seed for each simulation
        )

        # Step 4: Extract posteriors and compute ranks
        for param in PARAMS:
            # Get posterior samples (all chains combined)
            posterior_samples = fit.stan_variable(param)

            # Compute rank of true value within posterior samples
            rank = np.sum(posterior_samples < true_params[param])

            # Compute statistics
            post_mean = np.mean(posterior_samples)
            post_sd = np.std(posterior_samples)

            # Credible intervals
            ci_95 = np.percentile(posterior_samples, [2.5, 97.5])
            ci_90 = np.percentile(posterior_samples, [5, 95])

            # Z-score
            z_score = (true_params[param] - post_mean) / post_sd if post_sd > 0 else 0

            # Store results
            results['ranks'][param].append(rank)
            results['true_values'][param].append(true_params[param])
            results['posterior_means'][param].append(post_mean)
            results['posterior_sds'][param].append(post_sd)
            results['in_95_CI'][param].append(ci_95[0] <= true_params[param] <= ci_95[1])
            results['in_90_CI'][param].append(ci_90[0] <= true_params[param] <= ci_90[1])
            results['z_scores'][param].append(z_score)

        # Step 5: Check convergence diagnostics
        summary = fit.summary()

        # Get max Rhat and min ESS for monitored parameters
        param_rows = summary[summary.index.isin(PARAMS)]
        max_rhat = param_rows['R_hat'].max()
        min_ess = param_rows['N_Eff'].min()

        # Count divergences
        divergences = fit.method_variables().get('divergent__', np.zeros(N_POSTERIOR_SAMPLES))
        n_divergences = np.sum(divergences)

        results['convergence']['max_rhat'].append(max_rhat)
        results['convergence']['min_ess_bulk'].append(min_ess)
        results['convergence']['n_divergences'].append(n_divergences)

        print(f"OK (Rhat={max_rhat:.3f}, ESS={min_ess:.0f}, Div={n_divergences})")

    except Exception as e:
        print(f"FAILED - {str(e)}")
        results['failed_sims'].append(sim)

print("=" * 70)
print(f"\nCompleted {N_SIMS - len(results['failed_sims'])}/{N_SIMS} simulations successfully")
print(f"Failed simulations: {len(results['failed_sims'])}\n")

# Save raw results
results_file = BASE_DIR / 'code' / 'sbc_results.json'
# Convert numpy types to Python types for JSON serialization
results_serializable = {
    'ranks': {p: [int(r) for r in results['ranks'][p]] for p in PARAMS},
    'true_values': {p: [float(v) for v in results['true_values'][p]] for p in PARAMS},
    'posterior_means': {p: [float(v) for v in results['posterior_means'][p]] for p in PARAMS},
    'posterior_sds': {p: [float(v) for v in results['posterior_sds'][p]] for p in PARAMS},
    'in_95_CI': {p: [bool(v) for v in results['in_95_CI'][p]] for p in PARAMS},
    'in_90_CI': {p: [bool(v) for v in results['in_90_CI'][p]] for p in PARAMS},
    'z_scores': {p: [float(v) for v in results['z_scores'][p]] for p in PARAMS},
    'convergence': {
        'max_rhat': [float(v) for v in results['convergence']['max_rhat']],
        'min_ess_bulk': [float(v) for v in results['convergence']['min_ess_bulk']],
        'n_divergences': [int(v) for v in results['convergence']['n_divergences']],
    },
    'failed_sims': results['failed_sims'],
    'config': {
        'n_sims': N_SIMS,
        'n_chains': N_CHAINS,
        'n_iter': N_ITER,
        'n_warmup': N_WARMUP,
        'n_observations': N,
        'seed': 42
    }
}

with open(results_file, 'w') as f:
    json.dump(results_serializable, f, indent=2)

print(f"Raw results saved to: {results_file}")

# ============================================================================
# VISUALIZATION AND DIAGNOSTICS
# ============================================================================

print("\nGenerating diagnostic plots...")

# Set plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# 1. RANK HISTOGRAMS (Primary SBC diagnostic)
# ============================================================================
print("  - Rank histograms")

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

# Expected uniform distribution
n_bins = 20
expected_per_bin = len(results['ranks'][PARAMS[0]]) / n_bins

for idx, param in enumerate(PARAMS):
    ax = axes[idx]

    ranks = results['ranks'][param]

    # Plot histogram
    counts, bins, patches = ax.hist(ranks, bins=n_bins, edgecolor='black', alpha=0.7)

    # Add expected uniform line
    ax.axhline(expected_per_bin, color='red', linestyle='--', linewidth=2,
               label=f'Expected (uniform)')

    # Chi-square test for uniformity
    chi2_stat, p_value = stats.chisquare(counts)

    # Color based on p-value
    if p_value < 0.01:
        title_color = 'red'
        result = 'FAIL'
    elif p_value < 0.05:
        title_color = 'orange'
        result = 'WARN'
    else:
        title_color = 'green'
        result = 'PASS'

    ax.set_title(f'{param}: χ²={chi2_stat:.1f}, p={p_value:.3f} [{result}]',
                 fontsize=12, fontweight='bold', color=title_color)
    ax.set_xlabel('Rank')
    ax.set_ylabel('Frequency')
    ax.legend()
    ax.grid(alpha=0.3)

# Remove extra subplot
axes[-1].axis('off')

plt.tight_layout()
plt.savefig(PLOTS_DIR / 'rank_histograms.png', dpi=300, bbox_inches='tight')
plt.close()

# 2. Z-SCORE PLOTS (Check for bias and calibration)
# ============================================================================
print("  - Z-score distributions")

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for idx, param in enumerate(PARAMS):
    ax = axes[idx]

    z_scores = results['z_scores'][param]

    # Histogram of z-scores
    ax.hist(z_scores, bins=20, edgecolor='black', alpha=0.7, density=True)

    # Overlay standard normal
    x_range = np.linspace(-4, 4, 100)
    ax.plot(x_range, stats.norm.pdf(x_range), 'r-', linewidth=2,
            label='Standard Normal')

    # Add vertical line at mean
    mean_z = np.mean(z_scores)
    ax.axvline(mean_z, color='blue', linestyle='--', linewidth=2,
               label=f'Mean = {mean_z:.3f}')

    # Assess bias
    if abs(mean_z) > 0.3:
        title_color = 'red'
        result = 'BIASED'
    elif abs(mean_z) > 0.2:
        title_color = 'orange'
        result = 'SLIGHT BIAS'
    else:
        title_color = 'green'
        result = 'UNBIASED'

    ax.set_title(f'{param}: Mean Z = {mean_z:.3f} [{result}]',
                 fontsize=12, fontweight='bold', color=title_color)
    ax.set_xlabel('Z-score')
    ax.set_ylabel('Density')
    ax.legend()
    ax.grid(alpha=0.3)

axes[-1].axis('off')

plt.tight_layout()
plt.savefig(PLOTS_DIR / 'z_score_distributions.png', dpi=300, bbox_inches='tight')
plt.close()

# 3. PARAMETER RECOVERY PLOT (Shrinkage)
# ============================================================================
print("  - Parameter recovery (true vs posterior mean)")

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for idx, param in enumerate(PARAMS):
    ax = axes[idx]

    true_vals = results['true_values'][param]
    post_means = results['posterior_means'][param]

    # Scatter plot
    ax.scatter(true_vals, post_means, alpha=0.6, s=50)

    # Perfect recovery line
    min_val = min(min(true_vals), min(post_means))
    max_val = max(max(true_vals), max(post_means))
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2,
            label='Perfect Recovery')

    # Compute correlation (shrinkage measure)
    corr = np.corrcoef(true_vals, post_means)[0, 1]

    # Compute RMSE
    rmse = np.sqrt(np.mean((np.array(true_vals) - np.array(post_means))**2))

    # Compute bias
    bias = np.mean(np.array(post_means) - np.array(true_vals))

    ax.set_title(f'{param}: r={corr:.3f}, RMSE={rmse:.4f}, Bias={bias:.4f}',
                 fontsize=12, fontweight='bold')
    ax.set_xlabel('True Value')
    ax.set_ylabel('Posterior Mean')
    ax.legend()
    ax.grid(alpha=0.3)

axes[-1].axis('off')

plt.tight_layout()
plt.savefig(PLOTS_DIR / 'parameter_recovery.png', dpi=300, bbox_inches='tight')
plt.close()

# 4. COVERAGE CALIBRATION PLOT
# ============================================================================
print("  - Coverage calibration")

fig, ax = plt.subplots(1, 1, figsize=(10, 6))

coverage_90 = [np.mean(results['in_90_CI'][p]) * 100 for p in PARAMS]
coverage_95 = [np.mean(results['in_95_CI'][p]) * 100 for p in PARAMS]

x_pos = np.arange(len(PARAMS))
width = 0.35

bars1 = ax.bar(x_pos - width/2, coverage_90, width, label='90% CI', alpha=0.8)
bars2 = ax.bar(x_pos + width/2, coverage_95, width, label='95% CI', alpha=0.8)

# Add nominal coverage lines
ax.axhline(90, color='blue', linestyle='--', linewidth=2, alpha=0.5, label='Nominal 90%')
ax.axhline(95, color='orange', linestyle='--', linewidth=2, alpha=0.5, label='Nominal 95%')

# Add acceptable range bands
ax.axhspan(88, 92, alpha=0.1, color='blue')
ax.axhspan(93, 97, alpha=0.1, color='orange')

ax.set_xlabel('Parameter')
ax.set_ylabel('Empirical Coverage (%)')
ax.set_title('Credible Interval Coverage Calibration', fontsize=14, fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels(PARAMS)
ax.legend()
ax.grid(alpha=0.3, axis='y')
ax.set_ylim([80, 100])

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig(PLOTS_DIR / 'coverage_calibration.png', dpi=300, bbox_inches='tight')
plt.close()

# 5. CONVERGENCE DIAGNOSTICS SUMMARY
# ============================================================================
print("  - Convergence diagnostics")

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Rhat distribution
ax = axes[0]
rhats = results['convergence']['max_rhat']
ax.hist(rhats, bins=20, edgecolor='black', alpha=0.7)
ax.axvline(1.01, color='red', linestyle='--', linewidth=2, label='Threshold (1.01)')
ax.axvline(np.mean(rhats), color='blue', linestyle='--', linewidth=2,
           label=f'Mean = {np.mean(rhats):.4f}')
ax.set_xlabel('Max R-hat')
ax.set_ylabel('Frequency')
ax.set_title('Convergence: R-hat Distribution', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)

# ESS distribution
ax = axes[1]
ess = results['convergence']['min_ess_bulk']
ax.hist(ess, bins=20, edgecolor='black', alpha=0.7)
ax.axvline(400, color='red', linestyle='--', linewidth=2, label='Threshold (400)')
ax.axvline(np.mean(ess), color='blue', linestyle='--', linewidth=2,
           label=f'Mean = {np.mean(ess):.0f}')
ax.set_xlabel('Min ESS')
ax.set_ylabel('Frequency')
ax.set_title('Effective Sample Size Distribution', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)

# Divergences
ax = axes[2]
divs = results['convergence']['n_divergences']
ax.hist(divs, bins=20, edgecolor='black', alpha=0.7)
ax.axvline(np.mean(divs), color='blue', linestyle='--', linewidth=2,
           label=f'Mean = {np.mean(divs):.1f}')
ax.set_xlabel('Number of Divergences')
ax.set_ylabel('Frequency')
ax.set_title('Divergent Transitions', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(PLOTS_DIR / 'convergence_diagnostics.png', dpi=300, bbox_inches='tight')
plt.close()

print("\nAll plots saved to:", PLOTS_DIR)
print("\n" + "="*70)
print("SBC COMPLETE")
print("="*70)
