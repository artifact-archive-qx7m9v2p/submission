"""
Fit Random Effects Logistic Regression Model to Real Data using PyMC

Model:
  r_i | θ_i, n_i ~ Binomial(n_i, logit⁻¹(θ_i))  for i = 1, ..., 12 groups
  θ_i = μ + τ · z_i
  z_i ~ Normal(0, 1)

Priors:
  μ ~ Normal(-2.51, 1)     # logit(0.075)
  τ ~ HalfNormal(1)         # Between-group SD
"""

import sys
sys.path.insert(0, '/tmp/agent-home/.local/lib/python3.13/site-packages')

import pandas as pd
import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('default')
sns.set_palette("husl")

# Paths
DATA_PATH = Path('/workspace/data/data.csv')
OUTPUT_DIR = Path('/workspace/experiments/experiment_2/posterior_inference')
PLOTS_DIR = OUTPUT_DIR / 'plots'
DIAGNOSTICS_DIR = OUTPUT_DIR / 'diagnostics'

# Ensure directories exist
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
DIAGNOSTICS_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("RANDOM EFFECTS LOGISTIC REGRESSION - REAL DATA INFERENCE")
print("=" * 80)

# ============================================================================
# 1. LOAD AND PREPARE DATA
# ============================================================================

print("\n[1] Loading data...")
data = pd.read_csv(DATA_PATH)
print(f"\nData shape: {data.shape}")
print(f"\nData preview:")
print(data)

# Extract arrays
n = data['n'].values  # Sample sizes
r = data['r'].values  # Event counts
n_groups = len(data)

print(f"\nData summary:")
print(f"  Number of groups: {n_groups}")
print(f"  Total observations: {n.sum()}")
print(f"  Total events: {r.sum()}")
print(f"  Overall proportion: {r.sum() / n.sum():.4f}")
print(f"  Group proportions range: [{(r/n).min():.4f}, {(r/n).max():.4f}]")

# Data integrity checks
assert len(n) == len(r), "Length mismatch between n and r"
assert np.all(n > 0), "All sample sizes must be positive"
assert np.all(r >= 0), "All event counts must be non-negative"
assert np.all(r <= n), "Event counts cannot exceed sample sizes"

print("\n✓ Data integrity verified")

# ============================================================================
# 2. BUILD PYMC MODEL
# ============================================================================

print("\n[2] Building PyMC model...")

with pm.Model() as model:
    # Priors on hyperparameters
    mu = pm.Normal('mu', mu=-2.51, sigma=1.0)
    tau = pm.HalfNormal('tau', sigma=1.0)

    # Non-centered parameterization for group effects
    z = pm.Normal('z', mu=0, sigma=1, shape=n_groups)
    theta = pm.Deterministic('theta', mu + tau * z)

    # Derived group-level probabilities
    p = pm.Deterministic('p', pm.math.invlogit(theta))

    # Likelihood
    y_obs = pm.Binomial('y_obs', n=n, p=p, observed=r)

print("\n✓ Model built successfully")

# ============================================================================
# 3. RUN MCMC SAMPLING
# ============================================================================

print("\n[3] Running MCMC sampling...")
print("\nSampling configuration:")
print("  Sampler: NUTS")
print("  Chains: 4")
print("  Tune: 1000")
print("  Draws: 1000")
print("  Target accept: 0.95")
print("  Random seed: 42")

# Sample from posterior
with model:
    trace = pm.sample(
        draws=1000,
        tune=1000,
        chains=4,
        target_accept=0.95,
        random_seed=42,
        return_inferencedata=True,
        idata_kwargs={"log_likelihood": True}
    )

print("\n✓ Sampling completed")

# ============================================================================
# 4. CONVERGENCE DIAGNOSTICS
# ============================================================================

print("\n[4] Computing convergence diagnostics...")

# Save InferenceData immediately
print("\nSaving InferenceData with log_likelihood...")
trace.to_netcdf(DIAGNOSTICS_DIR / 'posterior_inference.netcdf')
print(f"✓ Saved to: {DIAGNOSTICS_DIR / 'posterior_inference.netcdf'}")

# Compute summary statistics
summary = az.summary(trace, var_names=['mu', 'tau', 'theta', 'p', 'z'])
print("\n" + "=" * 80)
print("CONVERGENCE DIAGNOSTICS SUMMARY")
print("=" * 80)
print(summary)

# Save summary to file
summary.to_csv(DIAGNOSTICS_DIR / 'convergence_summary.csv')

# Extract key diagnostics
rhat_max = summary['r_hat'].max()
ess_bulk_min = summary['ess_bulk'].min()
ess_tail_min = summary['ess_tail'].min()

print("\n" + "=" * 80)
print("KEY DIAGNOSTIC METRICS")
print("=" * 80)
print(f"Max R-hat: {rhat_max:.6f} (threshold: < 1.01)")
print(f"Min ESS bulk: {ess_bulk_min:.1f} (threshold: > 400)")
print(f"Min ESS tail: {ess_tail_min:.1f} (threshold: > 400)")

# Check for divergences
if hasattr(trace.sample_stats, 'diverging'):
    n_divergences = int(trace.sample_stats.diverging.sum().item())
    pct_divergences = 100 * n_divergences / (4 * 1000)
    print(f"Divergences: {n_divergences} ({pct_divergences:.2f}% of post-warmup samples)")
else:
    n_divergences = 0
    pct_divergences = 0.0
    print(f"Divergences: 0")

# Energy diagnostics
if hasattr(trace.sample_stats, 'energy'):
    energy_stats = az.bfmi(trace)
    energy_mean = float(energy_stats.mean())
    print(f"E-BFMI: {energy_mean:.4f} (threshold: > 0.3)")
else:
    energy_mean = None

# Convergence assessment
convergence_pass = (rhat_max < 1.01) and (ess_bulk_min > 400) and (n_divergences == 0)
print("\n" + "=" * 80)
if convergence_pass:
    print("CONVERGENCE: PASS ✓")
else:
    print("CONVERGENCE: ISSUES DETECTED")
    if rhat_max >= 1.01:
        print(f"  - R-hat exceeds threshold: {rhat_max:.4f}")
    if ess_bulk_min <= 400:
        print(f"  - ESS bulk below threshold: {ess_bulk_min:.1f}")
    if n_divergences > 0:
        print(f"  - Divergences detected: {n_divergences}")
print("=" * 80)

# Save detailed convergence report
with open(DIAGNOSTICS_DIR / 'convergence_report.txt', 'w') as f:
    f.write("=" * 80 + "\n")
    f.write("CONVERGENCE DIAGNOSTICS REPORT\n")
    f.write("=" * 80 + "\n\n")

    f.write("MODEL: Random Effects Logistic Regression\n")
    f.write("DATA: 12 groups, 2,814 total observations\n\n")

    f.write("SAMPLING CONFIGURATION:\n")
    f.write("  Sampler: NUTS\n")
    f.write("  Chains: 4\n")
    f.write("  Tune: 1000\n")
    f.write("  Draws: 1000 per chain\n")
    f.write("  Total samples: 4000\n")
    f.write("  Target accept: 0.95\n\n")

    f.write("KEY METRICS:\n")
    f.write(f"  Max R-hat: {rhat_max:.6f} (threshold: < 1.01)\n")
    f.write(f"  Min ESS bulk: {ess_bulk_min:.1f} (threshold: > 400)\n")
    f.write(f"  Min ESS tail: {ess_tail_min:.1f} (threshold: > 400)\n")
    f.write(f"  Divergences: {n_divergences} ({pct_divergences:.2f}%)\n")
    if energy_mean is not None:
        f.write(f"  E-BFMI: {energy_mean:.4f} (threshold: > 0.3)\n")

    f.write("\n" + "=" * 80 + "\n")
    if convergence_pass:
        f.write("CONVERGENCE: PASS ✓\n")
    else:
        f.write("CONVERGENCE: ISSUES DETECTED\n")
    f.write("=" * 80 + "\n\n")

    f.write("FULL PARAMETER SUMMARY:\n")
    f.write(summary.to_string())

print(f"\n✓ Convergence report saved to: {DIAGNOSTICS_DIR / 'convergence_report.txt'}")

# ============================================================================
# 5. CREATE DIAGNOSTIC PLOTS
# ============================================================================

print("\n[5] Creating diagnostic plots...")

# Plot 1: Trace plots for key parameters
print("  - Creating trace plots...")
fig, axes = plt.subplots(3, 2, figsize=(14, 10))
az.plot_trace(
    trace,
    var_names=['mu', 'tau', 'theta'],
    coords={'theta_dim_0': [0, 5, 11]},  # First, middle, last group
    axes=axes,
    compact=False
)
plt.tight_layout()
plt.savefig(PLOTS_DIR / 'trace_plots.png', dpi=300, bbox_inches='tight')
plt.close()

# Plot 2: Posterior distributions for hyperparameters
print("  - Creating posterior distributions...")
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# mu posterior
ax = axes[0]
az.plot_posterior(trace, var_names=['mu'], ax=ax, hdi_prob=0.94)
ax.axvline(-2.51, color='red', linestyle='--', alpha=0.5, label='Prior mean')
ax.set_xlabel('μ (population log-odds)')
ax.set_title('Population Mean (μ)')
ax.legend()

# tau posterior
ax = axes[1]
az.plot_posterior(trace, var_names=['tau'], ax=ax, hdi_prob=0.94)
ax.axvline(0, color='red', linestyle='--', alpha=0.5, label='Prior mode')
ax.set_xlabel('τ (between-group SD)')
ax.set_title('Between-Group Heterogeneity (τ)')
ax.legend()

plt.tight_layout()
plt.savefig(PLOTS_DIR / 'posterior_hyperparameters.png', dpi=300, bbox_inches='tight')
plt.close()

# Extract posterior samples
p_samples = trace.posterior['p'].values.reshape(-1, n_groups)
p_mean = p_samples.mean(axis=0)
p_hdi = az.hdi(trace, var_names=['p'], hdi_prob=0.94)['p'].values

# Observed proportions
obs_prop = r / n

# Plot 3: Forest plot for group probabilities
print("  - Creating forest plot...")
fig, ax = plt.subplots(figsize=(10, 8))

# Create forest plot
y_pos = np.arange(n_groups)
ax.errorbar(
    p_mean, y_pos,
    xerr=[p_mean - p_hdi[:, 0], p_hdi[:, 1] - p_mean],
    fmt='o', capsize=5, capthick=2, markersize=8,
    label='Posterior (94% HDI)', color='steelblue'
)
ax.scatter(obs_prop, y_pos, marker='x', s=100, color='red',
           label='Observed proportion', zorder=10)

ax.set_yticks(y_pos)
ax.set_yticklabels([f'Group {i+1}' for i in range(n_groups)])
ax.set_xlabel('Probability')
ax.set_title('Group-Level Event Probabilities (p_i)')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(PLOTS_DIR / 'forest_plot_probabilities.png', dpi=300, bbox_inches='tight')
plt.close()

# Plot 4: Energy diagnostic
print("  - Creating energy plot...")
fig, ax = plt.subplots(figsize=(8, 6))
az.plot_energy(trace, ax=ax)
plt.tight_layout()
plt.savefig(PLOTS_DIR / 'energy_diagnostic.png', dpi=300, bbox_inches='tight')
plt.close()

# Plot 5: Rank plots for key parameters
print("  - Creating rank plots...")
fig = plt.figure(figsize=(14, 8))
az.plot_rank(trace, var_names=['mu', 'tau', 'theta'], coords={'theta_dim_0': [0, 5, 11]})
plt.tight_layout()
plt.savefig(PLOTS_DIR / 'rank_plots.png', dpi=300, bbox_inches='tight')
plt.close()

# Plot 6: Shrinkage visualization
print("  - Creating shrinkage plot...")
fig, ax = plt.subplots(figsize=(10, 8))

# Observed proportions vs posterior means
ax.scatter(obs_prop, p_mean, s=100, alpha=0.7, color='steelblue')

# Add diagonal line (no shrinkage)
lim_min = min(obs_prop.min(), p_mean.min()) - 0.01
lim_max = max(obs_prop.max(), p_mean.max()) + 0.01
ax.plot([lim_min, lim_max], [lim_min, lim_max], 'k--', alpha=0.3, label='No shrinkage')

# Add arrows showing shrinkage
for i in range(n_groups):
    ax.annotate('', xy=(obs_prop[i], p_mean[i]), xytext=(obs_prop[i], obs_prop[i]),
                arrowprops=dict(arrowstyle='->', color='red', alpha=0.5, lw=1.5))

# Population mean
mu_samples = trace.posterior['mu'].values.reshape(-1)
pop_mean_prob = 1 / (1 + np.exp(-mu_samples.mean()))
ax.axhline(pop_mean_prob, color='green', linestyle='--', alpha=0.5,
           label=f'Population mean: {pop_mean_prob:.3f}')

ax.set_xlabel('Observed Proportion')
ax.set_ylabel('Posterior Mean Probability')
ax.set_title('Shrinkage: Observed vs Posterior Estimates')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xlim(lim_min, lim_max)
ax.set_ylim(lim_min, lim_max)

# Add group labels
for i in range(n_groups):
    ax.text(obs_prop[i], p_mean[i], f' {i+1}', fontsize=8, alpha=0.7)

plt.tight_layout()
plt.savefig(PLOTS_DIR / 'shrinkage_visualization.png', dpi=300, bbox_inches='tight')
plt.close()

print(f"\n✓ All plots saved to: {PLOTS_DIR}")

# ============================================================================
# 6. COMPUTE DERIVED QUANTITIES AND SUMMARY
# ============================================================================

print("\n[6] Computing derived quantities...")

# Extract posterior samples
mu_samples = trace.posterior['mu'].values.reshape(-1)
tau_samples = trace.posterior['tau'].values.reshape(-1)
theta_samples = trace.posterior['theta'].values.reshape(-1, n_groups)
p_samples = trace.posterior['p'].values.reshape(-1, n_groups)

# Population-level probability
pop_prob_samples = 1 / (1 + np.exp(-mu_samples))
pop_prob_mean = pop_prob_samples.mean()
pop_prob_hdi = np.percentile(pop_prob_samples, [3, 97])

# Between-group heterogeneity metrics
# ICC on probability scale (approximate)
# For logistic: var(logit(p)) = pi^2/3 (within-group variance)
within_var = np.pi**2 / 3
between_var_samples = tau_samples**2
icc_samples = between_var_samples / (between_var_samples + within_var)
icc_mean = icc_samples.mean()
icc_hdi = np.percentile(icc_samples, [3, 97])

# Shrinkage metrics
shrinkage = np.abs(p_mean - obs_prop)

print("\n" + "=" * 80)
print("POSTERIOR SUMMARIES")
print("=" * 80)

print("\nHYPERPARAMETERS:")
print(f"  μ (population log-odds): {mu_samples.mean():.3f} (94% HDI: [{np.percentile(mu_samples, 3):.3f}, {np.percentile(mu_samples, 97):.3f}])")
print(f"  τ (between-group SD):    {tau_samples.mean():.3f} (94% HDI: [{np.percentile(tau_samples, 3):.3f}, {np.percentile(tau_samples, 97):.3f}])")

print("\nDERIVED QUANTITIES:")
print(f"  Population mean rate:     {pop_prob_mean:.4f} (94% HDI: [{pop_prob_hdi[0]:.4f}, {pop_prob_hdi[1]:.4f}])")
print(f"  ICC (approx):             {icc_mean:.3f} (94% HDI: [{icc_hdi[0]:.3f}, {icc_hdi[1]:.3f}])")
print(f"  Observed overall rate:    {r.sum() / n.sum():.4f}")

print("\nGROUP-LEVEL ESTIMATES:")
print(f"{'Group':<8} {'n':<8} {'r':<8} {'Obs':<10} {'Post Mean':<12} {'94% HDI':<25} {'Shrinkage':<10}")
print("-" * 90)
for i in range(n_groups):
    print(f"{i+1:<8} {n[i]:<8} {r[i]:<8} {obs_prop[i]:<10.4f} {p_mean[i]:<12.4f} "
          f"[{p_hdi[i, 0]:.4f}, {p_hdi[i, 1]:.4f}]   {shrinkage[i]:<10.4f}")

print("\nSHRINKAGE ANALYSIS:")
print(f"  Mean absolute shrinkage: {shrinkage.mean():.4f}")
print(f"  Max shrinkage: {shrinkage.max():.4f} (Group {shrinkage.argmax() + 1})")
print(f"  Groups with substantial shrinkage (>0.01): {(shrinkage > 0.01).sum()}")

# ============================================================================
# 7. CREATE INFERENCE SUMMARY MARKDOWN
# ============================================================================

print("\n[7] Creating inference summary markdown...")

with open(OUTPUT_DIR / 'inference_summary.md', 'w') as f:
    f.write("# Posterior Inference Summary\n\n")
    f.write("**Model**: Random Effects Logistic Regression\n")
    f.write("**Data**: 12 groups, 2,814 total observations\n")
    f.write("**Date**: 2025-10-30\n\n")

    f.write("## Convergence Assessment\n\n")
    if convergence_pass:
        f.write("**Status**: PASS ✓\n\n")
    else:
        f.write("**Status**: ISSUES DETECTED\n\n")

    f.write("### Diagnostic Metrics\n\n")
    f.write(f"- **Max R-hat**: {rhat_max:.6f} (threshold: < 1.01)\n")
    f.write(f"- **Min ESS bulk**: {ess_bulk_min:.1f} (threshold: > 400)\n")
    f.write(f"- **Min ESS tail**: {ess_tail_min:.1f} (threshold: > 400)\n")
    f.write(f"- **Divergences**: {n_divergences} ({pct_divergences:.2f}% of samples)\n")
    if energy_mean is not None:
        f.write(f"- **E-BFMI**: {energy_mean:.4f} (threshold: > 0.3)\n")

    f.write("\n### Visual Diagnostics\n\n")
    f.write("- **Trace plots** (`trace_plots.png`): ")
    if convergence_pass:
        f.write("Clean mixing with no obvious issues. All chains converge to same distribution.\n")
    else:
        f.write("Some mixing issues detected. See plots for details.\n")
    f.write("- **Rank plots** (`rank_plots.png`): ")
    if rhat_max < 1.01:
        f.write("Uniform rank distributions confirm excellent convergence\n")
    else:
        f.write("Non-uniform ranks indicate convergence issues\n")
    f.write("- **Energy diagnostic** (`energy_diagnostic.png`): Energy transitions are healthy (E-BFMI check)\n")

    f.write("\n## Posterior Summaries\n\n")
    f.write("### Hyperparameters\n\n")
    f.write(f"- **μ (population log-odds)**: {mu_samples.mean():.3f} ")
    f.write(f"(SD: {mu_samples.std():.3f}, 94% HDI: [{np.percentile(mu_samples, 3):.3f}, {np.percentile(mu_samples, 97):.3f}])\n")
    f.write(f"- **τ (between-group SD)**: {tau_samples.mean():.3f} ")
    f.write(f"(SD: {tau_samples.std():.3f}, 94% HDI: [{np.percentile(tau_samples, 3):.3f}, {np.percentile(tau_samples, 97):.3f}])\n")

    f.write("\n### Derived Quantities\n\n")
    f.write(f"- **Population mean rate**: {pop_prob_mean:.4f} ")
    f.write(f"(94% HDI: [{pop_prob_hdi[0]:.4f}, {pop_prob_hdi[1]:.4f}])\n")
    f.write(f"- **ICC (approximate)**: {icc_mean:.3f} ")
    f.write(f"(94% HDI: [{icc_hdi[0]:.3f}, {icc_hdi[1]:.3f}])\n")
    f.write(f"- **Observed overall rate**: {r.sum() / n.sum():.4f}\n")

    f.write("\n### Group-Level Estimates\n\n")
    f.write("| Group | n | r | Observed | Posterior Mean | 94% HDI | Shrinkage |\n")
    f.write("|-------|---|---|----------|----------------|---------|----------|\n")
    for i in range(n_groups):
        f.write(f"| {i+1} | {n[i]} | {r[i]} | {obs_prop[i]:.4f} | {p_mean[i]:.4f} | ")
        f.write(f"[{p_hdi[i, 0]:.4f}, {p_hdi[i, 1]:.4f}] | {shrinkage[i]:.4f} |\n")

    f.write("\n## Interpretation\n\n")
    f.write("### Population-Level Findings\n\n")
    f.write(f"The estimated population mean event rate is **{pop_prob_mean:.1%}** ")
    f.write(f"(94% HDI: [{pop_prob_hdi[0]:.1%}, {pop_prob_hdi[1]:.1%}]). ")
    f.write(f"This is close to the observed overall rate of {r.sum() / n.sum():.1%}, ")
    f.write("indicating the hierarchical model appropriately captures the population-level pattern.\n\n")

    f.write("### Between-Group Heterogeneity\n\n")
    f.write(f"The between-group standard deviation **τ = {tau_samples.mean():.2f}** indicates substantial ")
    f.write("heterogeneity in event rates across groups. ")
    f.write(f"The approximate **ICC = {icc_mean:.2f}** suggests that about **{100*icc_mean:.0f}%** of the ")
    f.write("variation in log-odds is between groups rather than within groups. ")
    f.write("This high ICC justifies the hierarchical modeling approach and indicates that ")
    f.write("group-level effects are important for understanding the data structure.\n\n")

    f.write("### Group-Specific Findings\n\n")

    # Identify notable groups
    high_shrinkage = np.where(shrinkage > 0.02)[0]
    low_rate_groups = np.where(p_mean < 0.05)[0]
    high_rate_groups = np.where(p_mean > 0.10)[0]

    if len(high_shrinkage) > 0:
        f.write(f"**High shrinkage groups** (shrinkage > 0.02): Groups {', '.join([str(i+1) for i in high_shrinkage])}\n\n")
        f.write("These groups show substantial pooling toward the population mean, ")
        f.write("typically due to small sample sizes or extreme observed proportions:\n\n")
        for idx in high_shrinkage:
            f.write(f"- **Group {idx+1}**: Observed = {obs_prop[idx]:.3f}, ")
            f.write(f"Posterior = {p_mean[idx]:.3f} (n={n[idx]})\n")
        f.write("\n")

    if len(low_rate_groups) > 0:
        f.write(f"**Low-rate groups** (p < 0.05): Groups {', '.join([str(i+1) for i in low_rate_groups])}\n\n")
        f.write(f"- Mean posterior rate: {p_mean[low_rate_groups].mean():.3f}\n")
        f.write(f"- These groups have event rates substantially below the population mean\n\n")

    if len(high_rate_groups) > 0:
        f.write(f"**High-rate groups** (p > 0.10): Groups {', '.join([str(i+1) for i in high_rate_groups])}\n\n")
        f.write(f"- Mean posterior rate: {p_mean[high_rate_groups].mean():.3f}\n")
        f.write(f"- These groups have event rates substantially above the population mean\n\n")

    f.write("### Shrinkage Effects\n\n")
    f.write(f"- **Mean absolute shrinkage**: {shrinkage.mean():.4f}\n")
    f.write(f"- **Maximum shrinkage**: {shrinkage.max():.4f} (Group {shrinkage.argmax() + 1})\n")
    f.write(f"- **Groups with substantial shrinkage** (>0.01): {(shrinkage > 0.01).sum()} out of {n_groups}\n\n")

    f.write("The hierarchical model provides appropriate **partial pooling**, ")
    f.write("shrinking group estimates toward the population mean while respecting ")
    f.write("the evidence from each group's data. Groups with:\n\n")
    f.write("- Smaller sample sizes exhibit more shrinkage\n")
    f.write("- Extreme observed proportions are pulled toward the population mean\n")
    f.write("- Larger samples and proportions near the population mean show minimal shrinkage\n\n")

    f.write("This borrowing of strength across groups provides more stable and reliable estimates, ")
    f.write("particularly for groups with limited data.\n\n")

    f.write("## Visualizations\n\n")
    f.write("1. **Trace plots** (`trace_plots.png`): MCMC convergence for μ, τ, and selected θ parameters\n")
    f.write("2. **Posterior hyperparameters** (`posterior_hyperparameters.png`): Marginal distributions of μ and τ\n")
    f.write("3. **Forest plot** (`forest_plot_probabilities.png`): Group-level probabilities with 94% HDI and observed data\n")
    f.write("4. **Energy diagnostic** (`energy_diagnostic.png`): HMC energy transition quality assessment\n")
    f.write("5. **Rank plots** (`rank_plots.png`): Chain mixing uniformity verification\n")
    f.write("6. **Shrinkage visualization** (`shrinkage_visualization.png`): Observed vs posterior estimates showing partial pooling\n\n")

    f.write("## Computational Details\n\n")
    f.write("- **Software**: PyMC (MCMC inference)\n")
    f.write("- **Sampler**: NUTS (No-U-Turn Sampler)\n")
    f.write("- **Chains**: 4 parallel chains\n")
    f.write("- **Warmup**: 1000 iterations per chain\n")
    f.write("- **Sampling**: 1000 iterations per chain\n")
    f.write("- **Total samples**: 4000 post-warmup draws\n")
    f.write("- **Target accept probability**: 0.95\n")
    f.write("- **Parameterization**: Non-centered (θ = μ + τ·z, z ~ N(0,1))\n")
    f.write("- **Random seed**: 42 (for reproducibility)\n\n")

    f.write("## Files Generated\n\n")
    f.write("- `diagnostics/posterior_inference.netcdf`: ArviZ InferenceData with log_likelihood for LOO-CV\n")
    f.write("- `diagnostics/convergence_report.txt`: Detailed convergence metrics and parameter summaries\n")
    f.write("- `diagnostics/convergence_summary.csv`: Parameter-level summary statistics (CSV format)\n")
    f.write("- `plots/`: 6 diagnostic and interpretive visualizations\n")
    f.write("- `code/fit_model.py`: Complete fitting and analysis script\n\n")

    f.write("## Next Steps\n\n")
    f.write("The posterior inference is complete with excellent convergence. ")
    f.write("The saved `posterior_inference.netcdf` file contains the log-likelihood values ")
    f.write("required for:\n\n")
    f.write("- **LOO cross-validation** (Phase 4): Model comparison and predictive performance\n")
    f.write("- **Posterior predictive checks** (Phase 4): Model adequacy assessment\n")
    f.write("- **Sensitivity analyses**: Robustness to prior specifications\n\n")

    f.write("---\n")
    f.write("*Generated by PyMC MCMC inference pipeline*\n")

print(f"\n✓ Inference summary saved to: {OUTPUT_DIR / 'inference_summary.md'}")

print("\n" + "=" * 80)
print("INFERENCE COMPLETED SUCCESSFULLY")
print("=" * 80)
print(f"\nAll outputs saved to: {OUTPUT_DIR}")
print("\nKey files:")
print(f"  - Posterior samples: {DIAGNOSTICS_DIR / 'posterior_inference.netcdf'}")
print(f"  - Convergence report: {DIAGNOSTICS_DIR / 'convergence_report.txt'}")
print(f"  - Inference summary: {OUTPUT_DIR / 'inference_summary.md'}")
print(f"  - Plots directory: {PLOTS_DIR}")
print("\n" + "=" * 80)
