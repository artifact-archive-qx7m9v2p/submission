"""
Prior Predictive Check for Experiment 2: NB-AR(1) Model

This script validates that the prior specifications generate plausible
time series data with temporal correlation before fitting the model.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path

# Set random seed for reproducibility
np.random.seed(42)

# Configuration
N_SIMS = 500  # Number of prior predictive simulations
OUTPUT_DIR = Path("/workspace/experiments/experiment_2/prior_predictive_check")
PLOTS_DIR = OUTPUT_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Load data to get structure
data = pd.read_csv("/workspace/data/data.csv")
N = len(data)
year = data['year'].values
C_observed = data['C'].values

print("="*80)
print("PRIOR PREDICTIVE CHECK: NB-AR(1) Model")
print("="*80)
print(f"Data: {N} observations, C range: [{C_observed.min()}, {C_observed.max()}]")
print(f"Running {N_SIMS} prior predictive simulations\n")

# ============================================================================
# STEP 1: Sample from priors and generate AR(1) time series
# ============================================================================

print("STEP 1: Sampling from priors and generating AR(1) time series...")

# Storage
prior_samples = {
    'beta_0': np.zeros(N_SIMS),
    'beta_1': np.zeros(N_SIMS),
    'phi': np.zeros(N_SIMS),
    'rho': np.zeros(N_SIMS),
    'sigma': np.zeros(N_SIMS)
}
C_sim = np.zeros((N_SIMS, N))
eta_sim = np.zeros((N_SIMS, N))
epsilon_sim = np.zeros((N_SIMS, N))

# Prior specifications
# β₀ ~ Normal(4.69, 1.0)
# β₁ ~ Normal(1.0, 0.5)
# φ ~ Gamma(2, 0.1)
# ρ ~ Beta(20, 2)  # E[ρ] = 0.91
# σ ~ Exponential(2)

for i in range(N_SIMS):
    # Sample parameters
    beta_0 = np.random.normal(4.69, 1.0)
    beta_1 = np.random.normal(1.0, 0.5)
    phi = np.random.gamma(2, 1/0.1)
    rho = np.random.beta(20, 2)
    sigma = np.random.exponential(1/2)

    # Store parameters
    prior_samples['beta_0'][i] = beta_0
    prior_samples['beta_1'][i] = beta_1
    prior_samples['phi'][i] = phi
    prior_samples['rho'][i] = rho
    prior_samples['sigma'][i] = sigma

    # Simulate AR(1) process for epsilon
    epsilon = np.zeros(N)
    # Initialize with stationary distribution
    epsilon[0] = np.random.normal(0, sigma / np.sqrt(1 - rho**2))

    for t in range(1, N):
        epsilon[t] = rho * epsilon[t-1] + np.random.normal(0, sigma)

    epsilon_sim[i] = epsilon

    # Generate log-rate
    eta = beta_0 + beta_1 * year + epsilon
    eta_sim[i] = eta

    # Generate counts
    mu = np.exp(eta)
    # Negative binomial parameterization: n=phi, p=phi/(phi+mu)
    p = phi / (phi + mu)
    C_sim[i] = np.random.negative_binomial(phi, p)

print(f"Prior samples generated: {N_SIMS} simulations")

# ============================================================================
# STEP 2: Calculate summary statistics
# ============================================================================

print("\nSTEP 2: Calculating summary statistics...")

# Prior parameter summaries
print("\nPRIOR PARAMETER DISTRIBUTIONS:")
print("-" * 80)
for param, samples in prior_samples.items():
    print(f"{param:8s}: mean={samples.mean():8.3f}, std={samples.std():7.3f}, "
          f"range=[{samples.min():8.3f}, {samples.max():8.3f}]")

# Expected values from priors
print("\nEXPECTED VALUES (theoretical):")
print(f"  E[β₀] = 4.69")
print(f"  E[β₁] = 1.00")
print(f"  E[φ]  = 2/0.1 = 20.00")
print(f"  E[ρ]  = 20/22 = 0.909")
print(f"  E[σ]  = 1/2 = 0.500")

# Prior predictive count summaries
C_sim_flat = C_sim.flatten()
print("\nPRIOR PREDICTIVE COUNT DISTRIBUTION:")
print("-" * 80)
print(f"Mean:       {C_sim.mean():.1f}")
print(f"Std:        {C_sim.std():.1f}")
print(f"Median:     {np.median(C_sim):.1f}")
print(f"Range:      [{C_sim.min():.0f}, {C_sim.max():.0f}]")
print(f"Percentiles:")
print(f"  1%:       {np.percentile(C_sim_flat, 1):.0f}")
print(f"  5%:       {np.percentile(C_sim_flat, 5):.0f}")
print(f"  50%:      {np.percentile(C_sim_flat, 50):.0f}")
print(f"  95%:      {np.percentile(C_sim_flat, 95):.0f}")
print(f"  99%:      {np.percentile(C_sim_flat, 99):.0f}")

# Plausibility checks
extreme_low = np.sum(C_sim < 1) / C_sim.size * 100
extreme_high = np.sum(C_sim > 5000) / C_sim.size * 100
very_high = np.sum(C_sim > 10000) / C_sim.size * 100

print(f"\nPLAUSIBILITY FLAGS:")
print(f"  % counts < 1:      {extreme_low:.2f}%")
print(f"  % counts > 5000:   {extreme_high:.2f}%")
print(f"  % counts > 10000:  {very_high:.2f}%")

# Observed data coverage
print(f"\nOBSERVED DATA COVERAGE:")
print(f"  Observed range:    [{C_observed.min()}, {C_observed.max()}]")
print(f"  Prior pred range:  [{C_sim.min():.0f}, {C_sim.max():.0f}]")
coverage = np.sum((C_sim.min() <= C_observed) & (C_observed <= C_sim.max())) / len(C_observed)
print(f"  Coverage:          {coverage*100:.1f}% of observed data in prior range")

# ============================================================================
# STEP 3: Calculate autocorrelation of prior predictive series
# ============================================================================

print("\nSTEP 3: Calculating autocorrelation structure...")

def compute_acf(x, nlags=5):
    """Compute autocorrelation function"""
    acf = np.zeros(nlags + 1)
    acf[0] = 1.0
    x_centered = x - x.mean()
    c0 = np.dot(x_centered, x_centered) / len(x)
    for lag in range(1, nlags + 1):
        c_lag = np.dot(x_centered[lag:], x_centered[:-lag]) / len(x)
        acf[lag] = c_lag / c0
    return acf

# Compute ACF for epsilon (AR process) and C (counts)
acf_epsilon = np.zeros((N_SIMS, 6))
acf_counts = np.zeros((N_SIMS, 6))

for i in range(N_SIMS):
    acf_epsilon[i] = compute_acf(epsilon_sim[i], nlags=5)
    acf_counts[i] = compute_acf(C_sim[i], nlags=5)

print("\nAUTOCORRELATION ANALYSIS:")
print("-" * 80)
print("AR(1) process (epsilon) - theoretical ACF(1) should match ρ:")
print(f"  Mean ACF(1) of epsilon: {acf_epsilon[:, 1].mean():.3f} (compare to E[ρ]=0.909)")
print(f"  Std ACF(1) of epsilon:  {acf_epsilon[:, 1].std():.3f}")

print("\nCount data autocorrelation (transformed through exp and NB):")
print(f"  Mean ACF(1) of counts:  {acf_counts[:, 1].mean():.3f}")
print(f"  Std ACF(1) of counts:   {acf_counts[:, 1].std():.3f}")
print(f"  Range ACF(1):           [{acf_counts[:, 1].min():.3f}, {acf_counts[:, 1].max():.3f}]")

# From EDA
print("\nCOMPARISON TO DATA:")
print(f"  Experiment 1 residual ACF(1): 0.511 (what AR(1) must address)")
print(f"  EDA raw ACF(1):               0.971 (motivates ρ prior)")

# ============================================================================
# STEP 4: Visualization
# ============================================================================

print("\nSTEP 4: Creating visualizations...")

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100

# ============================================================================
# PLOT 1: Prior Parameter Distributions
# ============================================================================

fig, axes = plt.subplots(2, 3, figsize=(14, 9))
fig.suptitle('Prior Parameter Distributions', fontsize=16, fontweight='bold', y=0.995)

# Beta_0
ax = axes[0, 0]
ax.hist(prior_samples['beta_0'], bins=40, alpha=0.7, edgecolor='black', density=True)
x = np.linspace(prior_samples['beta_0'].min(), prior_samples['beta_0'].max(), 100)
ax.plot(x, stats.norm.pdf(x, 4.69, 1.0), 'r-', lw=2, label='Theoretical')
ax.axvline(prior_samples['beta_0'].mean(), color='blue', linestyle='--', lw=2, label=f'Sample mean={prior_samples["beta_0"].mean():.2f}')
ax.set_xlabel('β₀ (Intercept)', fontsize=11)
ax.set_ylabel('Density', fontsize=11)
ax.set_title('β₀ ~ Normal(4.69, 1.0)', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Beta_1
ax = axes[0, 1]
ax.hist(prior_samples['beta_1'], bins=40, alpha=0.7, edgecolor='black', density=True)
x = np.linspace(prior_samples['beta_1'].min(), prior_samples['beta_1'].max(), 100)
ax.plot(x, stats.norm.pdf(x, 1.0, 0.5), 'r-', lw=2, label='Theoretical')
ax.axvline(prior_samples['beta_1'].mean(), color='blue', linestyle='--', lw=2, label=f'Sample mean={prior_samples["beta_1"].mean():.2f}')
ax.set_xlabel('β₁ (Year effect)', fontsize=11)
ax.set_ylabel('Density', fontsize=11)
ax.set_title('β₁ ~ Normal(1.0, 0.5)', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Phi
ax = axes[0, 2]
ax.hist(prior_samples['phi'], bins=40, alpha=0.7, edgecolor='black', density=True)
x = np.linspace(0.1, prior_samples['phi'].max(), 100)
ax.plot(x, stats.gamma.pdf(x, 2, scale=1/0.1), 'r-', lw=2, label='Theoretical')
ax.axvline(prior_samples['phi'].mean(), color='blue', linestyle='--', lw=2, label=f'Sample mean={prior_samples["phi"].mean():.1f}')
ax.set_xlabel('φ (Dispersion)', fontsize=11)
ax.set_ylabel('Density', fontsize=11)
ax.set_title('φ ~ Gamma(2, 0.1)', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Rho - CRITICAL PARAMETER
ax = axes[1, 0]
ax.hist(prior_samples['rho'], bins=40, alpha=0.7, edgecolor='black', density=True, color='darkred')
x = np.linspace(0, 1, 100)
ax.plot(x, stats.beta.pdf(x, 20, 2), 'r-', lw=2, label='Theoretical')
ax.axvline(prior_samples['rho'].mean(), color='blue', linestyle='--', lw=2, label=f'Sample mean={prior_samples["rho"].mean():.3f}')
ax.axvline(20/22, color='green', linestyle=':', lw=2, label=f'E[ρ]=0.909')
ax.set_xlabel('ρ (AR coefficient)', fontsize=11)
ax.set_ylabel('Density', fontsize=11)
ax.set_title('ρ ~ Beta(20, 2) - STRONG PRIOR', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Sigma
ax = axes[1, 1]
ax.hist(prior_samples['sigma'], bins=40, alpha=0.7, edgecolor='black', density=True)
x = np.linspace(0.01, prior_samples['sigma'].max(), 100)
ax.plot(x, stats.expon.pdf(x, scale=1/2), 'r-', lw=2, label='Theoretical')
ax.axvline(prior_samples['sigma'].mean(), color='blue', linestyle='--', lw=2, label=f'Sample mean={prior_samples["sigma"].mean():.3f}')
ax.set_xlabel('σ (Innovation SD)', fontsize=11)
ax.set_ylabel('Density', fontsize=11)
ax.set_title('σ ~ Exponential(2)', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Remove empty subplot
axes[1, 2].axis('off')

plt.tight_layout()
plt.savefig(PLOTS_DIR / 'prior_parameter_distributions.png', dpi=300, bbox_inches='tight')
print(f"  Saved: prior_parameter_distributions.png")
plt.close()

# ============================================================================
# PLOT 2: Correlation Prior Detailed Check
# ============================================================================

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle('Temporal Correlation Prior Diagnostics', fontsize=16, fontweight='bold')

# ρ distribution with quantiles
ax = axes[0]
ax.hist(prior_samples['rho'], bins=50, alpha=0.7, edgecolor='black', color='darkred')
quantiles = [0.025, 0.25, 0.5, 0.75, 0.975]
for q in quantiles:
    val = np.quantile(prior_samples['rho'], q)
    ax.axvline(val, color='blue', linestyle='--', alpha=0.6, lw=1.5)
    ax.text(val, ax.get_ylim()[1]*0.95, f'{q:.3f}\n{val:.3f}',
            ha='center', fontsize=8, bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
ax.set_xlabel('ρ (AR coefficient)', fontsize=12)
ax.set_ylabel('Frequency', fontsize=12)
ax.set_title(f'ρ ~ Beta(20, 2)\nMean={prior_samples["rho"].mean():.3f}, SD={prior_samples["rho"].std():.3f}',
             fontsize=11, fontweight='bold')
ax.grid(True, alpha=0.3)

# ρ vs σ joint distribution
ax = axes[1]
sc = ax.scatter(prior_samples['rho'], prior_samples['sigma'],
                alpha=0.3, c=prior_samples['phi'], cmap='viridis', s=20)
ax.set_xlabel('ρ (AR coefficient)', fontsize=12)
ax.set_ylabel('σ (Innovation SD)', fontsize=12)
ax.set_title('Joint Prior: ρ vs σ\n(color = φ)', fontsize=11, fontweight='bold')
ax.grid(True, alpha=0.3)
plt.colorbar(sc, ax=ax, label='φ (dispersion)')

# Theoretical vs empirical ACF(1) of epsilon
ax = axes[2]
ax.scatter(prior_samples['rho'], acf_epsilon[:, 1], alpha=0.4, s=20)
ax.plot([0, 1], [0, 1], 'r--', lw=2, label='Perfect match')
ax.set_xlabel('ρ (parameter)', fontsize=12)
ax.set_ylabel('Empirical ACF(1) of ε', fontsize=12)
ax.set_title('AR(1) Validation:\nρ vs Realized ACF(1)', fontsize=11, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xlim([0.6, 1.0])
ax.set_ylim([0.6, 1.0])

plt.tight_layout()
plt.savefig(PLOTS_DIR / 'temporal_correlation_diagnostics.png', dpi=300, bbox_inches='tight')
print(f"  Saved: temporal_correlation_diagnostics.png")
plt.close()

# ============================================================================
# PLOT 3: Prior Predictive Time Series
# ============================================================================

fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('Prior Predictive Time Series (50 random draws)', fontsize=16, fontweight='bold', y=0.995)

# Plot subset of simulations
n_display = 50
display_idx = np.random.choice(N_SIMS, n_display, replace=False)

# Panel A: Count trajectories
ax = axes[0, 0]
for i in display_idx:
    ax.plot(year, C_sim[i], alpha=0.3, lw=1, color='steelblue')
ax.plot(year, C_observed, 'ro-', lw=2, markersize=4, label='Observed data', zorder=100)
ax.set_xlabel('Year (standardized)', fontsize=11)
ax.set_ylabel('Count', fontsize=11)
ax.set_title('A. Prior Predictive Count Trajectories', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_ylim([0, ax.get_ylim()[1]])

# Panel B: Log-scale for wide range
ax = axes[0, 1]
for i in display_idx:
    ax.plot(year, C_sim[i], alpha=0.3, lw=1, color='steelblue')
ax.plot(year, C_observed, 'ro-', lw=2, markersize=4, label='Observed data', zorder=100)
ax.set_xlabel('Year (standardized)', fontsize=11)
ax.set_ylabel('Count (log scale)', fontsize=11)
ax.set_yscale('log')
ax.set_title('B. Prior Predictive (Log Scale)', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, which='both')

# Panel C: AR(1) epsilon trajectories
ax = axes[1, 0]
for i in display_idx:
    ax.plot(year, epsilon_sim[i], alpha=0.3, lw=1, color='darkgreen')
ax.set_xlabel('Year (standardized)', fontsize=11)
ax.set_ylabel('ε (AR process)', fontsize=11)
ax.set_title('C. AR(1) Error Process Trajectories', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.axhline(0, color='red', linestyle='--', lw=1.5, alpha=0.7)

# Panel D: η (log-rate) trajectories
ax = axes[1, 1]
for i in display_idx:
    ax.plot(year, eta_sim[i], alpha=0.3, lw=1, color='purple')
ax.set_xlabel('Year (standardized)', fontsize=11)
ax.set_ylabel('η = β₀ + β₁×year + ε', fontsize=11)
ax.set_title('D. Log-Rate (η) Trajectories', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(PLOTS_DIR / 'prior_predictive_trajectories.png', dpi=300, bbox_inches='tight')
print(f"  Saved: prior_predictive_trajectories.png")
plt.close()

# ============================================================================
# PLOT 4: Autocorrelation Analysis
# ============================================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Prior Predictive Autocorrelation Structure', fontsize=16, fontweight='bold', y=0.995)

# Panel A: ACF of epsilon (AR process)
ax = axes[0, 0]
lags = np.arange(6)
acf_mean = acf_epsilon.mean(axis=0)
acf_lower = np.percentile(acf_epsilon, 2.5, axis=0)
acf_upper = np.percentile(acf_epsilon, 97.5, axis=0)
ax.bar(lags, acf_mean, alpha=0.7, color='darkgreen', edgecolor='black')
ax.fill_between(lags, acf_lower, acf_upper, alpha=0.3, color='darkgreen', label='95% interval')
ax.set_xlabel('Lag', fontsize=11)
ax.set_ylabel('ACF', fontsize=11)
ax.set_title('A. ACF of AR(1) Process (ε)\nMean across prior draws', fontsize=12, fontweight='bold')
ax.set_xticks(lags)
ax.grid(True, alpha=0.3, axis='y')
ax.axhline(0, color='black', lw=1)
ax.legend(fontsize=10)

# Panel B: ACF of counts
ax = axes[0, 1]
acf_mean_counts = acf_counts.mean(axis=0)
acf_lower_counts = np.percentile(acf_counts, 2.5, axis=0)
acf_upper_counts = np.percentile(acf_counts, 97.5, axis=0)
ax.bar(lags, acf_mean_counts, alpha=0.7, color='steelblue', edgecolor='black')
ax.fill_between(lags, acf_lower_counts, acf_upper_counts, alpha=0.3, color='steelblue', label='95% interval')
ax.set_xlabel('Lag', fontsize=11)
ax.set_ylabel('ACF', fontsize=11)
ax.set_title('B. ACF of Count Data (C)\nMean across prior draws', fontsize=12, fontweight='bold')
ax.set_xticks(lags)
ax.grid(True, alpha=0.3, axis='y')
ax.axhline(0, color='black', lw=1)
ax.legend(fontsize=10)

# Panel C: Distribution of ACF(1) for counts
ax = axes[1, 0]
ax.hist(acf_counts[:, 1], bins=50, alpha=0.7, edgecolor='black', color='steelblue')
ax.axvline(acf_counts[:, 1].mean(), color='red', linestyle='--', lw=2, label=f'Mean={acf_counts[:, 1].mean():.3f}')
ax.axvline(0.511, color='orange', linestyle=':', lw=2, label='Exp1 residual ACF(1)=0.511')
ax.axvline(0.971, color='green', linestyle=':', lw=2, label='EDA raw ACF(1)=0.971')
ax.set_xlabel('ACF(1) of count data', fontsize=11)
ax.set_ylabel('Frequency', fontsize=11)
ax.set_title('C. Distribution of Lag-1 Autocorrelation\n(Prior predictive counts)', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Panel D: Sample ACF for a few individual series
ax = axes[1, 1]
for i in range(10):
    idx = display_idx[i]
    ax.plot(lags, acf_counts[idx], 'o-', alpha=0.5, lw=1, markersize=4)
ax.plot(lags, acf_mean_counts, 'r-', lw=3, marker='s', markersize=8, label='Mean ACF')
ax.set_xlabel('Lag', fontsize=11)
ax.set_ylabel('ACF', fontsize=11)
ax.set_title('D. ACF of 10 Individual Prior Draws', fontsize=12, fontweight='bold')
ax.set_xticks(lags)
ax.grid(True, alpha=0.3)
ax.axhline(0, color='black', lw=1)
ax.legend(fontsize=10)

plt.tight_layout()
plt.savefig(PLOTS_DIR / 'prior_acf_structure.png', dpi=300, bbox_inches='tight')
print(f"  Saved: prior_acf_structure.png")
plt.close()

# ============================================================================
# PLOT 5: Prior Predictive Coverage & Plausibility
# ============================================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Prior Predictive Coverage & Plausibility Assessment', fontsize=16, fontweight='bold', y=0.995)

# Panel A: Envelope plot
ax = axes[0, 0]
percentiles = [5, 25, 50, 75, 95]
colors = plt.cm.Blues(np.linspace(0.3, 0.8, len(percentiles)//2))
for i in range(len(percentiles)//2):
    lower = np.percentile(C_sim, percentiles[i], axis=0)
    upper = np.percentile(C_sim, percentiles[-(i+1)], axis=0)
    ax.fill_between(year, lower, upper, alpha=0.3, color=colors[i],
                    label=f'{percentiles[i]}-{percentiles[-(i+1)]}%')
ax.plot(year, np.median(C_sim, axis=0), 'b-', lw=2, label='Median')
ax.plot(year, C_observed, 'ro-', lw=2, markersize=5, label='Observed', zorder=100)
ax.set_xlabel('Year (standardized)', fontsize=11)
ax.set_ylabel('Count', fontsize=11)
ax.set_title('A. Prior Predictive Envelope', fontsize=12, fontweight='bold')
ax.legend(fontsize=9, loc='upper left')
ax.grid(True, alpha=0.3)

# Panel B: Distribution of all prior predictive counts
ax = axes[0, 1]
ax.hist(C_sim_flat, bins=100, alpha=0.7, edgecolor='black', color='steelblue', log=True)
ax.axvline(C_observed.min(), color='red', linestyle='--', lw=2, label=f'Obs min={C_observed.min()}')
ax.axvline(C_observed.max(), color='red', linestyle='--', lw=2, label=f'Obs max={C_observed.max()}')
ax.axvline(5000, color='orange', linestyle=':', lw=2, label='Plausibility threshold (5000)')
ax.set_xlabel('Count value', fontsize=11)
ax.set_ylabel('Frequency (log scale)', fontsize=11)
ax.set_title('B. Distribution of All Prior Predictive Counts', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_xlim([0, min(10000, C_sim_flat.max())])

# Panel C: Growth rate distribution
ax = axes[1, 0]
# Calculate implied growth at mean year
growth_rates = np.exp(prior_samples['beta_1']) - 1
ax.hist(growth_rates * 100, bins=50, alpha=0.7, edgecolor='black', color='darkgreen')
ax.axvline(growth_rates.mean() * 100, color='red', linestyle='--', lw=2,
          label=f'Mean={growth_rates.mean()*100:.1f}%')
ax.set_xlabel('Implied annual growth rate (%)', fontsize=11)
ax.set_ylabel('Frequency', fontsize=11)
ax.set_title('C. Prior Distribution of Growth Rate\n(exp(β₁) - 1)', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# Panel D: Max counts per series
ax = axes[1, 1]
max_counts = C_sim.max(axis=1)
ax.hist(max_counts, bins=50, alpha=0.7, edgecolor='black', color='purple')
ax.axvline(max_counts.mean(), color='red', linestyle='--', lw=2,
          label=f'Mean={max_counts.mean():.0f}')
ax.axvline(C_observed.max(), color='green', linestyle=':', lw=2,
          label=f'Observed max={C_observed.max()}')
ax.axvline(5000, color='orange', linestyle=':', lw=2, label='Threshold=5000')
ax.set_xlabel('Maximum count in series', fontsize=11)
ax.set_ylabel('Frequency', fontsize=11)
ax.set_title('D. Distribution of Maximum Counts\nAcross prior predictive series', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
if max_counts.max() > 10000:
    ax.set_xlim([0, 10000])

plt.tight_layout()
plt.savefig(PLOTS_DIR / 'prior_predictive_coverage.png', dpi=300, bbox_inches='tight')
print(f"  Saved: prior_predictive_coverage.png")
plt.close()

# ============================================================================
# STEP 5: Final Assessment
# ============================================================================

print("\n" + "="*80)
print("PRIOR PREDICTIVE CHECK ASSESSMENT")
print("="*80)

# Define pass/fail criteria
checks = {}

# Check 1: Count range plausibility
checks['count_range'] = extreme_high < 5.0  # Less than 5% above 5000
print(f"\n1. COUNT RANGE PLAUSIBILITY:")
print(f"   {extreme_high:.2f}% of counts > 5000 (threshold: < 5%)")
print(f"   Status: {'PASS' if checks['count_range'] else 'FAIL'}")

# Check 2: Extreme outliers
checks['extreme_outliers'] = very_high < 1.0  # Less than 1% above 10000
print(f"\n2. EXTREME OUTLIER CHECK:")
print(f"   {very_high:.2f}% of counts > 10000 (threshold: < 1%)")
print(f"   Status: {'PASS' if checks['extreme_outliers'] else 'FAIL'}")

# Check 3: ρ concentration
rho_mean = prior_samples['rho'].mean()
rho_in_range = 0.7 <= rho_mean <= 0.95
checks['rho_concentration'] = rho_in_range
print(f"\n3. AR COEFFICIENT CONCENTRATION:")
print(f"   Mean ρ = {rho_mean:.3f} (expected range: [0.7, 0.95])")
print(f"   Status: {'PASS' if checks['rho_concentration'] else 'FAIL'}")

# Check 4: ACF structure realism
acf1_mean = acf_counts[:, 1].mean()
acf_realistic = 0.3 <= acf1_mean <= 0.99
checks['acf_realistic'] = acf_realistic
print(f"\n4. TEMPORAL CORRELATION REALISM:")
print(f"   Mean ACF(1) of counts = {acf1_mean:.3f} (expected: [0.3, 0.99])")
print(f"   Covers Exp1 residual ACF(1)=0.511: {acf_counts[:, 1].min() <= 0.511 <= acf_counts[:, 1].max()}")
print(f"   Status: {'PASS' if checks['acf_realistic'] else 'FAIL'}")

# Check 5: Growth patterns
growth_mean = (np.exp(prior_samples['beta_1']) - 1).mean() * 100
growth_plausible = -50 <= growth_mean <= 500
checks['growth_plausible'] = growth_plausible
print(f"\n5. GROWTH PATTERN PLAUSIBILITY:")
print(f"   Mean growth rate = {growth_mean:.1f}% (expected: [-50%, 500%])")
print(f"   Status: {'PASS' if checks['growth_plausible'] else 'FAIL'}")

# Check 6: Innovation scale
sigma_mean = prior_samples['sigma'].mean()
sigma_reasonable = sigma_mean < 2.0  # Should be small relative to trend
checks['sigma_scale'] = sigma_reasonable
print(f"\n6. INNOVATION SCALE:")
print(f"   Mean σ = {sigma_mean:.3f} (should be < 2.0 for stability)")
print(f"   Status: {'PASS' if checks['sigma_scale'] else 'FAIL'}")

# Check 7: AR(1) validation
rho_acf_corr = np.corrcoef(prior_samples['rho'], acf_epsilon[:, 1])[0, 1]
ar_valid = rho_acf_corr > 0.95  # Strong correlation between ρ and realized ACF
checks['ar_validation'] = ar_valid
print(f"\n7. AR(1) PROCESS VALIDATION:")
print(f"   Correlation between ρ and ACF(1) of ε = {rho_acf_corr:.4f}")
print(f"   Status: {'PASS' if checks['ar_validation'] else 'FAIL'}")

# Overall decision
all_pass = all(checks.values())
print("\n" + "="*80)
print("OVERALL DECISION:")
print("="*80)
if all_pass:
    print("STATUS: PASS")
    print("\nThe prior specifications generate plausible AR(1) time series:")
    print("  - Counts are in scientifically reasonable range")
    print("  - Temporal correlation structure is realistic")
    print("  - AR(1) process behaves as expected")
    print("  - Growth patterns with correlation are plausible")
    print("  - Innovation scale is appropriate")
    print("\nRECOMMENDATION: Proceed to model fitting")
else:
    print("STATUS: FAIL")
    print("\nIssues detected:")
    for check, passed in checks.items():
        if not passed:
            print(f"  - {check}: FAILED")
    print("\nRECOMMENDATION: Revise prior specifications before fitting")

print("\n" + "="*80)
print("Prior predictive check complete!")
print(f"Outputs saved to: {OUTPUT_DIR}")
print("="*80)
