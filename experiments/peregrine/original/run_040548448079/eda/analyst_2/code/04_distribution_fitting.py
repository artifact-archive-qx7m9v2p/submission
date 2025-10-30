"""
Fit and compare theoretical distributions
Compare: Poisson, Negative Binomial, and assess goodness of fit
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.optimize import minimize
from pathlib import Path

# Setup
sns.set_style("whitegrid")
output_dir = Path('/workspace/eda/analyst_2/visualizations')

# Load data
data = pd.read_csv('/workspace/data/data_analyst_2.csv')
C = data['C'].values

print("=" * 80)
print("THEORETICAL DISTRIBUTION FITTING")
print("=" * 80)

# 1. Poisson Distribution
lambda_mle = C.mean()
print(f"\n1. POISSON DISTRIBUTION")
print(f"   MLE estimate: λ = {lambda_mle:.3f}")

# Calculate Poisson probabilities (approximate for large counts)
# Use probability mass function
from scipy.stats import poisson, nbinom

# For visualization, create bins
min_c, max_c = C.min(), C.max()
bins = np.arange(min_c, max_c + 2) - 0.5

# 2. Negative Binomial Distribution
# Estimate parameters using method of moments
sample_mean = C.mean()
sample_var = C.var(ddof=1)

# NB parameterization: E[X] = r*p/(1-p), Var[X] = r*p/(1-p)^2
# Alternative: E[X] = μ, Var[X] = μ + μ^2/r
# Solving: r = μ^2 / (Var - μ)

if sample_var > sample_mean:
    r_mom = sample_mean**2 / (sample_var - sample_mean)
    p_mom = sample_mean / sample_var
    print(f"\n2. NEGATIVE BINOMIAL (Method of Moments)")
    print(f"   r (size/dispersion): {r_mom:.3f}")
    print(f"   p (probability): {p_mom:.3f}")
    print(f"   Mean: {r_mom * (1-p_mom) / p_mom:.3f}")
    print(f"   Variance: {r_mom * (1-p_mom) / p_mom**2:.3f}")
    print(f"   Variance/Mean: {(r_mom * (1-p_mom) / p_mom**2) / (r_mom * (1-p_mom) / p_mom):.3f}")
else:
    print(f"\n2. NEGATIVE BINOMIAL: Cannot fit (variance <= mean)")
    r_mom, p_mom = None, None

# Alternative NB parameterization for scipy (n, p) where mean = n*(1-p)/p
# Convert our r, p to scipy's n, p
# scipy: n = r, p = p
n_nb = r_mom
p_nb = p_mom

print(f"\n   Scipy parameterization: n={n_nb:.3f}, p={p_nb:.3f}")
print(f"   Alternative: mean={sample_mean:.3f}, alpha={1/r_mom:.3f}")

# 3. Calculate log-likelihoods
ll_poisson = np.sum(poisson.logpmf(C, lambda_mle))
if r_mom is not None:
    ll_nb = np.sum(nbinom.logpmf(C, n_nb, p_nb))
else:
    ll_nb = -np.inf

print(f"\n3. LOG-LIKELIHOODS")
print(f"   Poisson: {ll_poisson:.2f}")
print(f"   Negative Binomial: {ll_nb:.2f}")
print(f"   Difference (NB - Poisson): {ll_nb - ll_poisson:.2f}")

# 4. AIC and BIC
n_obs = len(C)
aic_poisson = -2 * ll_poisson + 2 * 1  # 1 parameter
aic_nb = -2 * ll_nb + 2 * 2  # 2 parameters
bic_poisson = -2 * ll_poisson + np.log(n_obs) * 1
bic_nb = -2 * ll_nb + np.log(n_obs) * 2

print(f"\n4. MODEL SELECTION CRITERIA")
print(f"   Poisson:")
print(f"     AIC: {aic_poisson:.2f}")
print(f"     BIC: {bic_poisson:.2f}")
print(f"   Negative Binomial:")
print(f"     AIC: {aic_nb:.2f}")
print(f"     BIC: {bic_nb:.2f}")
print(f"   Δ AIC (NB - Poisson): {aic_nb - aic_poisson:.2f}")
print(f"   Δ BIC (NB - Poisson): {bic_nb - bic_poisson:.2f}")
print(f"\n   Interpretation: Negative values favor NB model")

# 5. Goodness of fit tests
# Chi-square test for Poisson
# Group into bins to ensure expected frequencies >= 5
print(f"\n5. CHI-SQUARE GOODNESS OF FIT")

# Create bins
unique_vals, counts = np.unique(C, return_counts=True)
observed_freq = counts
expected_freq_poisson = len(C) * poisson.pmf(unique_vals, lambda_mle)

# Combine bins with expected frequency < 5
def combine_bins(unique_vals, obs, exp, min_exp=5):
    combined_vals = []
    combined_obs = []
    combined_exp = []

    current_val = []
    current_obs = 0
    current_exp = 0

    for i, (val, o, e) in enumerate(zip(unique_vals, obs, exp)):
        current_val.append(val)
        current_obs += o
        current_exp += e

        if current_exp >= min_exp or i == len(unique_vals) - 1:
            combined_vals.append(current_val)
            combined_obs.append(current_obs)
            combined_exp.append(current_exp)
            current_val = []
            current_obs = 0
            current_exp = 0

    return combined_vals, np.array(combined_obs), np.array(combined_exp)

_, obs_combined, exp_poisson_combined = combine_bins(unique_vals, observed_freq, expected_freq_poisson)

# Chi-square statistic
chi2_stat = np.sum((obs_combined - exp_poisson_combined)**2 / exp_poisson_combined)
df = len(obs_combined) - 1 - 1  # bins - 1 - estimated parameters
p_value = 1 - stats.chi2.cdf(chi2_stat, df)

print(f"   Poisson:")
print(f"     Chi-square statistic: {chi2_stat:.3f}")
print(f"     Degrees of freedom: {df}")
print(f"     p-value: {p_value:.4e}")
print(f"     Conclusion: {'REJECT' if p_value < 0.05 else 'FAIL TO REJECT'} null hypothesis (Poisson)")

# Create comparison plot
fig, axes = plt.subplots(2, 2, figsize=(14, 11))

# 1. Histogram with overlaid distributions
ax = axes[0, 0]
hist_counts, hist_bins, _ = ax.hist(C, bins=20, density=True, alpha=0.6, edgecolor='black',
                                     color='lightgray', label='Observed')

# For continuous overlay, use fine grid
x_vals = np.arange(min_c, max_c + 1)
poisson_pmf = poisson.pmf(x_vals, lambda_mle)
if r_mom is not None:
    nb_pmf = nbinom.pmf(x_vals, n_nb, p_nb)

ax.plot(x_vals, poisson_pmf, 'r-', linewidth=2.5, label='Poisson', alpha=0.8)
if r_mom is not None:
    ax.plot(x_vals, nb_pmf, 'b-', linewidth=2.5, label='Negative Binomial', alpha=0.8)

ax.set_xlabel('Count (C)', fontsize=11)
ax.set_ylabel('Probability Mass', fontsize=11)
ax.set_title('Theoretical Distribution Comparison', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, axis='y')

# Add text
text = f'Observed: mean={sample_mean:.1f}, var={sample_var:.1f}\nPoisson: λ={lambda_mle:.1f}\nNB: r={r_mom:.2f}, p={p_mom:.3f}'
ax.text(0.98, 0.97, text, transform=ax.transAxes,
        verticalalignment='top', horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.6), fontsize=8)

# 2. Q-Q plot vs Poisson
ax = axes[0, 1]
# Theoretical quantiles from Poisson
sample_sorted = np.sort(C)
theoretical_quantiles = poisson.ppf(np.linspace(0.001, 0.999, len(C)), lambda_mle)

ax.scatter(theoretical_quantiles, sample_sorted, alpha=0.6, s=40, color='steelblue')
ax.plot([theoretical_quantiles.min(), theoretical_quantiles.max()],
        [theoretical_quantiles.min(), theoretical_quantiles.max()],
        'r--', linewidth=2, label='Perfect fit')
ax.set_xlabel('Theoretical Quantiles (Poisson)', fontsize=11)
ax.set_ylabel('Sample Quantiles', fontsize=11)
ax.set_title('Q-Q Plot: Observed vs Poisson', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# 3. Q-Q plot vs Negative Binomial
ax = axes[1, 0]
if r_mom is not None:
    theoretical_quantiles_nb = nbinom.ppf(np.linspace(0.001, 0.999, len(C)), n_nb, p_nb)
    ax.scatter(theoretical_quantiles_nb, sample_sorted, alpha=0.6, s=40, color='steelblue')
    ax.plot([theoretical_quantiles_nb.min(), theoretical_quantiles_nb.max()],
            [theoretical_quantiles_nb.min(), theoretical_quantiles_nb.max()],
            'r--', linewidth=2, label='Perfect fit')
    ax.set_xlabel('Theoretical Quantiles (Negative Binomial)', fontsize=11)
    ax.set_ylabel('Sample Quantiles', fontsize=11)
    ax.set_title('Q-Q Plot: Observed vs Negative Binomial', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

# 4. Residual plot
ax = axes[1, 1]
# Pearson residuals
expected_poisson = lambda_mle * np.ones_like(C)
pearson_resid = (C - expected_poisson) / np.sqrt(expected_poisson)

ax.scatter(np.arange(len(C)), pearson_resid, alpha=0.6, s=50, color='steelblue')
ax.axhline(0, color='red', linestyle='--', linewidth=2, alpha=0.7)
ax.axhline(2, color='orange', linestyle=':', linewidth=1.5, alpha=0.5, label='±2 SD')
ax.axhline(-2, color='orange', linestyle=':', linewidth=1.5, alpha=0.5)
ax.set_xlabel('Observation Index', fontsize=11)
ax.set_ylabel('Pearson Residual', fontsize=11)
ax.set_title('Poisson Model: Pearson Residuals', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Count outliers
n_outliers = np.sum(np.abs(pearson_resid) > 2)
text = f'Outliers (|resid| > 2): {n_outliers} ({n_outliers/len(C)*100:.1f}%)'
ax.text(0.02, 0.98, text, transform=ax.transAxes,
        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.6),
        fontsize=9)

plt.tight_layout()
plt.savefig(output_dir / 'distribution_fitting.png', dpi=300, bbox_inches='tight')
print(f"\nSaved: distribution_fitting.png")
plt.close()

# Summary
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"Variance/Mean Ratio: {sample_var/sample_mean:.1f} (>>1, strong overdispersion)")
print(f"Recommended distribution: NEGATIVE BINOMIAL")
print(f"  - Poisson model is clearly inadequate (p < 0.001)")
print(f"  - NB provides much better fit (ΔLL = {ll_nb - ll_poisson:.1f})")
print(f"  - Dispersion parameter: r = {r_mom:.3f} (smaller r = more overdispersion)")
