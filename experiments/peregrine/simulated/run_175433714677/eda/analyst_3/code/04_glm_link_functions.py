"""
GLM Link Function Analysis for Count Data
Focus: Poisson and Negative Binomial GLM structures
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.optimize import minimize

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100

# Load data
with open('/workspace/data/data_analyst_3.json', 'r') as f:
    data = json.load(f)

df = pd.DataFrame({'year': data['year'], 'C': data['C']})
X = df['year'].values
y = df['C'].values
n = len(y)

print("="*70)
print("GLM LINK FUNCTION ANALYSIS FOR COUNT DATA")
print("="*70)

# Prepare design matrix
X_design = np.column_stack([np.ones(n), X])

# 1. POISSON GLM with Log Link
print("\n1. POISSON GLM: log(λ) = β₀ + β₁*year")
print("   E[C] = exp(β₀ + β₁*year)")

# Poisson log-likelihood
def poisson_loglik(params, X, y):
    linear_pred = X @ params
    lambda_pred = np.exp(linear_pred)
    # Log-likelihood: sum(y*log(λ) - λ - log(y!))
    # We can drop the log(y!) term for optimization
    loglik = np.sum(y * np.log(lambda_pred + 1e-10) - lambda_pred)
    return -loglik  # Return negative for minimization

# Fit Poisson GLM
result_poisson = minimize(poisson_loglik, x0=[4, 0.8], args=(X_design, y), method='BFGS')
beta_poisson = result_poisson.x
lambda_poisson = np.exp(X_design @ beta_poisson)

print(f"   β₀ (intercept): {beta_poisson[0]:.4f}")
print(f"   β₁ (slope): {beta_poisson[1]:.4f}")
print(f"   Implied growth rate: {(np.exp(beta_poisson[1]) - 1) * 100:.2f}% per unit year")

# Calculate deviance residuals for Poisson
sign_resid = np.sign(y - lambda_poisson)
deviance_resid_poisson = sign_resid * np.sqrt(2 * (y * np.log((y + 1e-10) / lambda_poisson) - (y - lambda_poisson)))

print(f"   Mean predicted count: {np.mean(lambda_poisson):.2f}")
print(f"   Deviance: {np.sum(deviance_resid_poisson**2):.2f}")
print(f"   AIC: {2 * len(beta_poisson) + 2 * result_poisson.fun:.2f}")

# Pearson residuals
pearson_resid_poisson = (y - lambda_poisson) / np.sqrt(lambda_poisson)
pearson_chi2 = np.sum(pearson_resid_poisson**2)
print(f"   Pearson χ²: {pearson_chi2:.2f}")
print(f"   Dispersion parameter (χ²/(n-p)): {pearson_chi2 / (n - len(beta_poisson)):.2f}")

if pearson_chi2 / (n - len(beta_poisson)) > 1.5:
    print("   -> OVERDISPERSION detected! Consider Negative Binomial")

# 2. NEGATIVE BINOMIAL approximation
print("\n2. NEGATIVE BINOMIAL GLM (Quasi-Poisson approach)")
print("   log(μ) = β₀ + β₁*year, Var(Y) = μ + μ²/θ")

# Use the same coefficients but adjust for overdispersion
lambda_nb = lambda_poisson
dispersion = pearson_chi2 / (n - len(beta_poisson))

print(f"   Using same β coefficients as Poisson")
print(f"   β₀: {beta_poisson[0]:.4f}, β₁: {beta_poisson[1]:.4f}")
print(f"   Overdispersion parameter: {dispersion:.2f}")

# Adjusted standard errors
pearson_resid_nb = (y - lambda_nb) / np.sqrt(lambda_nb * dispersion)

print(f"   Mean predicted count: {np.mean(lambda_nb):.2f}")
print(f"   Adjusted Pearson χ²: {np.sum(pearson_resid_nb**2):.2f}")

# 3. Alternative link functions
print("\n3. ALTERNATIVE LINK FUNCTIONS")

# Identity link (not recommended for count data, but for comparison)
slope_id, intercept_id, _, _, _ = stats.linregress(X, y)
y_pred_identity = intercept_id + slope_id * X
print(f"\n   Identity link: E[C] = β₀ + β₁*year")
print(f"   β₀: {intercept_id:.4f}, β₁: {slope_id:.4f}")
print(f"   Problem: Can produce negative predictions!")
print(f"   Min prediction: {np.min(y_pred_identity):.2f}")

# Square root link (variance stabilizing for Poisson)
sqrt_y = np.sqrt(y)
slope_sqrt, intercept_sqrt, _, _, _ = stats.linregress(X, sqrt_y)
y_pred_sqrt_link = (intercept_sqrt + slope_sqrt * X) ** 2
print(f"\n   Square-root link: √E[C] = β₀ + β₁*year")
print(f"   β₀: {intercept_sqrt:.4f}, β₁: {slope_sqrt:.4f}")

# Visualize different link functions
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Sort for plotting
sort_idx = np.argsort(X)
X_sorted = X[sort_idx]
y_sorted = y[sort_idx]

# A. Log link (Poisson)
axes[0, 0].plot(X_sorted, y_sorted, 'o', alpha=0.6, label='Data', markersize=6)
axes[0, 0].plot(X_sorted, lambda_poisson[sort_idx], 'r-', linewidth=2, label='Poisson (log link)')
axes[0, 0].set_xlabel('Year (standardized)')
axes[0, 0].set_ylabel('Count (C)')
axes[0, 0].set_title('A. Poisson GLM with Log Link')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# B. All link functions compared
axes[0, 1].plot(X_sorted, y_sorted, 'o', alpha=0.6, label='Data', markersize=5)
axes[0, 1].plot(X_sorted, lambda_poisson[sort_idx], '-', linewidth=2, label='Log link')
axes[0, 1].plot(X_sorted, y_pred_identity[sort_idx], '-', linewidth=2, label='Identity link')
axes[0, 1].plot(X_sorted, y_pred_sqrt_link[sort_idx], '-', linewidth=2, label='Sqrt link')
axes[0, 1].set_xlabel('Year (standardized)')
axes[0, 1].set_ylabel('Count (C)')
axes[0, 1].set_title('B. Comparison of Link Functions')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# C. Deviance residuals (Poisson)
axes[1, 0].scatter(lambda_poisson, deviance_resid_poisson, alpha=0.6)
axes[1, 0].axhline(0, color='red', linestyle='--', linewidth=1)
axes[1, 0].axhline(2, color='orange', linestyle=':', linewidth=1, label='±2 SD')
axes[1, 0].axhline(-2, color='orange', linestyle=':', linewidth=1)
axes[1, 0].set_xlabel('Fitted Values (λ)')
axes[1, 0].set_ylabel('Deviance Residuals')
axes[1, 0].set_title('C. Poisson GLM: Deviance Residuals')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# D. Pearson residuals vs fitted (checking for overdispersion pattern)
axes[1, 1].scatter(lambda_poisson, pearson_resid_poisson, alpha=0.6, label='Poisson')
axes[1, 1].axhline(0, color='red', linestyle='--', linewidth=1)
axes[1, 1].axhline(2, color='orange', linestyle=':', linewidth=1)
axes[1, 1].axhline(-2, color='orange', linestyle=':', linewidth=1)
axes[1, 1].set_xlabel('Fitted Values (λ)')
axes[1, 1].set_ylabel('Pearson Residuals')
axes[1, 1].set_title('D. Pearson Residuals (overdispersion check)')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/workspace/eda/analyst_3/visualizations/06_glm_link_functions.png', dpi=150, bbox_inches='tight')
print("\n[SAVED] GLM link function analysis: visualizations/06_glm_link_functions.png")
plt.close()

# Variance-mean relationship plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Group data into bins to estimate variance-mean relationship
n_bins = 8
bins = np.linspace(X.min(), X.max(), n_bins + 1)
bin_means = []
bin_vars = []
bin_centers = []

for i in range(n_bins):
    mask = (X >= bins[i]) & (X < bins[i+1])
    if np.sum(mask) > 1:
        bin_means.append(np.mean(y[mask]))
        bin_vars.append(np.var(y[mask]))
        bin_centers.append(np.mean(X[mask]))

bin_means = np.array(bin_means)
bin_vars = np.array(bin_vars)

# A. Variance vs Mean (checking Poisson assumption)
axes[0].scatter(bin_means, bin_vars, s=100, alpha=0.7)
axes[0].plot([0, bin_means.max()], [0, bin_means.max()], 'r--', label='Var = Mean (Poisson)', linewidth=2)

# Fit quadratic relationship (NB: Var = μ + αμ²)
if len(bin_means) > 2:
    fit_coeffs = np.polyfit(bin_means, bin_vars, 2)
    x_fit = np.linspace(0, bin_means.max(), 100)
    y_fit = np.polyval(fit_coeffs, x_fit)
    axes[0].plot(x_fit, y_fit, 'g-', label=f'Quadratic fit', linewidth=2)

axes[0].set_xlabel('Mean Count')
axes[0].set_ylabel('Variance')
axes[0].set_title('A. Variance-Mean Relationship')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# B. Log-Log plot (for power law relationship)
axes[1].scatter(np.log(bin_means), np.log(bin_vars), s=100, alpha=0.7)
axes[1].plot(np.log(bin_means), np.log(bin_means), 'r--', label='Var = Mean (slope=1)', linewidth=2)

# Fit line to estimate power law
if len(bin_means) > 2:
    slope_varmean, intercept_varmean = np.polyfit(np.log(bin_means), np.log(bin_vars), 1)
    x_log = np.linspace(np.log(bin_means).min(), np.log(bin_means).max(), 100)
    y_log = intercept_varmean + slope_varmean * x_log
    axes[1].plot(x_log, y_log, 'g-', label=f'Fitted (slope={slope_varmean:.2f})', linewidth=2)

axes[1].set_xlabel('log(Mean Count)')
axes[1].set_ylabel('log(Variance)')
axes[1].set_title('B. Log-Log Variance-Mean Relationship')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/workspace/eda/analyst_3/visualizations/07_variance_mean_relationship.png', dpi=150, bbox_inches='tight')
print("[SAVED] Variance-mean relationship: visualizations/07_variance_mean_relationship.png")
plt.close()

print("\n" + "="*70)
print("GLM RECOMMENDATIONS FOR BAYESIAN MODELING")
print("="*70)
print("""
1. PREFERRED MODEL: Negative Binomial GLM with Log Link
   - log(μ) = β₀ + β₁*year
   - Var(C) = μ + μ²/φ (allows overdispersion)
   - Handles the observed overdispersion (ratio = 70.43)

   Stan/PyMC specification:
   C[i] ~ NegativeBinomial(μ[i], φ)
   log(μ[i]) = β₀ + β₁*year[i]
   β₀ ~ Normal(4, 2)
   β₁ ~ Normal(0.8, 0.5)
   φ ~ Exponential(0.1)

2. ALTERNATIVE: Poisson GLM with Random Effects
   - log(μ) = β₀ + β₁*year + ε[i]
   - ε[i] ~ Normal(0, σ²) (individual-level random effect)
   - Accounts for extra-Poisson variation

3. NOT RECOMMENDED: Standard Poisson GLM
   - Assumes Var = Mean (violated severely)
   - Will underestimate uncertainty
   - Standard errors will be too small

4. COEFFICIENT INTERPRETATION:
   - β₁ = 0.862 means 137% increase per unit change in year
   - Standardized year ranges from -1.67 to 1.67
   - Total range implies exp(0.862 * 3.34) ≈ 17x increase
""")

# Save model comparison for Bayesian priors
print("\n" + "="*70)
print("PRIOR RECOMMENDATIONS (based on data exploration)")
print("="*70)
print(f"""
Based on observed data:
- Intercept (log scale): around {beta_poisson[0]:.2f} (exp = {np.exp(beta_poisson[0]):.1f} counts)
- Slope (log scale): around {beta_poisson[1]:.2f}
- Overdispersion: substantial (φ likely between 5-15)

Weakly informative priors:
- β₀ ~ Normal(4.3, 1)      # Allows baseline counts of 20-200
- β₁ ~ Normal(0.9, 0.3)    # Allows growth rates of 50-200%
- φ ~ Gamma(2, 0.1)        # Weakly informative for overdispersion
or
- φ ~ Exponential(0.05)    # Alternative for φ

These priors are weakly informative, allowing the data to dominate
while providing regularization and numerical stability.
""")
