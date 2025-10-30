"""
Deep dive into Check 7 failure
"""
import numpy as np
from scipy.stats import norm, gamma, t as student_t, halfnorm
import matplotlib.pyplot as plt

np.random.seed(42)

N_OBS = 27
X_MIN, X_MAX = 1.0, 31.5
Y_MEAN, Y_SD = 2.33, 0.27
x_observed = np.linspace(X_MIN, X_MAX, N_OBS)

N_SAMPLES = 1000

# Sample v1 priors (sigma=0.15)
alpha_prior = norm.rvs(loc=2.0, scale=0.5, size=N_SAMPLES)
beta_prior = norm.rvs(loc=0.3, scale=0.2, size=N_SAMPLES)
c_prior = gamma.rvs(a=2, scale=1/2, size=N_SAMPLES)
nu_prior = gamma.rvs(a=2, scale=1/0.1, size=N_SAMPLES)
sigma_prior = halfnorm.rvs(loc=0, scale=0.15, size=N_SAMPLES)

# Generate predictions
y_sim = np.zeros((N_SAMPLES, N_OBS))
mu = np.zeros((N_SAMPLES, N_OBS))

for i in range(N_SAMPLES):
    mu[i, :] = alpha_prior[i] + beta_prior[i] * np.log(x_observed + c_prior[i])
    for j in range(N_OBS):
        y_sim[i, j] = student_t.rvs(df=nu_prior[i], loc=mu[i, j], scale=sigma_prior[i])

mean_y_sim = np.mean(y_sim, axis=1)

# Analyze Check 7
target_lower = Y_MEAN - 2*Y_SD
target_upper = Y_MEAN + 2*Y_SD

within_range = (mean_y_sim >= target_lower) & (mean_y_sim <= target_upper)
pct_within = np.mean(within_range) * 100

print(f"Check 7 Analysis (sigma=0.15):")
print(f"Target range: [{target_lower:.2f}, {target_upper:.2f}]")
print(f"Pass rate: {pct_within:.1f}%")
print()

# Find problem cases
outliers = ~within_range
print(f"Number of failed datasets: {np.sum(outliers)}")
print()

# Analyze failed cases
failed_means = mean_y_sim[outliers]
print(f"Failed dataset means: min={np.min(failed_means):.2f}, max={np.max(failed_means):.2f}")
print()

# Check correlation with nu and sigma
print("Correlation analysis:")
print(f"  mean_y vs nu:    {np.corrcoef(mean_y_sim, nu_prior)[0,1]:.3f}")
print(f"  mean_y vs sigma: {np.corrcoef(mean_y_sim, sigma_prior)[0,1]:.3f}")
print()

# Check what fraction have extreme nu or sigma
failed_indices = np.where(outliers)[0]
for idx in failed_indices[:5]:  # Show first 5
    print(f"Failed case {idx}: nu={nu_prior[idx]:.1f}, sigma={sigma_prior[idx]:.3f}, mean_y={mean_y_sim[idx]:.2f}")
    # Check for extreme values in this dataset
    n_extreme = np.sum((y_sim[idx, :] < 0) | (y_sim[idx, :] > 10))
    print(f"  Extreme values in dataset: {n_extreme}/27")
    print(f"  Y range: [{np.min(y_sim[idx, :]):.2f}, {np.max(y_sim[idx, :]):.2f}]")
    print()

