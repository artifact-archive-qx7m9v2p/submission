"""
Overdispersion Analysis for Binomial Dataset
=============================================
This script examines whether the data shows evidence of overdispersion
compared to a standard binomial model.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats

# Setup paths
DATA_PATH = Path("/workspace/data/data.csv")

# Load data
df = pd.read_csv(DATA_PATH)

print("="*60)
print("OVERDISPERSION ANALYSIS")
print("="*60)

# Calculate pooled proportion (MLE under constant probability model)
pooled_p = df['r'].sum() / df['n'].sum()
print(f"\n1. POOLED ESTIMATE")
print("-"*60)
print(f"Pooled proportion (p_hat): {pooled_p:.6f}")
print(f"Total successes: {df['r'].sum()}")
print(f"Total trials: {df['n'].sum()}")

# Calculate observed variance in proportions
obs_var = df['proportion'].var(ddof=1)
obs_mean = df['proportion'].mean()

print(f"\n2. OBSERVED VARIATION IN PROPORTIONS")
print("-"*60)
print(f"Mean proportion: {obs_mean:.6f}")
print(f"Observed variance: {obs_var:.8f}")
print(f"Observed std: {np.sqrt(obs_var):.6f}")

# Expected variance under binomial model with constant p
# For proportions from different sample sizes, we need to account for this
# E[Var(p_i)] = p(1-p) / n_i weighted by sample sizes

print(f"\n3. EXPECTED VARIANCE UNDER BINOMIAL MODEL")
print("-"*60)
print("Under a constant probability binomial model:")
print(f"  Expected variance of proportion for sample size n: p(1-p)/n")

# Calculate expected variance for each observation
df['expected_var'] = pooled_p * (1 - pooled_p) / df['n']
df['expected_std'] = np.sqrt(df['expected_var'])

print("\nPer-observation expected variation:")
for idx, row in df.iterrows():
    print(f"  Trial {row['trial_id']:2.0f} (n={row['n']:3.0f}): "
          f"E[Var(p)] = {row['expected_var']:.6f}, "
          f"E[SD(p)] = {row['expected_std']:.6f}")

# Average expected variance (weighted by sample size is more appropriate)
avg_expected_var_simple = df['expected_var'].mean()
print(f"\nSimple average of expected variances: {avg_expected_var_simple:.8f}")

# Better approach: compare standardized residuals
df['standardized_resid'] = (df['proportion'] - pooled_p) / df['expected_std']

print(f"\n4. STANDARDIZED RESIDUALS")
print("-"*60)
print("(Observed proportion - pooled p) / expected SD")
print("\nUnder binomial model, these should have mean~0, variance~1")
for idx, row in df.iterrows():
    print(f"  Trial {row['trial_id']:2.0f}: {row['standardized_resid']:7.3f}")

print(f"\nMean of standardized residuals: {df['standardized_resid'].mean():.4f}")
print(f"Variance of standardized residuals: {df['standardized_resid'].var(ddof=1):.4f}")
print(f"Expected variance: 1.0")

# Chi-square goodness of fit test
chi_square_stat = ((df['r'] - df['n'] * pooled_p)**2 / (df['n'] * pooled_p * (1 - pooled_p))).sum()
df_chi = len(df) - 1  # degrees of freedom (n_obs - 1 for estimating p)
p_value = 1 - stats.chi2.cdf(chi_square_stat, df_chi)

print(f"\n5. CHI-SQUARE GOODNESS OF FIT TEST")
print("-"*60)
print(f"H0: Data follows binomial(n_i, p) with constant p")
print(f"Chi-square statistic: {chi_square_stat:.4f}")
print(f"Degrees of freedom: {df_chi}")
print(f"P-value: {p_value:.6f}")
if p_value < 0.05:
    print("Result: REJECT H0 (evidence of deviation from constant p model)")
else:
    print("Result: FAIL TO REJECT H0 (consistent with constant p model)")

# Dispersion parameter estimate
dispersion_param = chi_square_stat / df_chi
print(f"\nDispersion parameter (chi-square / df): {dispersion_param:.4f}")
if dispersion_param > 1:
    print(f"  -> OVERDISPERSION detected ({dispersion_param:.2f}x expected variance)")
elif dispersion_param < 1:
    print(f"  -> UNDERDISPERSION detected ({dispersion_param:.2f}x expected variance)")
else:
    print("  -> Consistent with binomial variance")

# Variance ratio test (comparing observed to expected)
print(f"\n6. VARIANCE COMPARISON")
print("-"*60)
# This is tricky with different sample sizes, but we can look at the
# variance of standardized residuals
var_ratio = df['standardized_resid'].var(ddof=1)
print(f"Variance of standardized residuals: {var_ratio:.4f}")
print(f"Expected variance: 1.0")
print(f"Ratio: {var_ratio:.4f}")

# Range of proportions
prop_range = df['proportion'].max() - df['proportion'].min()
print(f"\n7. ADDITIONAL INDICATORS")
print("-"*60)
print(f"Range of proportions: {prop_range:.4f}")
print(f"IQR of proportions: {df['proportion'].quantile(0.75) - df['proportion'].quantile(0.25):.4f}")
print(f"Coefficient of variation: {df['proportion'].std() / df['proportion'].mean():.4f}")

# Check for outliers using standardized residuals
outliers = df[np.abs(df['standardized_resid']) > 2]
print(f"\nObservations with |standardized residual| > 2:")
if len(outliers) > 0:
    for idx, row in outliers.iterrows():
        print(f"  Trial {row['trial_id']:2.0f}: z = {row['standardized_resid']:.3f}, "
              f"p = {row['proportion']:.4f}")
else:
    print("  None")

# Beta-binomial estimate of overdispersion
# If we assume Beta-Binomial, we can estimate alpha and beta via method of moments
# Var(p) = E[Var(p|theta)] + Var[E(p|theta)]
#        = E[theta(1-theta)/n] + Var(theta)
# For Beta(alpha, beta): E[theta] = alpha/(alpha+beta), Var[theta] = alpha*beta/[(alpha+beta)^2*(alpha+beta+1)]

print(f"\n8. BETA-BINOMIAL CONSIDERATION")
print("-"*60)
print(f"If data follows Beta-Binomial model:")
print(f"  - Binomial variance: p(1-p)/n")
print(f"  - Additional variance from beta distribution of p")
print(f"  - Dispersion parameter phi: Var(standardized resid) = {var_ratio:.4f}")

if var_ratio > 1.5:
    print(f"\nStrong evidence of overdispersion - Beta-Binomial may be appropriate")
elif var_ratio > 1.1:
    print(f"\nModerate evidence of overdispersion - consider Beta-Binomial")
else:
    print(f"\nLittle evidence of overdispersion - simple Binomial may suffice")

print("\n" + "="*60)
print("OVERDISPERSION ANALYSIS COMPLETE")
print("="*60)
