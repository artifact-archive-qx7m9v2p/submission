"""
Hypothesis Testing and Distribution Fitting
Focus: Testing competing hypotheses about the count distribution
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import poisson, nbinom, chi2, kstest
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

# Load data
with open('/workspace/data/data_analyst_2.json', 'r') as f:
    data = json.load(f)

df = pd.DataFrame({'C': data['C'], 'year': data['year']})
df['time_index'] = np.arange(len(df))

print("=" * 80)
print("HYPOTHESIS TESTING: COUNT DISTRIBUTION")
print("=" * 80)

# ============================================================================
# HYPOTHESIS 1: Data follows a Poisson distribution
# ============================================================================
print("\n" + "=" * 80)
print("HYPOTHESIS 1: Poisson Distribution")
print("=" * 80)

mean_C = df['C'].mean()
var_C = df['C'].var(ddof=1)

print(f"\nPoisson assumption: Mean = Variance")
print(f"Observed mean: {mean_C:.2f}")
print(f"Observed variance: {var_C:.2f}")
print(f"Variance/Mean ratio: {var_C/mean_C:.2f}")

# Goodness-of-fit test using Chi-square
observed_counts = df['C'].values
expected_counts = np.array([poisson.pmf(c, mean_C) * len(df) for c in observed_counts])

# Calculate chi-square statistic
chi_sq = np.sum((observed_counts - expected_counts)**2 / expected_counts)
df_chi = len(df) - 1 - 1  # n - k - 1, where k=1 (one parameter estimated)
p_value_chi = 1 - chi2.cdf(chi_sq, df_chi)

print(f"\nChi-square goodness-of-fit test:")
print(f"  Chi-square statistic: {chi_sq:.2f}")
print(f"  Degrees of freedom: {df_chi}")
print(f"  p-value: {p_value_chi:.6f}")
print(f"  Conclusion: {'REJECT Poisson' if p_value_chi < 0.05 else 'Cannot reject Poisson'}")

# Kolmogorov-Smirnov test
# For discrete distribution, we use the CDF
poisson_cdf = lambda x: poisson.cdf(x, mean_C)
ks_stat, ks_pvalue = kstest(df['C'], poisson_cdf)
print(f"\nKolmogorov-Smirnov test:")
print(f"  KS statistic: {ks_stat:.4f}")
print(f"  p-value: {ks_pvalue:.6f}")
print(f"  Conclusion: {'REJECT Poisson' if ks_pvalue < 0.05 else 'Cannot reject Poisson'}")

# Log-likelihood for Poisson
ll_poisson = np.sum(poisson.logpmf(df['C'], mean_C))
aic_poisson = 2 * 1 - 2 * ll_poisson  # 1 parameter (lambda)
bic_poisson = np.log(len(df)) * 1 - 2 * ll_poisson

print(f"\nPoisson model fit:")
print(f"  Log-likelihood: {ll_poisson:.2f}")
print(f"  AIC: {aic_poisson:.2f}")
print(f"  BIC: {bic_poisson:.2f}")

# ============================================================================
# HYPOTHESIS 2: Data follows a Negative Binomial distribution
# ============================================================================
print("\n" + "=" * 80)
print("HYPOTHESIS 2: Negative Binomial Distribution")
print("=" * 80)

# Method of moments estimation
# E[X] = r(1-p)/p = mu
# Var[X] = r(1-p)/p^2 = mu/p = mu + mu^2/r
# Solving: r = mu^2 / (var - mu)

r_mom = mean_C**2 / (var_C - mean_C) if var_C > mean_C else 1
p_mom = r_mom / (r_mom + mean_C)

print(f"\nMethod of Moments estimation:")
print(f"  r (dispersion parameter): {r_mom:.4f}")
print(f"  p (probability parameter): {p_mom:.4f}")
print(f"  Implied mean: {r_mom * (1-p_mom) / p_mom:.2f}")
print(f"  Implied variance: {r_mom * (1-p_mom) / p_mom**2:.2f}")

# Maximum Likelihood Estimation
def negbinom_negloglik(params, data):
    r, p = params
    if r <= 0 or p <= 0 or p >= 1:
        return np.inf
    try:
        ll = np.sum(nbinom.logpmf(data, r, p))
        return -ll
    except:
        return np.inf

# Use MoM as initial values
initial_params = [r_mom, p_mom]
result = minimize(negbinom_negloglik, initial_params, args=(df['C'].values,),
                  method='Nelder-Mead', options={'maxiter': 10000})

if result.success:
    r_mle, p_mle = result.x
    print(f"\nMaximum Likelihood estimation:")
    print(f"  r (dispersion parameter): {r_mle:.4f}")
    print(f"  p (probability parameter): {p_mle:.4f}")
    print(f"  Implied mean: {r_mle * (1-p_mle) / p_mle:.2f}")
    print(f"  Implied variance: {r_mle * (1-p_mle) / p_mle**2:.2f}")
else:
    print("\nMLE optimization did not converge, using MoM estimates")
    r_mle, p_mle = r_mom, p_mom

# Log-likelihood for Negative Binomial
ll_nb = np.sum(nbinom.logpmf(df['C'], r_mle, p_mle))
aic_nb = 2 * 2 - 2 * ll_nb  # 2 parameters (r, p)
bic_nb = np.log(len(df)) * 2 - 2 * ll_nb

print(f"\nNegative Binomial model fit:")
print(f"  Log-likelihood: {ll_nb:.2f}")
print(f"  AIC: {aic_nb:.2f}")
print(f"  BIC: {bic_nb:.2f}")

# KS test for Negative Binomial
nb_cdf = lambda x: nbinom.cdf(x, r_mle, p_mle)
ks_stat_nb, ks_pvalue_nb = kstest(df['C'], nb_cdf)
print(f"\nKolmogorov-Smirnov test:")
print(f"  KS statistic: {ks_stat_nb:.4f}")
print(f"  p-value: {ks_pvalue_nb:.6f}")
print(f"  Conclusion: {'REJECT NegBinom' if ks_pvalue_nb < 0.05 else 'Cannot reject NegBinom'}")

# ============================================================================
# MODEL COMPARISON
# ============================================================================
print("\n" + "=" * 80)
print("MODEL COMPARISON")
print("=" * 80)

print(f"\n{'Model':<20} {'Log-Lik':<12} {'AIC':<12} {'BIC':<12} {'Parameters'}")
print("-" * 80)
print(f"{'Poisson':<20} {ll_poisson:>11.2f} {aic_poisson:>11.2f} {bic_poisson:>11.2f} {1:>11}")
print(f"{'Negative Binomial':<20} {ll_nb:>11.2f} {aic_nb:>11.2f} {bic_nb:>11.2f} {2:>11}")

print(f"\nDelta AIC (Poisson - NegBinom): {aic_poisson - aic_nb:.2f}")
print(f"Delta BIC (Poisson - NegBinom): {bic_poisson - bic_nb:.2f}")

if aic_nb < aic_poisson:
    delta_aic = aic_poisson - aic_nb
    evidence_strength = "Very strong" if delta_aic > 10 else "Strong" if delta_aic > 4 else "Moderate"
    print(f"\n{evidence_strength} evidence for Negative Binomial over Poisson (Delta AIC = {delta_aic:.2f})")
else:
    print("\nPoisson is preferred by AIC")

# Likelihood ratio test (Poisson is nested in NB when r -> infinity)
lr_statistic = 2 * (ll_nb - ll_poisson)
lr_pvalue = 1 - chi2.cdf(lr_statistic, df=1)  # 1 extra parameter in NB
print(f"\nLikelihood Ratio Test (NB vs Poisson):")
print(f"  LR statistic: {lr_statistic:.2f}")
print(f"  p-value: {lr_pvalue:.6f}")
print(f"  Conclusion: {'Negative Binomial significantly better' if lr_pvalue < 0.05 else 'No significant difference'}")

# ============================================================================
# HYPOTHESIS 3: Overdispersion is constant vs time-varying
# ============================================================================
print("\n" + "=" * 80)
print("HYPOTHESIS 3: Time-Varying vs Constant Dispersion")
print("=" * 80)

# Split data into quartiles and test for homogeneity
n_groups = 4
group_size = len(df) // n_groups
groups = []

print(f"\nAnalysis by time quartile:")
print(f"{'Quartile':<12} {'Mean':<12} {'Variance':<12} {'Var/Mean':<12} {'CV':<12}")
print("-" * 80)

for i in range(n_groups):
    start = i * group_size
    end = start + group_size if i < n_groups - 1 else len(df)
    group_data = df['C'].iloc[start:end]
    groups.append(group_data)

    mean_g = group_data.mean()
    var_g = group_data.var(ddof=1)
    cv_g = group_data.std(ddof=1) / mean_g

    print(f"{'Q' + str(i+1):<12} {mean_g:>11.2f} {var_g:>11.2f} {var_g/mean_g:>11.2f} {cv_g:>11.2f}")

# Levene's test for equality of variances
levene_stat, levene_pvalue = stats.levene(*groups)
print(f"\nLevene's test for equal variances:")
print(f"  Test statistic: {levene_stat:.4f}")
print(f"  p-value: {levene_pvalue:.6f}")
print(f"  Conclusion: {'Variances are NOT equal across time' if levene_pvalue < 0.05 else 'Cannot reject equal variances'}")

# Brown-Forsythe test (more robust to non-normality)
bf_stat, bf_pvalue = stats.levene(*groups, center='median')
print(f"\nBrown-Forsythe test for equal variances:")
print(f"  Test statistic: {bf_stat:.4f}")
print(f"  p-value: {bf_pvalue:.6f}")
print(f"  Conclusion: {'Variances are NOT equal across time' if bf_pvalue < 0.05 else 'Cannot reject equal variances'}")

# ============================================================================
# HYPOTHESIS 4: Zero-inflation
# ============================================================================
print("\n" + "=" * 80)
print("HYPOTHESIS 4: Zero-Inflation")
print("=" * 80)

n_zeros = (df['C'] == 0).sum()
prop_zeros = n_zeros / len(df)
expected_zeros_poisson = poisson.pmf(0, mean_C)
expected_zeros_nb = nbinom.pmf(0, r_mle, p_mle)

print(f"\nObserved proportion of zeros: {prop_zeros:.4f} ({n_zeros} out of {len(df)})")
print(f"Expected under Poisson({mean_C:.1f}): {expected_zeros_poisson:.6f}")
print(f"Expected under NegBinom(r={r_mle:.2f}): {expected_zeros_nb:.6f}")

if n_zeros > 0:
    excess_zeros = prop_zeros - expected_zeros_nb
    print(f"\nExcess zeros (observed - expected NB): {excess_zeros:.4f}")
    if excess_zeros > 0.05:
        print("CONCLUSION: Evidence of zero-inflation")
    else:
        print("CONCLUSION: No evidence of zero-inflation")
else:
    print("\nCONCLUSION: No zeros observed - zero-inflation not present")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("HYPOTHESIS TESTING SUMMARY")
print("=" * 80)

print("\n1. POISSON DISTRIBUTION:")
print(f"   - Variance/Mean ratio: {var_C/mean_C:.2f} (should be ~1)")
print(f"   - Chi-square test: p = {p_value_chi:.6f} ({'REJECT' if p_value_chi < 0.05 else 'FAIL TO REJECT'})")
print(f"   - KS test: p = {ks_pvalue:.6f} ({'REJECT' if ks_pvalue < 0.05 else 'FAIL TO REJECT'})")
print(f"   - VERDICT: Poisson is {'NOT appropriate' if p_value_chi < 0.05 else 'potentially appropriate'}")

print("\n2. NEGATIVE BINOMIAL DISTRIBUTION:")
print(f"   - Estimated r: {r_mle:.4f}")
print(f"   - KS test: p = {ks_pvalue_nb:.6f} ({'REJECT' if ks_pvalue_nb < 0.05 else 'FAIL TO REJECT'})")
print(f"   - LR test vs Poisson: p = {lr_pvalue:.6f} ({'SIGNIFICANTLY BETTER' if lr_pvalue < 0.05 else 'NOT significantly better'})")
print(f"   - Delta AIC vs Poisson: {aic_poisson - aic_nb:.2f} ({'MUCH BETTER' if aic_poisson - aic_nb > 10 else 'BETTER' if aic_poisson - aic_nb > 4 else 'SLIGHTLY BETTER'})")
print(f"   - VERDICT: Negative Binomial is {'strongly recommended' if aic_poisson - aic_nb > 10 else 'recommended' if aic_poisson - aic_nb > 4 else 'potentially appropriate'}")

print("\n3. TIME-VARYING DISPERSION:")
print(f"   - Levene's test: p = {levene_pvalue:.6f} ({'EVIDENCE of time-varying dispersion' if levene_pvalue < 0.05 else 'NO clear evidence'})")
print(f"   - Variance increases with mean: {'YES' if groups[-1].var() > groups[0].var() * 2 else 'MODERATE'}")

print("\n4. ZERO-INFLATION:")
print(f"   - Observed zeros: {n_zeros}")
print(f"   - VERDICT: {'Zero-inflation present' if n_zeros > 0 and prop_zeros > expected_zeros_nb + 0.05 else 'No zero-inflation'}")

print("\n" + "=" * 80)
