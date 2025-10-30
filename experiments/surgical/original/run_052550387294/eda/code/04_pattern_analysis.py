"""
Pattern and Hypothesis Testing Analysis
========================================
Tests specific hypotheses about data structure and patterns.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
from scipy.stats import spearmanr, pearsonr

# Load data
DATA_PATH = Path("/workspace/data/data.csv")
df = pd.read_csv(DATA_PATH)

# Calculate metrics
pooled_p = df['r'].sum() / df['n'].sum()
df['expected_var'] = pooled_p * (1 - pooled_p) / df['n']
df['expected_std'] = np.sqrt(df['expected_var'])
df['standardized_resid'] = (df['proportion'] - pooled_p) / df['expected_std']

print("="*70)
print("PATTERN AND HYPOTHESIS TESTING ANALYSIS")
print("="*70)

# ============================================================================
# HYPOTHESIS 1: Is there a temporal trend?
# ============================================================================
print("\n1. TEMPORAL PATTERN ANALYSIS")
print("-"*70)
print("H0: No temporal trend in proportions across trial_id")

# Correlation tests
pearson_r, pearson_p = pearsonr(df['trial_id'], df['proportion'])
spearman_r, spearman_p = spearmanr(df['trial_id'], df['proportion'])

print(f"\nPearson correlation (trial_id vs proportion):")
print(f"  r = {pearson_r:.4f}, p-value = {pearson_p:.4f}")

print(f"\nSpearman correlation (trial_id vs proportion):")
print(f"  rho = {spearman_r:.4f}, p-value = {spearman_p:.4f}")

if pearson_p < 0.05 or spearman_p < 0.05:
    print("\nResult: Evidence of temporal trend")
else:
    print("\nResult: No significant temporal trend detected")

# Linear regression
from scipy.stats import linregress
slope, intercept, r_value, p_value, std_err = linregress(df['trial_id'], df['proportion'])
print(f"\nLinear regression: proportion = {intercept:.6f} + {slope:.6f} * trial_id")
print(f"  R² = {r_value**2:.4f}, p-value = {p_value:.4f}")

# ============================================================================
# HYPOTHESIS 2: Is there a relationship between sample size and proportion?
# ============================================================================
print("\n2. SAMPLE SIZE VS PROPORTION ANALYSIS")
print("-"*70)
print("H0: No relationship between sample size and observed proportion")

pearson_r2, pearson_p2 = pearsonr(df['n'], df['proportion'])
spearman_r2, spearman_p2 = spearmanr(df['n'], df['proportion'])

print(f"\nPearson correlation (n vs proportion):")
print(f"  r = {pearson_r2:.4f}, p-value = {pearson_p2:.4f}")

print(f"\nSpearman correlation (n vs proportion):")
print(f"  rho = {spearman_r2:.4f}, p-value = {spearman_p2:.4f}")

if pearson_p2 < 0.05 or spearman_p2 < 0.05:
    print("\nResult: Evidence of relationship between sample size and proportion")
    print("  (This could indicate bias or that larger samples come from different populations)")
else:
    print("\nResult: No significant relationship - supports homogeneity assumption")

# ============================================================================
# HYPOTHESIS 3: Are there distinct groups/clusters?
# ============================================================================
print("\n3. CLUSTERING ANALYSIS")
print("-"*70)
print("Visual inspection for potential groups:")

# Simple 2-group split by median
median_prop = df['proportion'].median()
df['group'] = (df['proportion'] > median_prop).astype(int)

print("\nTwo-group split by median:")
for group in [0, 1]:
    group_data = df[df['group'] == group]
    print(f"\n  Group {group} ({'Below' if group == 0 else 'Above'} median):")
    print(f"    Trials: {group_data['trial_id'].tolist()}")
    print(f"    Mean proportion: {group_data['proportion'].mean():.4f}")
    print(f"    Std proportion: {group_data['proportion'].std():.4f}")
    print(f"    Size: {len(group_data)} observations")

# Statistical test between groups
group_0_props = df[df['group'] == 0]['proportion']
group_1_props = df[df['group'] == 1]['proportion']
t_stat, t_p = stats.ttest_ind(group_0_props, group_1_props)
print(f"\n  T-test between groups: t={t_stat:.4f}, p={t_p:.4f}")

# Alternative: split by terciles to see if there are 3 regimes
terciles = df['proportion'].quantile([1/3, 2/3])
df['tercile'] = pd.cut(df['proportion'],
                       bins=[-np.inf, terciles.iloc[0], terciles.iloc[1], np.inf],
                       labels=['Low', 'Medium', 'High'])

print("\nThree-group split by terciles:")
for tercile in ['Low', 'Medium', 'High']:
    terc_data = df[df['tercile'] == tercile]
    if len(terc_data) > 0:
        print(f"\n  {tercile} group:")
        print(f"    Trials: {terc_data['trial_id'].tolist()}")
        print(f"    Mean proportion: {terc_data['proportion'].mean():.4f}")
        print(f"    Size: {len(terc_data)} observations")

# ============================================================================
# HYPOTHESIS 4: Is there evidence of multiple probability regimes?
# ============================================================================
print("\n4. MULTIPLE PROBABILITY REGIMES")
print("-"*70)

# Gap analysis - are there natural breaks?
sorted_props = np.sort(df['proportion'])
gaps = np.diff(sorted_props)
largest_gap_idx = np.argmax(gaps)
largest_gap = gaps[largest_gap_idx]

print(f"Largest gap in sorted proportions:")
print(f"  Gap size: {largest_gap:.4f}")
print(f"  Between {sorted_props[largest_gap_idx]:.4f} and {sorted_props[largest_gap_idx + 1]:.4f}")

# Show all gaps
print(f"\nAll gaps (sorted by size):")
gap_indices = np.argsort(gaps)[::-1]
for i, idx in enumerate(gap_indices[:5], 1):
    print(f"  {i}. Gap of {gaps[idx]:.4f} between {sorted_props[idx]:.4f} and {sorted_props[idx+1]:.4f}")

# Identify potential outliers
q1 = df['proportion'].quantile(0.25)
q3 = df['proportion'].quantile(0.75)
iqr = q3 - q1
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr

outliers = df[(df['proportion'] < lower_bound) | (df['proportion'] > upper_bound)]
print(f"\nIQR-based outliers (1.5*IQR rule):")
print(f"  Lower bound: {lower_bound:.4f}")
print(f"  Upper bound: {upper_bound:.4f}")
if len(outliers) > 0:
    for idx, row in outliers.iterrows():
        print(f"  Trial {row['trial_id']:2.0f}: p = {row['proportion']:.4f}")
else:
    print("  None detected")

# ============================================================================
# HYPOTHESIS 5: Do certain trials systematically differ?
# ============================================================================
print("\n5. SYSTEMATIC DIFFERENCES")
print("-"*70)

# Identify high and low proportion trials
high_threshold = pooled_p + 1.5 * df['proportion'].std()
low_threshold = pooled_p - 1.5 * df['proportion'].std()

high_prop_trials = df[df['proportion'] > high_threshold]
low_prop_trials = df[df['proportion'] < low_threshold]

print(f"Trials with unusually high proportions (> {high_threshold:.4f}):")
if len(high_prop_trials) > 0:
    for idx, row in high_prop_trials.iterrows():
        print(f"  Trial {row['trial_id']:2.0f}: p = {row['proportion']:.4f}, n = {row['n']}, r = {row['r']}")
else:
    print("  None")

print(f"\nTrials with unusually low proportions (< {low_threshold:.4f}):")
if len(low_prop_trials) > 0:
    for idx, row in low_prop_trials.iterrows():
        print(f"  Trial {row['trial_id']:2.0f}: p = {row['proportion']:.4f}, n = {row['n']}, r = {row['r']}")
else:
    print("  None")

# Check trial 1 specifically (has 0 successes)
trial_1 = df[df['trial_id'] == 1]
print(f"\nSpecial case - Trial 1 (zero successes):")
print(f"  n = {trial_1['n'].values[0]}, r = {trial_1['r'].values[0]}")
print(f"  Probability of 0 successes if p = {pooled_p:.4f}:")
prob_zero = (1 - pooled_p) ** trial_1['n'].values[0]
print(f"  P(r=0 | n={trial_1['n'].values[0]}, p={pooled_p:.4f}) = {prob_zero:.6f}")

# ============================================================================
# HYPOTHESIS 6: Runs test for randomness
# ============================================================================
print("\n6. RUNS TEST FOR RANDOMNESS")
print("-"*70)
print("Testing if sequence of proportions (above/below median) is random")

# Binary sequence: 1 if above median, 0 if below
median_prop = df['proportion'].median()
binary_sequence = (df['proportion'] > median_prop).astype(int)

print(f"\nBinary sequence (1=above median, 0=below):")
print(f"  {binary_sequence.tolist()}")

# Count runs
runs = 1
for i in range(1, len(binary_sequence)):
    if binary_sequence.iloc[i] != binary_sequence.iloc[i-1]:
        runs += 1

n1 = (binary_sequence == 1).sum()
n0 = (binary_sequence == 0).sum()

# Expected runs under randomness
expected_runs = ((2 * n0 * n1) / (n0 + n1)) + 1
var_runs = (2 * n0 * n1 * (2 * n0 * n1 - n0 - n1)) / ((n0 + n1)**2 * (n0 + n1 - 1))
z_runs = (runs - expected_runs) / np.sqrt(var_runs)
p_runs = 2 * (1 - stats.norm.cdf(abs(z_runs)))

print(f"\nObserved runs: {runs}")
print(f"Expected runs: {expected_runs:.2f}")
print(f"Z-score: {z_runs:.4f}")
print(f"P-value: {p_runs:.4f}")

if p_runs < 0.05:
    print("Result: Sequence is NOT random (evidence of pattern)")
else:
    print("Result: Sequence appears random")

# ============================================================================
# HYPOTHESIS 7: Variance homogeneity across subgroups
# ============================================================================
print("\n7. VARIANCE HOMOGENEITY TEST")
print("-"*70)

# Split by large vs small sample size
median_n = df['n'].median()
large_n = df[df['n'] > median_n]
small_n = df[df['n'] <= median_n]

print(f"Comparing variance in proportions:")
print(f"  Large samples (n > {median_n}): variance = {large_n['proportion'].var():.6f}")
print(f"  Small samples (n <= {median_n}): variance = {small_n['proportion'].var():.6f}")

# Levene test for equality of variances
levene_stat, levene_p = stats.levene(large_n['proportion'], small_n['proportion'])
print(f"\nLevene test: statistic = {levene_stat:.4f}, p-value = {levene_p:.4f}")

# ============================================================================
# SUMMARY OF PATTERN TESTS
# ============================================================================
print("\n" + "="*70)
print("PATTERN ANALYSIS SUMMARY")
print("="*70)

results = {
    'Temporal trend': 'Yes' if (pearson_p < 0.05 or spearman_p < 0.05) else 'No',
    'Sample size effect': 'Yes' if (pearson_p2 < 0.05 or spearman_p2 < 0.05) else 'No',
    'Distinct groups (median split)': 'Yes' if t_p < 0.05 else 'Weak/None',
    'Outliers present (IQR)': 'Yes' if len(outliers) > 0 else 'No',
    'Random sequence': 'Yes' if p_runs >= 0.05 else 'No (patterned)',
    'Variance homogeneity': 'Yes' if levene_p >= 0.05 else 'No'
}

for test, result in results.items():
    print(f"  {test:.<50} {result}")

print("\n" + "="*70)
