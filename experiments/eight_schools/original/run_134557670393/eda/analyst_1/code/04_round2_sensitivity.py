"""
Round 2: Sensitivity Analysis and Grouping Investigation
"""

import pandas as pd
import numpy as np
from scipy import stats

# Load data
data = pd.read_csv('/workspace/eda/analyst_1/code/processed_data_with_metrics.csv')

print("="*80)
print("ROUND 2: DEEPER INVESTIGATION")
print("="*80)

# ============================================================================
# INVESTIGATION 1: Leave-One-Out Analysis
# ============================================================================
print("\n1. LEAVE-ONE-OUT SENSITIVITY ANALYSIS")
print("-"*80)
print("Testing robustness of pooled estimate and heterogeneity metrics\n")

loo_results = []

for idx in range(len(data)):
    # Remove one study
    subset = data.drop(index=idx)
    study_removed = int(data.iloc[idx]['study'])

    # Calculate pooled estimate
    weights = 1 / (subset['sigma'] ** 2)
    pooled = sum(weights * subset['y']) / sum(weights)
    pooled_se = np.sqrt(1 / sum(weights))

    # Calculate Q statistic
    Q = sum(weights * (subset['y'] - pooled) ** 2)
    df = len(subset) - 1
    p_value = 1 - stats.chi2.cdf(Q, df) if df > 0 else 1

    # Calculate I²
    I_squared = max(0, 100 * (Q - df) / Q) if Q > 0 else 0

    loo_results.append({
        'study_removed': study_removed,
        'pooled_effect': pooled,
        'pooled_se': pooled_se,
        'Q': Q,
        'Q_pvalue': p_value,
        'I_squared': I_squared
    })

loo_df = pd.DataFrame(loo_results)

print("Pooled effect estimates:")
for idx, row in loo_df.iterrows():
    change = row['pooled_effect'] - 7.686  # Original pooled effect
    print(f"  Without Study {row['study_removed']}: {row['pooled_effect']:.3f} "
          f"(change: {change:+.3f}, SE: {row['pooled_se']:.3f})")

print(f"\nRange of pooled estimates: [{loo_df['pooled_effect'].min():.3f}, "
      f"{loo_df['pooled_effect'].max():.3f}]")
print(f"SD of pooled estimates: {loo_df['pooled_effect'].std():.3f}")

print("\nI² values:")
for idx, row in loo_df.iterrows():
    print(f"  Without Study {row['study_removed']}: {row['I_squared']:.1f}% "
          f"(Q={row['Q']:.3f}, p={row['Q_pvalue']:.3f})")

# Find most influential study
most_influential_idx = np.argmax(np.abs(loo_df['pooled_effect'] - 7.686))
most_influential = loo_df.iloc[most_influential_idx]
print(f"\nMost influential study: {most_influential['study_removed']} "
      f"(removing it changes estimate by {abs(most_influential['pooled_effect'] - 7.686):.3f})")

loo_df.to_csv('/workspace/eda/analyst_1/code/leave_one_out_results.csv', index=False)

# ============================================================================
# INVESTIGATION 2: Simple K-means Clustering (Manual Implementation)
# ============================================================================
print("\n2. CLUSTERING ANALYSIS: Are there natural study groups?")
print("-"*80)

# Standardize features
y_std = (data['y'] - data['y'].mean()) / data['y'].std()
sigma_std = (data['sigma'] - data['sigma'].mean()) / data['sigma'].std()

# Simple 2-means clustering (manual implementation)
# Initialize with most extreme y values
cluster_centers = np.array([
    [y_std.min(), sigma_std[y_std.argmin()]],
    [y_std.max(), sigma_std[y_std.argmax()]]
])

# K-means iterations
for iteration in range(10):
    # Assign to nearest cluster
    distances_0 = (y_std - cluster_centers[0, 0])**2 + (sigma_std - cluster_centers[0, 1])**2
    distances_1 = (y_std - cluster_centers[1, 0])**2 + (sigma_std - cluster_centers[1, 1])**2
    labels = np.where(distances_0 < distances_1, 0, 1)

    # Update centers
    for k in range(2):
        if sum(labels == k) > 0:
            cluster_centers[k, 0] = y_std[labels == k].mean()
            cluster_centers[k, 1] = sigma_std[labels == k].mean()

data['cluster'] = labels

print("\nDetailed 2-cluster analysis:")
print("-"*40)
for cluster_id in range(2):
    cluster_data = data[data['cluster'] == cluster_id]
    print(f"\nCluster {cluster_id}:")
    print(f"  Studies: {sorted(list(cluster_data['study'].astype(int)))}")
    print(f"  N: {len(cluster_data)}")
    print(f"  Mean effect: {cluster_data['y'].mean():.2f} (SD: {cluster_data['y'].std():.2f})")
    print(f"  Mean SE: {cluster_data['sigma'].mean():.2f} (SD: {cluster_data['sigma'].std():.2f})")
    print(f"  Effect range: [{cluster_data['y'].min()}, {cluster_data['y'].max()}]")

# Test for between-cluster differences
cluster0 = data[data['cluster'] == 0]['y']
cluster1 = data[data['cluster'] == 1]['y']
if len(cluster0) > 1 and len(cluster1) > 1:
    t_stat, p_val = stats.ttest_ind(cluster0, cluster1)
    print(f"\nT-test between clusters: t={t_stat:.3f}, p={p_val:.3f}")
    print(f"Interpretation: {'Significant' if p_val < 0.05 else 'Not significant'} difference at alpha=0.05")

# Alternative: Simple thresholding
print("\n3. THRESHOLD-BASED GROUPING")
print("-"*80)
threshold = data['y'].median()
data['group'] = ['High' if y > threshold else 'Low' for y in data['y']]

print(f"Using median threshold = {threshold}")
for group in ['Low', 'High']:
    group_data = data[data['group'] == group]
    print(f"\n{group} effect group:")
    print(f"  Studies: {sorted(list(group_data['study'].astype(int)))}")
    print(f"  N: {len(group_data)}")
    print(f"  Mean effect: {group_data['y'].mean():.2f}")
    print(f"  Mean SE: {group_data['sigma'].mean():.2f}")

# Meta-analysis by group
print("\nMeta-analysis by group:")
for group in ['Low', 'High']:
    group_data = data[data['group'] == group]
    weights = 1 / (group_data['sigma'] ** 2)
    pooled = sum(weights * group_data['y']) / sum(weights)
    pooled_se = np.sqrt(1 / sum(weights))
    print(f"  {group}: {pooled:.2f} (SE: {pooled_se:.2f}), "
          f"95% CI: [{pooled - 1.96*pooled_se:.2f}, {pooled + 1.96*pooled_se:.2f}]")

# Test for difference between groups
low_data = data[data['group'] == 'Low']
high_data = data[data['group'] == 'High']
t_stat_group, p_val_group = stats.ttest_ind(low_data['y'], high_data['y'])
print(f"\nT-test between Low and High groups: t={t_stat_group:.3f}, p={p_val_group:.3f}")

# ============================================================================
# INVESTIGATION 3: Understanding the "Low Heterogeneity Paradox"
# ============================================================================
print("\n4. THE LOW HETEROGENEITY PARADOX: Why I²=0% despite large range?")
print("-"*80)

# Calculate what I² would be if standard errors were smaller
print("\nSimulation: What if standard errors were different?\n")

data_sim = data.copy()

# Try different SE reductions
print("I² vs SE scaling:")
for scale in [1.0, 0.75, 0.5, 0.25, 0.1]:
    data_temp = data.copy()
    data_temp['sigma_scaled'] = data_temp['sigma'] * scale
    weights_temp = 1 / (data_temp['sigma_scaled'] ** 2)
    pooled_temp = sum(weights_temp * data_temp['y']) / sum(weights_temp)
    Q_temp = sum(weights_temp * (data_temp['y'] - pooled_temp) ** 2)
    df_temp = len(data_temp) - 1
    I_squared_temp = max(0, 100 * (Q_temp - df_temp) / Q_temp)
    p_temp = 1 - stats.chi2.cdf(Q_temp, df_temp)
    print(f"  SE × {scale:.2f}: I² = {I_squared_temp:.1f}%, Q = {Q_temp:.1f}, p = {p_temp:.4f}")

print("\nKEY INSIGHT: Large standard errors dominate the heterogeneity statistics!")
print("With smaller SEs, the same effect size variation would show substantial heterogeneity.")

# Calculate the ratio of between-effect variance to within-study variance
between_var = data['y'].var()
within_var = np.mean(data['sigma'] ** 2)
print(f"\nVariance decomposition:")
print(f"  Between-study variance (observed effects): {between_var:.2f}")
print(f"  Mean within-study variance (SE²): {within_var:.2f}")
print(f"  Ratio (between/within): {between_var / within_var:.3f}")
print(f"\nInterpretation: Within-study variance is {within_var / between_var:.1f}x larger than between-study variance")

# ============================================================================
# INVESTIGATION 4: Overlap Analysis
# ============================================================================
print("\n5. CONFIDENCE INTERVAL OVERLAP ANALYSIS")
print("-"*80)

# Calculate pairwise overlaps
non_overlapping_pairs = []

for i in range(len(data)):
    for j in range(i+1, len(data)):
        ci1_lower = data.iloc[i]['ci_lower']
        ci1_upper = data.iloc[i]['ci_upper']
        ci2_lower = data.iloc[j]['ci_lower']
        ci2_upper = data.iloc[j]['ci_upper']

        # Check if they overlap
        if ci1_upper < ci2_lower or ci2_upper < ci1_lower:
            study_i = int(data.iloc[i]['study'])
            study_j = int(data.iloc[j]['study'])
            non_overlapping_pairs.append((study_i, study_j))

if len(non_overlapping_pairs) > 0:
    print("Non-overlapping confidence interval pairs:")
    for pair in non_overlapping_pairs:
        print(f"  Study {pair[0]} and Study {pair[1]}")
else:
    print("All study confidence intervals overlap!")

total_pairs = len(data) * (len(data) - 1) // 2
overlap_rate = 100 * (1 - len(non_overlapping_pairs) / total_pairs)
print(f"\nOverlap rate: {overlap_rate:.1f}% of study pairs have overlapping CIs")
print(f"Non-overlapping pairs: {len(non_overlapping_pairs)} out of {int(total_pairs)}")

# ============================================================================
# INVESTIGATION 5: Direction of Effects
# ============================================================================
print("\n6. DIRECTION OF EFFECTS ANALYSIS")
print("-"*80)

positive_effects = data[data['y'] > 0]
negative_effects = data[data['y'] < 0]
zero_effects = data[data['y'] == 0]

print(f"Positive effects: {len(positive_effects)} studies ({100*len(positive_effects)/len(data):.1f}%)")
print(f"  Studies: {sorted(list(positive_effects['study'].astype(int)))}")
print(f"  Mean: {positive_effects['y'].mean():.2f}, Range: [{positive_effects['y'].min()}, {positive_effects['y'].max()}]")

print(f"\nNegative effects: {len(negative_effects)} studies ({100*len(negative_effects)/len(data):.1f}%)")
print(f"  Studies: {sorted(list(negative_effects['study'].astype(int)))}")
print(f"  Mean: {negative_effects['y'].mean():.2f}, Range: [{negative_effects['y'].min()}, {negative_effects['y'].max()}]")

# Do negative effect studies have different precision?
if len(negative_effects) > 0:
    print(f"\nPrecision comparison:")
    print(f"  Positive effects mean SE: {positive_effects['sigma'].mean():.2f}")
    print(f"  Negative effects mean SE: {negative_effects['sigma'].mean():.2f}")

    # Test for difference
    if len(positive_effects) > 1 and len(negative_effects) > 1:
        t_stat_prec, p_val_prec = stats.ttest_ind(positive_effects['sigma'],
                                                    negative_effects['sigma'])
        print(f"  T-test for SE difference: t={t_stat_prec:.3f}, p={p_val_prec:.3f}")

# Check if negative effects' CIs include positive values
print(f"\nNegative effect studies' CIs:")
for idx, row in negative_effects.iterrows():
    includes_positive = row['ci_upper'] > 0
    includes_large_positive = row['ci_upper'] > 10
    print(f"  Study {int(row['study'])}: [{row['ci_lower']:.1f}, {row['ci_upper']:.1f}] - "
          f"{'Includes positive values' if includes_positive else 'Strictly negative'}")
    if includes_large_positive:
        print(f"    Note: CI includes effects as large as {row['ci_upper']:.1f}")

print("\n" + "="*80)
print("Round 2 analysis complete!")
print("="*80)
