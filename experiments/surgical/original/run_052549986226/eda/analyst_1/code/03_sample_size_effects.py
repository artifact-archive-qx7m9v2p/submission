"""
Sample Size Effects Analysis: Investigating relationship between n_trials and success_rate
EDA Analyst 1
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats

# Set up paths
BASE_DIR = Path('/workspace')
DATA_PATH = BASE_DIR / 'data' / 'data_analyst_1.csv'
VIZ_DIR = BASE_DIR / 'eda' / 'analyst_1' / 'visualizations'

# Load data
df = pd.read_csv(DATA_PATH)

print("=" * 70)
print("SAMPLE SIZE EFFECTS ANALYSIS")
print("=" * 70)

# ============================================================================
# Correlation Analysis
# ============================================================================
print("\nCORRELATION ANALYSIS:")
print("-" * 70)

# Pearson correlation (linear relationship)
pearson_r, pearson_p = stats.pearsonr(df['n_trials'], df['success_rate'])
print(f"\nPearson Correlation:")
print(f"  r = {pearson_r:.4f}")
print(f"  p-value = {pearson_p:.4f}")
print(f"  Interpretation: {'Significant' if pearson_p < 0.05 else 'Not significant'} (α=0.05)")

# Spearman correlation (monotonic relationship)
spearman_r, spearman_p = stats.spearmanr(df['n_trials'], df['success_rate'])
print(f"\nSpearman Correlation:")
print(f"  ρ = {spearman_r:.4f}")
print(f"  p-value = {spearman_p:.4f}")
print(f"  Interpretation: {'Significant' if spearman_p < 0.05 else 'Not significant'} (α=0.05)")

# ============================================================================
# Variance Analysis by Sample Size
# ============================================================================
print("\n" + "=" * 70)
print("VARIANCE ANALYSIS BY SAMPLE SIZE")
print("=" * 70)

# Divide groups into small, medium, large sample sizes
df_sorted = df.sort_values('n_trials')
n_per_bin = 4
bins = [df_sorted.iloc[i:i+n_per_bin] for i in range(0, len(df_sorted), n_per_bin)]

print("\nVariance by Sample Size Tertile:")
for i, bin_df in enumerate(bins):
    print(f"\n{'Small' if i==0 else 'Medium' if i==1 else 'Large'} Sample Size (n={bin_df['n_trials'].min()}-{bin_df['n_trials'].max()}):")
    print(f"  Mean success rate: {bin_df['success_rate'].mean():.4f}")
    print(f"  Std dev: {bin_df['success_rate'].std():.4f}")
    print(f"  Variance: {bin_df['success_rate'].var():.6f}")
    print(f"  Range: {bin_df['success_rate'].max() - bin_df['success_rate'].min():.4f}")
    print(f"  CV: {bin_df['success_rate'].std() / bin_df['success_rate'].mean():.4f}")

# ============================================================================
# FIGURE: Sample Size vs Success Rate with Confidence Bands
# ============================================================================
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Sample Size Effects on Success Rate', fontsize=16, fontweight='bold')

# Panel A: Scatter plot with confidence intervals
ax = axes[0]

# Plot points with size proportional to number of successes
scatter = ax.scatter(df['n_trials'], df['success_rate'],
                     s=df['r_successes']*10 + 50,
                     alpha=0.6, c=df['r_successes'], cmap='viridis',
                     edgecolor='black', linewidth=1)

# Add group labels
for idx, row in df.iterrows():
    ax.annotate(f"G{row['group']}",
                (row['n_trials'], row['success_rate']),
                xytext=(5, 5), textcoords='offset points',
                fontsize=8, alpha=0.7)

# Calculate and plot overall mean
overall_mean = df['success_rate'].mean()
ax.axhline(overall_mean, color='red', linestyle='--', linewidth=2,
           label=f'Overall Mean = {overall_mean:.4f}', alpha=0.8)

# Calculate 95% confidence intervals for binomial proportions
# Using Wilson score interval
n_range = np.linspace(df['n_trials'].min(), df['n_trials'].max(), 100)
p_hat = overall_mean
z = 1.96  # 95% confidence

ci_upper = []
ci_lower = []
for n in n_range:
    # Wilson score interval
    denominator = 1 + z**2/n
    center = (p_hat + z**2/(2*n)) / denominator
    margin = z * np.sqrt(p_hat*(1-p_hat)/n + z**2/(4*n**2)) / denominator
    ci_upper.append(center + margin)
    ci_lower.append(center - margin)

ax.fill_between(n_range, ci_lower, ci_upper, alpha=0.2, color='red',
                label='95% CI (Wilson)')

# Add colorbar
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('Number of Successes', fontsize=10)

ax.set_xlabel('Number of Trials (n_trials)', fontsize=11)
ax.set_ylabel('Success Rate', fontsize=11)
ax.set_title('(A) Success Rate vs Sample Size\n(bubble size = successes)',
             fontsize=12, fontweight='bold')
ax.legend(loc='best')
ax.grid(True, alpha=0.3)

# Panel B: Residuals from mean (checking for systematic patterns)
ax = axes[1]

residuals = df['success_rate'] - overall_mean
ax.scatter(df['n_trials'], residuals, s=100, alpha=0.6,
           c=df['r_successes'], cmap='viridis',
           edgecolor='black', linewidth=1)

# Add group labels
for idx, row in df.iterrows():
    ax.annotate(f"G{row['group']}",
                (row['n_trials'], residuals.iloc[idx]),
                xytext=(5, 5), textcoords='offset points',
                fontsize=8, alpha=0.7)

ax.axhline(0, color='red', linestyle='--', linewidth=2, label='Zero residual')

# Add expected standard error bands
se_upper = []
se_lower = []
for n in n_range:
    se = np.sqrt(p_hat * (1 - p_hat) / n)
    se_upper.append(1.96 * se)
    se_lower.append(-1.96 * se)

ax.fill_between(n_range, se_lower, se_upper, alpha=0.2, color='blue',
                label='Expected 95% range')

ax.set_xlabel('Number of Trials (n_trials)', fontsize=11)
ax.set_ylabel('Residual (observed - mean)', fontsize=11)
ax.set_title('(B) Residuals vs Sample Size\n(checking for funnel pattern)',
             fontsize=12, fontweight='bold')
ax.legend(loc='best')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(VIZ_DIR / 'sample_size_effects.png', dpi=300, bbox_inches='tight')
print(f"\nSaved: sample_size_effects.png")
plt.close()

# ============================================================================
# Calculate variance metrics
# ============================================================================
print("\n" + "=" * 70)
print("EXPECTED VS OBSERVED VARIANCE")
print("=" * 70)

# Calculate expected variance under binomial model
# For each group, expected variance is p(1-p)/n where p is the pooled proportion
pooled_p = df['r_successes'].sum() / df['n_trials'].sum()
print(f"\nPooled proportion (overall): {pooled_p:.4f}")

df['expected_variance'] = pooled_p * (1 - pooled_p) / df['n_trials']
df['expected_se'] = np.sqrt(df['expected_variance'])

print(f"\nExpected variance (weighted mean): {np.average(df['expected_variance'], weights=df['n_trials']):.6f}")
print(f"Observed variance: {df['success_rate'].var():.6f}")
print(f"Ratio (observed/expected): {df['success_rate'].var() / np.average(df['expected_variance'], weights=df['n_trials']):.4f}")

# Calculate standardized residuals
df['standardized_residual'] = (df['success_rate'] - pooled_p) / df['expected_se']

print("\nStandardized Residuals (Z-scores):")
print(df[['group', 'n_trials', 'success_rate', 'standardized_residual']].to_string(index=False))
