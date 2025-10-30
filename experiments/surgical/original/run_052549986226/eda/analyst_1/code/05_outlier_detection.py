"""
Outlier Detection: Identifying unusual groups
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
print("OUTLIER DETECTION ANALYSIS")
print("=" * 70)

# ============================================================================
# Calculate metrics for outlier detection
# ============================================================================

# Pooled proportion
pooled_p = df['r_successes'].sum() / df['n_trials'].sum()

# Expected values under binomial
df['expected_successes'] = df['n_trials'] * pooled_p
df['expected_variance'] = pooled_p * (1 - pooled_p) / df['n_trials']
df['expected_se'] = np.sqrt(df['expected_variance'])

# Standardized residuals (z-scores)
df['standardized_residual'] = (df['success_rate'] - pooled_p) / df['expected_se']

# Deviance residuals
df['deviance_residual'] = np.sign(df['r_successes'] - df['expected_successes']) * np.sqrt(
    2 * (df['r_successes'] * np.log((df['r_successes'] + 0.5) / df['expected_successes']) +
         (df['n_trials'] - df['r_successes']) * np.log((df['n_trials'] - df['r_successes'] + 0.5) /
                                                        (df['n_trials'] - df['expected_successes'])))
)

# Cook's distance analog (leverage * standardized residual^2)
# Leverage for binomial: h_i = 1/n (simple approximation)
df['leverage'] = 1 / len(df)
df['cooks_d'] = df['leverage'] * df['standardized_residual']**2 / 2

# ============================================================================
# METHOD 1: Z-score based outliers
# ============================================================================
print("\n" + "-" * 70)
print("METHOD 1: Standardized Residuals (Z-scores)")
print("-" * 70)

print("\nAll groups with standardized residuals:")
print(df[['group', 'n_trials', 'success_rate', 'standardized_residual']].to_string(index=False))

outliers_2sigma = df[np.abs(df['standardized_residual']) > 1.96]
outliers_3sigma = df[np.abs(df['standardized_residual']) > 2.576]

print(f"\nOutliers at 95% level (|z| > 1.96): {len(outliers_2sigma)} groups")
if len(outliers_2sigma) > 0:
    print(outliers_2sigma[['group', 'n_trials', 'r_successes', 'success_rate', 'standardized_residual']].to_string(index=False))

print(f"\nOutliers at 99% level (|z| > 2.576): {len(outliers_3sigma)} groups")
if len(outliers_3sigma) > 0:
    print(outliers_3sigma[['group', 'n_trials', 'r_successes', 'success_rate', 'standardized_residual']].to_string(index=False))

# ============================================================================
# METHOD 2: IQR method on success rates
# ============================================================================
print("\n" + "-" * 70)
print("METHOD 2: IQR Method on Success Rates")
print("-" * 70)

q1 = df['success_rate'].quantile(0.25)
q3 = df['success_rate'].quantile(0.75)
iqr = q3 - q1

lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr

print(f"\nQ1: {q1:.4f}")
print(f"Q3: {q3:.4f}")
print(f"IQR: {iqr:.4f}")
print(f"Lower bound (Q1 - 1.5*IQR): {lower_bound:.4f}")
print(f"Upper bound (Q3 + 1.5*IQR): {upper_bound:.4f}")

outliers_iqr = df[(df['success_rate'] < lower_bound) | (df['success_rate'] > upper_bound)]
print(f"\nOutliers by IQR method: {len(outliers_iqr)} groups")
if len(outliers_iqr) > 0:
    print(outliers_iqr[['group', 'n_trials', 'r_successes', 'success_rate']].to_string(index=False))

# ============================================================================
# METHOD 3: Modified Z-score (using MAD)
# ============================================================================
print("\n" + "-" * 70)
print("METHOD 3: Modified Z-score (MAD-based)")
print("-" * 70)

median = df['success_rate'].median()
mad = np.median(np.abs(df['success_rate'] - median))
modified_z = 0.6745 * (df['success_rate'] - median) / mad

df['modified_z'] = modified_z

print(f"\nMedian: {median:.4f}")
print(f"MAD: {mad:.4f}")

outliers_mad = df[np.abs(df['modified_z']) > 3.5]
print(f"\nOutliers by modified Z-score (|z*| > 3.5): {len(outliers_mad)} groups")
if len(outliers_mad) > 0:
    print(outliers_mad[['group', 'n_trials', 'success_rate', 'modified_z']].to_string(index=False))

# ============================================================================
# METHOD 4: Extreme values check
# ============================================================================
print("\n" + "-" * 70)
print("METHOD 4: Extreme Values")
print("-" * 70)

print("\nGroups with success_rate = 0:")
zero_rate = df[df['success_rate'] == 0]
if len(zero_rate) > 0:
    print(zero_rate[['group', 'n_trials', 'r_successes']].to_string(index=False))
else:
    print("None")

print("\nGroups in top 10% of success rates:")
top_10pct = df[df['success_rate'] >= df['success_rate'].quantile(0.9)]
print(top_10pct[['group', 'n_trials', 'r_successes', 'success_rate']].to_string(index=False))

print("\nGroups in bottom 10% of success rates:")
bottom_10pct = df[df['success_rate'] <= df['success_rate'].quantile(0.1)]
print(bottom_10pct[['group', 'n_trials', 'r_successes', 'success_rate']].to_string(index=False))

# ============================================================================
# FIGURE: Outlier Detection Visualization
# ============================================================================
fig = plt.figure(figsize=(16, 12))
gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

fig.suptitle('Outlier Detection Analysis', fontsize=16, fontweight='bold')

# Panel 1: Standardized residuals
ax1 = fig.add_subplot(gs[0, 0])
colors = ['red' if abs(z) > 1.96 else 'orange' if abs(z) > 1.5 else 'blue'
          for z in df['standardized_residual']]
ax1.bar(df['group'], df['standardized_residual'], color=colors, alpha=0.7, edgecolor='black')
ax1.axhline(0, color='black', linestyle='-', linewidth=1)
ax1.axhline(1.96, color='red', linestyle='--', linewidth=1, label='95% threshold')
ax1.axhline(-1.96, color='red', linestyle='--', linewidth=1)
ax1.axhline(2.576, color='darkred', linestyle=':', linewidth=1, label='99% threshold')
ax1.axhline(-2.576, color='darkred', linestyle=':', linewidth=1)
ax1.set_xlabel('Group', fontsize=11)
ax1.set_ylabel('Standardized Residual (Z-score)', fontsize=11)
ax1.set_title('(A) Standardized Residuals by Group', fontsize=12, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3, axis='y')

# Panel 2: Box plot with outliers
ax2 = fig.add_subplot(gs[0, 1])
bp = ax2.boxplot(df['success_rate'], vert=True, patch_artist=True, widths=0.5,
                 showfliers=True, flierprops=dict(marker='o', markerfacecolor='red',
                                                   markersize=10, alpha=0.7))
bp['boxes'][0].set_facecolor('lightblue')
bp['boxes'][0].set_edgecolor('black')

# Add horizontal lines for all data points
for idx, rate in enumerate(df['success_rate']):
    color = 'red' if rate in outliers_iqr['success_rate'].values else 'blue'
    ax2.scatter(1, rate, s=100, alpha=0.6, c=color, edgecolor='black', linewidth=1, zorder=3)
    ax2.text(1.15, rate, f"G{df.iloc[idx]['group']}", fontsize=9, va='center',
            fontweight='bold' if color=='red' else 'normal')

ax2.set_ylabel('Success Rate', fontsize=11)
ax2.set_title('(B) Box Plot with Outliers (IQR method)', fontsize=12, fontweight='bold')
ax2.set_xticks([])
ax2.grid(True, alpha=0.3, axis='y')

# Panel 3: Deviance residuals
ax3 = fig.add_subplot(gs[1, 0])
colors = ['red' if abs(z) > 2 else 'orange' if abs(z) > 1.5 else 'blue'
          for z in df['deviance_residual']]
ax3.bar(df['group'], df['deviance_residual'], color=colors, alpha=0.7, edgecolor='black')
ax3.axhline(0, color='black', linestyle='-', linewidth=1)
ax3.axhline(2, color='red', linestyle='--', linewidth=1, label='Threshold ±2')
ax3.axhline(-2, color='red', linestyle='--', linewidth=1)
ax3.set_xlabel('Group', fontsize=11)
ax3.set_ylabel('Deviance Residual', fontsize=11)
ax3.set_title('(C) Deviance Residuals by Group', fontsize=12, fontweight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3, axis='y')

# Panel 4: Cook's distance analog
ax4 = fig.add_subplot(gs[1, 1])
colors = ['red' if d > 0.5 else 'orange' if d > 0.2 else 'blue'
          for d in df['cooks_d']]
ax4.bar(df['group'], df['cooks_d'], color=colors, alpha=0.7, edgecolor='black')
ax4.axhline(0.5, color='red', linestyle='--', linewidth=1, label='Threshold 0.5')
ax4.axhline(0.2, color='orange', linestyle=':', linewidth=1, label='Threshold 0.2')
ax4.set_xlabel('Group', fontsize=11)
ax4.set_ylabel("Cook's Distance (analog)", fontsize=11)
ax4.set_title("(D) Cook's Distance by Group", fontsize=12, fontweight='bold')
ax4.legend()
ax4.grid(True, alpha=0.3, axis='y')

# Panel 5: Q-Q plot
ax5 = fig.add_subplot(gs[2, 0])
stats.probplot(df['standardized_residual'], dist="norm", plot=ax5)
ax5.set_title('(E) Q-Q Plot of Standardized Residuals', fontsize=12, fontweight='bold')
ax5.grid(True, alpha=0.3)

# Highlight outliers in Q-Q plot
outlier_indices = df[np.abs(df['standardized_residual']) > 1.96].index
if len(outlier_indices) > 0:
    sorted_std_res = np.sort(df['standardized_residual'])
    theoretical_quantiles = stats.norm.ppf((np.arange(len(df)) + 0.5) / len(df))
    for idx in outlier_indices:
        val = df.loc[idx, 'standardized_residual']
        pos = np.where(sorted_std_res == val)[0][0]
        ax5.plot(theoretical_quantiles[pos], val, 'ro', markersize=10, alpha=0.7)

# Panel 6: Success rate vs sample size with outliers highlighted
ax6 = fig.add_subplot(gs[2, 1])

outlier_groups = set(outliers_2sigma['group'].values)
colors = ['red' if g in outlier_groups else 'blue' for g in df['group']]
sizes = [200 if g in outlier_groups else 100 for g in df['group']]

ax6.scatter(df['n_trials'], df['success_rate'], s=sizes, alpha=0.6,
           c=colors, edgecolor='black', linewidth=1.5)

# Add labels to outliers
for idx, row in df.iterrows():
    if row['group'] in outlier_groups:
        ax6.annotate(f"G{row['group']}\n(z={df.loc[idx, 'standardized_residual']:.2f})",
                    (row['n_trials'], row['success_rate']),
                    xytext=(10, 10), textcoords='offset points',
                    fontsize=9, fontweight='bold', color='red',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.5),
                    arrowprops=dict(arrowstyle='->', color='red', lw=1.5))

ax6.axhline(pooled_p, color='green', linestyle='--', linewidth=2, label=f'Pooled mean = {pooled_p:.4f}')
ax6.set_xlabel('Sample Size (n_trials)', fontsize=11)
ax6.set_ylabel('Success Rate', fontsize=11)
ax6.set_title('(F) Outliers in Context (red = |z| > 1.96)', fontsize=12, fontweight='bold')
ax6.legend()
ax6.grid(True, alpha=0.3)

plt.savefig(VIZ_DIR / 'outlier_detection.png', dpi=300, bbox_inches='tight')
print(f"\nSaved: outlier_detection.png")
plt.close()

# ============================================================================
# Summary Table
# ============================================================================
print("\n" + "=" * 70)
print("OUTLIER SUMMARY TABLE")
print("=" * 70)

# Create summary
summary_df = df[['group', 'n_trials', 'r_successes', 'success_rate',
                 'standardized_residual', 'deviance_residual', 'cooks_d']].copy()
summary_df['is_outlier_zscore'] = np.abs(summary_df['standardized_residual']) > 1.96
summary_df['is_outlier_iqr'] = summary_df['success_rate'].isin(outliers_iqr['success_rate'])
summary_df['is_outlier_any'] = summary_df['is_outlier_zscore'] | summary_df['is_outlier_iqr']

# Sort by absolute standardized residual
summary_df = summary_df.sort_values('standardized_residual', key=abs, ascending=False)

print("\nAll groups sorted by |standardized residual|:")
print(summary_df.to_string(index=False))

print("\n" + "=" * 70)
print("IDENTIFIED OUTLIERS WITH JUSTIFICATION")
print("=" * 70)

outlier_list = []

for idx, row in df.iterrows():
    reasons = []

    if abs(row['standardized_residual']) > 2.576:
        reasons.append(f"Z-score = {row['standardized_residual']:.2f} (>99% threshold)")
    elif abs(row['standardized_residual']) > 1.96:
        reasons.append(f"Z-score = {row['standardized_residual']:.2f} (>95% threshold)")

    if row['group'] in outliers_iqr['group'].values:
        reasons.append("Outside IQR bounds")

    if row['success_rate'] == 0:
        reasons.append("Zero success rate")

    if row['success_rate'] >= df['success_rate'].quantile(0.9):
        reasons.append("Top 10% success rate")
    elif row['success_rate'] <= df['success_rate'].quantile(0.1):
        reasons.append("Bottom 10% success rate")

    if len(reasons) > 0:
        outlier_list.append({
            'group': row['group'],
            'n_trials': row['n_trials'],
            'r_successes': row['r_successes'],
            'success_rate': row['success_rate'],
            'z_score': row['standardized_residual'],
            'reasons': '; '.join(reasons)
        })

if outlier_list:
    for item in sorted(outlier_list, key=lambda x: abs(x['z_score']), reverse=True):
        print(f"\nGroup {item['group']}:")
        print(f"  Sample size: {item['n_trials']}")
        print(f"  Successes: {item['r_successes']}")
        print(f"  Success rate: {item['success_rate']:.4f}")
        print(f"  Z-score: {item['z_score']:.2f}")
        print(f"  Reasons: {item['reasons']}")
else:
    print("\nNo outliers detected.")

print("\n" + "=" * 70)
print("CONCLUSION")
print("=" * 70)
print(f"\nTotal groups: {len(df)}")
print(f"Outliers (Z-score > 1.96): {len(outliers_2sigma)}")
print(f"Outliers (IQR method): {len(outliers_iqr)}")
print(f"Unique outliers (any method): {len(outlier_list)}")
print(f"\nPercentage of outliers: {len(outlier_list) / len(df) * 100:.1f}%")
