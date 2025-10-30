"""
Visualization Script for Binomial Dataset
==========================================
Creates comprehensive visualizations to understand data patterns.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats

# Setup
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100
DATA_PATH = Path("/workspace/data/data.csv")
VIZ_DIR = Path("/workspace/eda/visualizations")

# Load data
df = pd.read_csv(DATA_PATH)

# Calculate additional metrics
pooled_p = df['r'].sum() / df['n'].sum()
df['expected_var'] = pooled_p * (1 - pooled_p) / df['n']
df['expected_std'] = np.sqrt(df['expected_var'])
df['standardized_resid'] = (df['proportion'] - pooled_p) / df['expected_std']

# ============================================================================
# PLOT 1: Distribution of Sample Sizes
# ============================================================================
fig, ax = plt.subplots(figsize=(10, 6))
ax.hist(df['n'], bins=15, edgecolor='black', alpha=0.7, color='steelblue')
ax.axvline(df['n'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean = {df["n"].mean():.1f}')
ax.axvline(df['n'].median(), color='orange', linestyle='--', linewidth=2, label=f'Median = {df["n"].median():.1f}')
ax.set_xlabel('Sample Size (n)', fontsize=12)
ax.set_ylabel('Frequency', fontsize=12)
ax.set_title('Distribution of Sample Sizes Across Trials', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(VIZ_DIR / 'sample_size_distribution.png', bbox_inches='tight')
plt.close()
print("Saved: sample_size_distribution.png")

# ============================================================================
# PLOT 2: Distribution of Proportions
# ============================================================================
fig, ax = plt.subplots(figsize=(10, 6))
ax.hist(df['proportion'], bins=10, edgecolor='black', alpha=0.7, color='seagreen')
ax.axvline(df['proportion'].mean(), color='red', linestyle='--', linewidth=2,
           label=f'Mean = {df["proportion"].mean():.4f}')
ax.axvline(pooled_p, color='purple', linestyle='--', linewidth=2,
           label=f'Pooled p = {pooled_p:.4f}')
ax.axvline(df['proportion'].median(), color='orange', linestyle='--', linewidth=2,
           label=f'Median = {df["proportion"].median():.4f}')
ax.set_xlabel('Success Proportion', fontsize=12)
ax.set_ylabel('Frequency', fontsize=12)
ax.set_title('Distribution of Success Proportions Across Trials', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(VIZ_DIR / 'proportion_distribution.png', bbox_inches='tight')
plt.close()
print("Saved: proportion_distribution.png")

# ============================================================================
# PLOT 3: Proportions vs Trial ID (Temporal Pattern)
# ============================================================================
fig, ax = plt.subplots(figsize=(12, 6))
ax.scatter(df['trial_id'], df['proportion'], s=100, alpha=0.6, color='steelblue', edgecolor='black')
ax.axhline(pooled_p, color='red', linestyle='--', linewidth=2, label=f'Pooled p = {pooled_p:.4f}')
ax.axhline(df['proportion'].mean(), color='green', linestyle=':', linewidth=2,
           label=f'Mean = {df["proportion"].mean():.4f}')

# Add error bars based on binomial SE
for idx, row in df.iterrows():
    se = row['expected_std']
    ax.plot([row['trial_id'], row['trial_id']],
            [row['proportion'] - 1.96*se, row['proportion'] + 1.96*se],
            color='gray', alpha=0.5, linewidth=1.5)

ax.set_xlabel('Trial ID', fontsize=12)
ax.set_ylabel('Success Proportion', fontsize=12)
ax.set_title('Success Proportions by Trial ID\n(Gray bars show 95% CI under binomial model)',
             fontsize=14, fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)
ax.set_xticks(df['trial_id'])
plt.tight_layout()
plt.savefig(VIZ_DIR / 'proportion_vs_trial.png', bbox_inches='tight')
plt.close()
print("Saved: proportion_vs_trial.png")

# ============================================================================
# PLOT 4: Proportion vs Sample Size
# ============================================================================
fig, ax = plt.subplots(figsize=(10, 6))
scatter = ax.scatter(df['n'], df['proportion'], s=150, alpha=0.6,
                     c=df['trial_id'], cmap='viridis', edgecolor='black')
ax.axhline(pooled_p, color='red', linestyle='--', linewidth=2, label=f'Pooled p = {pooled_p:.4f}')

# Add labels for each point
for idx, row in df.iterrows():
    ax.annotate(f"{int(row['trial_id'])}",
                (row['n'], row['proportion']),
                xytext=(5, 5), textcoords='offset points', fontsize=9)

ax.set_xlabel('Sample Size (n)', fontsize=12)
ax.set_ylabel('Success Proportion', fontsize=12)
ax.set_title('Success Proportion vs Sample Size', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)
plt.colorbar(scatter, ax=ax, label='Trial ID')
plt.tight_layout()
plt.savefig(VIZ_DIR / 'proportion_vs_sample_size.png', bbox_inches='tight')
plt.close()
print("Saved: proportion_vs_sample_size.png")

# ============================================================================
# PLOT 5: Standardized Residuals
# ============================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Residuals vs Trial ID
ax1.scatter(df['trial_id'], df['standardized_resid'], s=100, alpha=0.6,
            color='steelblue', edgecolor='black')
ax1.axhline(0, color='black', linestyle='-', linewidth=1)
ax1.axhline(2, color='red', linestyle='--', linewidth=1, alpha=0.5, label='±2 SD')
ax1.axhline(-2, color='red', linestyle='--', linewidth=1, alpha=0.5)
ax1.set_xlabel('Trial ID', fontsize=12)
ax1.set_ylabel('Standardized Residual', fontsize=12)
ax1.set_title('Standardized Residuals by Trial', fontsize=12, fontweight='bold')
ax1.grid(alpha=0.3)
ax1.legend()
ax1.set_xticks(df['trial_id'])

# Histogram of standardized residuals
ax2.hist(df['standardized_resid'], bins=8, edgecolor='black', alpha=0.7, color='coral')
# Overlay normal distribution
x = np.linspace(-3, 4, 100)
ax2.plot(x, len(df) * stats.norm.pdf(x, 0, 1) * (df['standardized_resid'].max() - df['standardized_resid'].min()) / 8,
         'r--', linewidth=2, label='N(0,1)')
ax2.axvline(0, color='black', linestyle='-', linewidth=1)
ax2.set_xlabel('Standardized Residual', fontsize=12)
ax2.set_ylabel('Frequency', fontsize=12)
ax2.set_title('Distribution of Standardized Residuals', fontsize=12, fontweight='bold')
ax2.grid(axis='y', alpha=0.3)
ax2.legend()

plt.tight_layout()
plt.savefig(VIZ_DIR / 'standardized_residuals.png', bbox_inches='tight')
plt.close()
print("Saved: standardized_residuals.png")

# ============================================================================
# PLOT 6: Observed vs Expected Variance (Multi-panel)
# ============================================================================
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 12))

# Panel 1: Success counts
ax1.bar(df['trial_id'], df['r'], alpha=0.7, color='steelblue', edgecolor='black')
ax1.set_xlabel('Trial ID', fontsize=11)
ax1.set_ylabel('Number of Successes (r)', fontsize=11)
ax1.set_title('Success Counts by Trial', fontsize=12, fontweight='bold')
ax1.grid(axis='y', alpha=0.3)
ax1.set_xticks(df['trial_id'])

# Panel 2: Sample sizes
ax2.bar(df['trial_id'], df['n'], alpha=0.7, color='seagreen', edgecolor='black')
ax2.set_xlabel('Trial ID', fontsize=11)
ax2.set_ylabel('Sample Size (n)', fontsize=11)
ax2.set_title('Sample Sizes by Trial', fontsize=12, fontweight='bold')
ax2.grid(axis='y', alpha=0.3)
ax2.set_xticks(df['trial_id'])

# Panel 3: Proportion with expected SE
x_pos = np.arange(len(df))
ax3.bar(x_pos, df['proportion'], alpha=0.7, color='coral', edgecolor='black', label='Observed')
ax3.errorbar(x_pos, df['proportion'], yerr=1.96*df['expected_std'],
             fmt='none', ecolor='gray', capsize=5, alpha=0.7,
             label='95% CI (binomial)')
ax3.axhline(pooled_p, color='red', linestyle='--', linewidth=2, label='Pooled p')
ax3.set_xlabel('Trial ID', fontsize=11)
ax3.set_ylabel('Proportion', fontsize=11)
ax3.set_title('Proportions with Binomial 95% CI', fontsize=12, fontweight='bold')
ax3.set_xticks(x_pos)
ax3.set_xticklabels(df['trial_id'])
ax3.grid(axis='y', alpha=0.3)
ax3.legend()

# Panel 4: Variance comparison
obs_var_by_trial = (df['proportion'] - pooled_p)**2
exp_var_by_trial = df['expected_var']
width = 0.35
x_pos = np.arange(len(df))
ax4.bar(x_pos - width/2, obs_var_by_trial, width, alpha=0.7,
        color='darkred', edgecolor='black', label='Observed (p - p_pooled)²')
ax4.bar(x_pos + width/2, exp_var_by_trial, width, alpha=0.7,
        color='darkblue', edgecolor='black', label='Expected (binomial)')
ax4.set_xlabel('Trial ID', fontsize=11)
ax4.set_ylabel('Variance', fontsize=11)
ax4.set_title('Observed vs Expected Variance by Trial', fontsize=12, fontweight='bold')
ax4.set_xticks(x_pos)
ax4.set_xticklabels(df['trial_id'])
ax4.grid(axis='y', alpha=0.3)
ax4.legend()
ax4.set_yscale('log')

plt.tight_layout()
plt.savefig(VIZ_DIR / 'comprehensive_comparison.png', bbox_inches='tight')
plt.close()
print("Saved: comprehensive_comparison.png")

# ============================================================================
# PLOT 7: QQ Plot for Normality Check of Standardized Residuals
# ============================================================================
fig, ax = plt.subplots(figsize=(8, 8))
stats.probplot(df['standardized_resid'], dist="norm", plot=ax)
ax.set_title('Q-Q Plot: Standardized Residuals vs Normal Distribution',
             fontsize=14, fontweight='bold')
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(VIZ_DIR / 'qq_plot.png', bbox_inches='tight')
plt.close()
print("Saved: qq_plot.png")

# ============================================================================
# PLOT 8: Funnel Plot (Proportion vs Precision)
# ============================================================================
fig, ax = plt.subplots(figsize=(10, 6))

# Precision = 1/SE
df['precision'] = 1 / df['expected_std']

ax.scatter(df['precision'], df['proportion'], s=100, alpha=0.6,
           color='steelblue', edgecolor='black')

# Add funnel limits (95% CI)
precision_range = np.linspace(df['precision'].min() * 0.9,
                              df['precision'].max() * 1.1, 100)
se_range = 1 / precision_range
upper_limit = pooled_p + 1.96 * se_range
lower_limit = pooled_p - 1.96 * se_range

ax.plot(precision_range, upper_limit, 'r--', alpha=0.5, label='95% CI limits')
ax.plot(precision_range, lower_limit, 'r--', alpha=0.5)
ax.axhline(pooled_p, color='red', linestyle='-', linewidth=2, label='Pooled p')

# Annotate outliers
for idx, row in df.iterrows():
    if abs(row['standardized_resid']) > 2:
        ax.annotate(f"Trial {int(row['trial_id'])}",
                    (row['precision'], row['proportion']),
                    xytext=(10, 10), textcoords='offset points',
                    fontsize=9, color='red',
                    arrowprops=dict(arrowstyle='->', color='red', alpha=0.5))

ax.set_xlabel('Precision (1/SE)', fontsize=12)
ax.set_ylabel('Success Proportion', fontsize=12)
ax.set_title('Funnel Plot: Proportion vs Precision\n(Points outside funnel indicate overdispersion)',
             fontsize=14, fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(VIZ_DIR / 'funnel_plot.png', bbox_inches='tight')
plt.close()
print("Saved: funnel_plot.png")

print("\nAll visualizations created successfully!")
