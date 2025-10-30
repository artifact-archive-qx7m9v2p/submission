"""
Diagnostic Visualizations for Model Assumptions
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path

# Set up paths
BASE_DIR = Path('/workspace')
DATA_PATH = BASE_DIR / 'eda' / 'analyst_3' / 'code' / 'data_with_diagnostics.csv'
VIZ_DIR = BASE_DIR / 'eda' / 'analyst_3' / 'visualizations'

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100

# Load data with diagnostics
df = pd.read_csv(DATA_PATH)

# ============================================================================
# FIGURE 1: Data Quality Overview (Multi-panel)
# ============================================================================
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Panel A: Sample sizes by group
ax = axes[0, 0]
colors = ['red' if n > 352 else 'steelblue' for n in df['n_trials']]
ax.bar(df['group'], df['n_trials'], color=colors, alpha=0.7, edgecolor='black')
ax.axhline(df['n_trials'].mean(), color='darkgreen', linestyle='--', linewidth=2, label='Mean')
ax.axhline(df['n_trials'].median(), color='orange', linestyle='--', linewidth=2, label='Median')
ax.set_xlabel('Group', fontsize=11, fontweight='bold')
ax.set_ylabel('Number of Trials', fontsize=11, fontweight='bold')
ax.set_title('A) Sample Size by Group\n(Red = outliers by IQR)', fontsize=12, fontweight='bold')
ax.legend(frameon=True)
ax.grid(axis='y', alpha=0.3)

# Panel B: Success rate by group
ax = axes[0, 1]
colors = ['red' if sr == 0 or sr > 0.1357 else 'steelblue' for sr in df['success_rate']]
ax.bar(df['group'], df['success_rate'], color=colors, alpha=0.7, edgecolor='black')
ax.axhline(df['success_rate'].mean(), color='darkgreen', linestyle='--', linewidth=2, label='Mean')
ax.axhline(df['success_rate'].median(), color='orange', linestyle='--', linewidth=2, label='Median')
ax.set_xlabel('Group', fontsize=11, fontweight='bold')
ax.set_ylabel('Success Rate', fontsize=11, fontweight='bold')
ax.set_title('B) Success Rate by Group\n(Red = outliers)', fontsize=12, fontweight='bold')
ax.legend(frameon=True)
ax.grid(axis='y', alpha=0.3)

# Panel C: Distribution of sample sizes
ax = axes[1, 0]
ax.hist(df['n_trials'], bins=8, color='steelblue', alpha=0.7, edgecolor='black')
ax.axvline(df['n_trials'].mean(), color='darkgreen', linestyle='--', linewidth=2, label='Mean')
ax.axvline(df['n_trials'].median(), color='orange', linestyle='--', linewidth=2, label='Median')
ax.set_xlabel('Number of Trials', fontsize=11, fontweight='bold')
ax.set_ylabel('Frequency', fontsize=11, fontweight='bold')
ax.set_title(f'C) Distribution of Sample Sizes\n(CV = {df["n_trials"].std()/df["n_trials"].mean():.2f})',
             fontsize=12, fontweight='bold')
ax.legend(frameon=True)
ax.grid(axis='y', alpha=0.3)

# Panel D: Distribution of success rates
ax = axes[1, 1]
ax.hist(df['success_rate'], bins=8, color='steelblue', alpha=0.7, edgecolor='black')
ax.axvline(df['success_rate'].mean(), color='darkgreen', linestyle='--', linewidth=2, label='Mean')
ax.axvline(df['success_rate'].median(), color='orange', linestyle='--', linewidth=2, label='Median')
ax.set_xlabel('Success Rate', fontsize=11, fontweight='bold')
ax.set_ylabel('Frequency', fontsize=11, fontweight='bold')
ax.set_title(f'D) Distribution of Success Rates\n(Range: {df["success_rate"].min():.3f} to {df["success_rate"].max():.3f})',
             fontsize=12, fontweight='bold')
ax.legend(frameon=True)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(VIZ_DIR / 'data_quality_overview.png', dpi=150, bbox_inches='tight')
plt.close()

print(f"Saved: {VIZ_DIR / 'data_quality_overview.png'}")

# ============================================================================
# FIGURE 2: Residual Diagnostics (Multi-panel)
# ============================================================================
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Panel A: Pearson residuals by group
ax = axes[0, 0]
colors = ['red' if abs(r) > 2 else 'steelblue' for r in df['pearson_residual']]
ax.scatter(df['group'], df['pearson_residual'], c=colors, s=100, alpha=0.7, edgecolor='black')
ax.axhline(0, color='black', linestyle='-', linewidth=1)
ax.axhline(2, color='red', linestyle='--', linewidth=1, alpha=0.5, label='±2 SD')
ax.axhline(-2, color='red', linestyle='--', linewidth=1, alpha=0.5)
ax.set_xlabel('Group', fontsize=11, fontweight='bold')
ax.set_ylabel('Pearson Residual', fontsize=11, fontweight='bold')
ax.set_title('A) Pearson Residuals vs Group\n(Red points: |residual| > 2)', fontsize=12, fontweight='bold')
ax.legend(frameon=True)
ax.grid(True, alpha=0.3)

# Panel B: Residuals vs fitted (expected successes)
ax = axes[0, 1]
colors = ['red' if abs(r) > 2 else 'steelblue' for r in df['pearson_residual']]
ax.scatter(df['expected_successes'], df['pearson_residual'], c=colors, s=100, alpha=0.7, edgecolor='black')
ax.axhline(0, color='black', linestyle='-', linewidth=1)
ax.axhline(2, color='red', linestyle='--', linewidth=1, alpha=0.5)
ax.axhline(-2, color='red', linestyle='--', linewidth=1, alpha=0.5)
ax.set_xlabel('Expected Successes (Fitted)', fontsize=11, fontweight='bold')
ax.set_ylabel('Pearson Residual', fontsize=11, fontweight='bold')
ax.set_title('B) Residuals vs Fitted Values', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)

# Panel C: Q-Q plot
ax = axes[1, 0]
stats.probplot(df['pearson_residual'], dist="norm", plot=ax)
ax.set_title('C) Q-Q Plot of Pearson Residuals\n(Should follow diagonal if normally distributed)',
             fontsize=12, fontweight='bold')
ax.set_xlabel('Theoretical Quantiles', fontsize=11, fontweight='bold')
ax.set_ylabel('Sample Quantiles', fontsize=11, fontweight='bold')
ax.grid(True, alpha=0.3)

# Panel D: Residuals vs sample size
ax = axes[1, 1]
colors = ['red' if abs(r) > 2 else 'steelblue' for r in df['pearson_residual']]
ax.scatter(df['n_trials'], df['pearson_residual'], c=colors, s=100, alpha=0.7, edgecolor='black')
ax.axhline(0, color='black', linestyle='-', linewidth=1)
ax.axhline(2, color='red', linestyle='--', linewidth=1, alpha=0.5)
ax.axhline(-2, color='red', linestyle='--', linewidth=1, alpha=0.5)
ax.set_xlabel('Number of Trials', fontsize=11, fontweight='bold')
ax.set_ylabel('Pearson Residual', fontsize=11, fontweight='bold')
ax.set_title('D) Residuals vs Sample Size\n(Checks for heteroscedasticity)', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(VIZ_DIR / 'residual_diagnostics.png', dpi=150, bbox_inches='tight')
plt.close()

print(f"Saved: {VIZ_DIR / 'residual_diagnostics.png'}")

# ============================================================================
# FIGURE 3: Binomial Fit Assessment (Single plot)
# ============================================================================
fig, ax = plt.subplots(figsize=(10, 6))

# Plot observed vs expected successes with error bars
pooled_p = df['r_successes'].sum() / df['n_trials'].sum()

# Error bars for expected (±1 SD under binomial)
ax.errorbar(df['group'], df['expected_successes'],
            yerr=df['expected_std'],
            fmt='o', color='orange', markersize=8, capsize=5, capthick=2,
            alpha=0.7, label='Expected (pooled model)', linewidth=2)

# Observed values
ax.scatter(df['group'], df['r_successes'],
           color='steelblue', s=120, alpha=0.8,
           edgecolor='black', linewidth=1.5, label='Observed', zorder=3)

# Highlight group 1 (zero successes)
group1_data = df[df['group'] == 1]
ax.scatter(group1_data['group'], group1_data['r_successes'],
           color='red', s=200, alpha=0.8, marker='*',
           edgecolor='black', linewidth=1.5, label='Group 1 (0 successes)', zorder=4)

ax.set_xlabel('Group', fontsize=12, fontweight='bold')
ax.set_ylabel('Number of Successes', fontsize=12, fontweight='bold')
ax.set_title(f'Observed vs Expected Successes (Pooled Binomial Model)\nPooled p = {pooled_p:.4f}, Dispersion = {(df["pearson_residual"]**2).sum()/11:.2f}',
             fontsize=13, fontweight='bold')
ax.legend(frameon=True, fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(VIZ_DIR / 'observed_vs_expected.png', dpi=150, bbox_inches='tight')
plt.close()

print(f"Saved: {VIZ_DIR / 'observed_vs_expected.png'}")

# ============================================================================
# FIGURE 4: Variance-Mean Relationship (Single plot)
# ============================================================================
fig, ax = plt.subplots(figsize=(10, 6))

# Calculate empirical variance (using success_rate * (1-success_rate) * n_trials)
df['empirical_variance'] = df['n_trials'] * df['success_rate'] * (1 - df['success_rate'])

# Plot expected vs empirical variance
ax.scatter(df['expected_variance'], df['empirical_variance'],
           s=100, alpha=0.7, color='steelblue', edgecolor='black')

# Add reference line (y=x)
max_var = max(df['expected_variance'].max(), df['empirical_variance'].max())
ax.plot([0, max_var], [0, max_var], 'r--', linewidth=2, alpha=0.7, label='y = x (perfect fit)')

# Add group labels
for idx, row in df.iterrows():
    ax.annotate(f"{int(row['group'])}",
                (row['expected_variance'], row['empirical_variance']),
                xytext=(5, 5), textcoords='offset points', fontsize=9)

ax.set_xlabel('Expected Variance (Pooled Model)', fontsize=12, fontweight='bold')
ax.set_ylabel('Empirical Variance', fontsize=12, fontweight='bold')
ax.set_title('Variance-Mean Relationship: Expected vs Empirical\n(Points above line suggest overdispersion)',
             fontsize=13, fontweight='bold')
ax.legend(frameon=True, fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(VIZ_DIR / 'variance_mean_relationship.png', dpi=150, bbox_inches='tight')
plt.close()

print(f"Saved: {VIZ_DIR / 'variance_mean_relationship.png'}")

# ============================================================================
# FIGURE 5: Sample Size Impact (Multi-panel)
# ============================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Panel A: Standard error by group
df['se_binomial'] = np.sqrt(df['success_rate'] * (1 - df['success_rate']) / df['n_trials'])
ax = axes[0]
colors = ['red' if n < 100 else 'steelblue' for n in df['n_trials']]
ax.bar(df['group'], df['se_binomial'], color=colors, alpha=0.7, edgecolor='black')
ax.set_xlabel('Group', fontsize=11, fontweight='bold')
ax.set_ylabel('Standard Error', fontsize=11, fontweight='bold')
ax.set_title('A) Binomial Standard Error by Group\n(Red = n < 100, smaller samples have larger SE)',
             fontsize=12, fontweight='bold')
ax.grid(axis='y', alpha=0.3)

# Panel B: Precision (1/SE) vs sample size
ax = axes[1]
df['precision'] = 1 / df['se_binomial']
ax.scatter(df['n_trials'], df['precision'], s=100, alpha=0.7,
           color='steelblue', edgecolor='black')

# Add group labels
for idx, row in df.iterrows():
    ax.annotate(f"{int(row['group'])}",
                (row['n_trials'], row['precision']),
                xytext=(5, 5), textcoords='offset points', fontsize=9)

ax.set_xlabel('Number of Trials', fontsize=11, fontweight='bold')
ax.set_ylabel('Precision (1/SE)', fontsize=11, fontweight='bold')
ax.set_title('B) Precision vs Sample Size\n(Larger samples = higher precision)',
             fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(VIZ_DIR / 'sample_size_impact.png', dpi=150, bbox_inches='tight')
plt.close()

print(f"Saved: {VIZ_DIR / 'sample_size_impact.png'}")

# ============================================================================
# FIGURE 6: Transformation Comparison (Multi-panel)
# ============================================================================
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Calculate different transformations (avoiding log(0))
df['logit'] = np.log((df['success_rate'] + 0.001) / (1 - df['success_rate'] + 0.001))
df['probit'] = stats.norm.ppf(df['success_rate'].clip(0.001, 0.999))
df['cloglog'] = np.log(-np.log(1 - df['success_rate'].clip(0.001, 0.999)))

# Panel A: Raw success rates
ax = axes[0, 0]
ax.hist(df['success_rate'], bins=8, color='steelblue', alpha=0.7, edgecolor='black')
ax.set_xlabel('Success Rate', fontsize=11, fontweight='bold')
ax.set_ylabel('Frequency', fontsize=11, fontweight='bold')
ax.set_title('A) Raw Success Rates\n(Original scale)', fontsize=12, fontweight='bold')
ax.grid(axis='y', alpha=0.3)

# Panel B: Logit transformation
ax = axes[0, 1]
ax.hist(df['logit'], bins=8, color='coral', alpha=0.7, edgecolor='black')
ax.set_xlabel('Logit(Success Rate)', fontsize=11, fontweight='bold')
ax.set_ylabel('Frequency', fontsize=11, fontweight='bold')
ax.set_title('B) Logit Transformation\n(log(p/(1-p)))', fontsize=12, fontweight='bold')
ax.grid(axis='y', alpha=0.3)

# Panel C: Probit transformation
ax = axes[1, 0]
ax.hist(df['probit'], bins=8, color='lightgreen', alpha=0.7, edgecolor='black')
ax.set_xlabel('Probit(Success Rate)', fontsize=11, fontweight='bold')
ax.set_ylabel('Frequency', fontsize=11, fontweight='bold')
ax.set_title('C) Probit Transformation\n(Φ⁻¹(p))', fontsize=12, fontweight='bold')
ax.grid(axis='y', alpha=0.3)

# Panel D: Complementary log-log transformation
ax = axes[1, 1]
ax.hist(df['cloglog'], bins=8, color='plum', alpha=0.7, edgecolor='black')
ax.set_xlabel('Cloglog(Success Rate)', fontsize=11, fontweight='bold')
ax.set_ylabel('Frequency', fontsize=11, fontweight='bold')
ax.set_title('D) Complementary Log-Log\n(log(-log(1-p)))', fontsize=12, fontweight='bold')
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(VIZ_DIR / 'transformation_comparison.png', dpi=150, bbox_inches='tight')
plt.close()

print(f"Saved: {VIZ_DIR / 'transformation_comparison.png'}")

print("\n" + "="*80)
print("All diagnostic plots created successfully!")
print("="*80)
