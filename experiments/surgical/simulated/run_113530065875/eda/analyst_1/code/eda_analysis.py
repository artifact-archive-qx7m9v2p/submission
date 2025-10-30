"""
Comprehensive EDA for Binomial Dataset - Analyst 1
Dataset: 12 groups with binomial data (n trials, r successes)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'

# ============================================================================
# 1. LOAD AND PREPARE DATA
# ============================================================================

data = pd.read_csv('/workspace/data/data_analyst_1.csv')
print("="*80)
print("BINOMIAL DATA - EXPLORATORY DATA ANALYSIS")
print("="*80)
print("\nRaw Data:")
print(data)
print(f"\nShape: {data.shape}")
print(f"Columns: {list(data.columns)}")

# Calculate success rates
data['success_rate'] = data['r'] / data['n']
data['failures'] = data['n'] - data['r']

print("\n" + "="*80)
print("DATA WITH COMPUTED METRICS")
print("="*80)
print(data)

# ============================================================================
# 2. DESCRIPTIVE STATISTICS
# ============================================================================

print("\n" + "="*80)
print("DESCRIPTIVE STATISTICS")
print("="*80)

# Overall statistics
print("\nOverall Pooled Statistics:")
total_trials = data['n'].sum()
total_successes = data['r'].sum()
pooled_rate = total_successes / total_trials
print(f"Total trials: {total_trials}")
print(f"Total successes: {total_successes}")
print(f"Pooled success rate: {pooled_rate:.4f}")

# Success rate statistics
print("\nSuccess Rate Statistics (across groups):")
print(f"Mean: {data['success_rate'].mean():.4f}")
print(f"Median: {data['success_rate'].median():.4f}")
print(f"Std Dev: {data['success_rate'].std():.4f}")
print(f"Min: {data['success_rate'].min():.4f} (Group {data.loc[data['success_rate'].idxmin(), 'group']})")
print(f"Max: {data['success_rate'].max():.4f} (Group {data.loc[data['success_rate'].idxmax(), 'group']})")
print(f"Range: {data['success_rate'].max() - data['success_rate'].min():.4f}")
print(f"IQR: {data['success_rate'].quantile(0.75) - data['success_rate'].quantile(0.25):.4f}")
print(f"CV (Coefficient of Variation): {data['success_rate'].std() / data['success_rate'].mean():.4f}")

# Sample size statistics
print("\nSample Size Statistics:")
print(f"Mean n: {data['n'].mean():.1f}")
print(f"Median n: {data['n'].median():.1f}")
print(f"Min n: {data['n'].min()} (Group {data.loc[data['n'].idxmin(), 'group']})")
print(f"Max n: {data['n'].max()} (Group {data.loc[data['n'].idxmax(), 'group']})")
print(f"Range n: {data['n'].max() - data['n'].min()}")

# Detailed group statistics
print("\nPer-Group Summary:")
summary = data[['group', 'n', 'r', 'success_rate']].copy()
summary['se_binomial'] = np.sqrt(pooled_rate * (1 - pooled_rate) / summary['n'])
summary = summary.sort_values('success_rate')
print(summary.to_string(index=False))

# ============================================================================
# 3. VARIABILITY ANALYSIS (OVERDISPERSION TEST)
# ============================================================================

print("\n" + "="*80)
print("VARIABILITY ANALYSIS: OVERDISPERSION DETECTION")
print("="*80)

# Empirical variance
empirical_var = data['success_rate'].var()
empirical_sd = data['success_rate'].std()

# Theoretical variance under common probability model
# Using pooled estimate for theoretical calculation
# For binomial, Var(r/n) = p(1-p)/n
# Expected variance across groups with different n's
data['theoretical_var'] = pooled_rate * (1 - pooled_rate) / data['n']
expected_mean_var = data['theoretical_var'].mean()

print(f"\nEmpirical variance (across groups): {empirical_var:.6f}")
print(f"Empirical SD: {empirical_sd:.4f}")
print(f"Expected mean variance (binomial): {expected_mean_var:.6f}")
print(f"Expected SD: {np.sqrt(expected_mean_var):.4f}")
print(f"Variance ratio (empirical/expected): {empirical_var / expected_mean_var:.2f}")

# Chi-square test for overdispersion
# If all groups have same p, then sum of (r - n*p_pooled)^2 / (n*p_pooled*(1-p_pooled)) ~ chi-square(k-1)
chi_square_stat = ((data['r'] - data['n'] * pooled_rate)**2 /
                   (data['n'] * pooled_rate * (1 - pooled_rate))).sum()
df_chi = len(data) - 1
p_value_chi = 1 - stats.chi2.cdf(chi_square_stat, df_chi)

print(f"\nChi-square test for homogeneity:")
print(f"Chi-square statistic: {chi_square_stat:.2f}")
print(f"Degrees of freedom: {df_chi}")
print(f"Expected value: {df_chi}")
print(f"P-value: {p_value_chi:.6f}")
if p_value_chi < 0.05:
    print("Result: REJECT null hypothesis - significant heterogeneity detected")
else:
    print("Result: FAIL TO REJECT null hypothesis - consistent with homogeneity")

# Dispersion parameter
dispersion = chi_square_stat / df_chi
print(f"\nDispersion parameter (phi): {dispersion:.2f}")
if dispersion > 1.5:
    print("Interpretation: OVERDISPERSION present (phi > 1.5)")
elif dispersion < 0.7:
    print("Interpretation: UNDERDISPERSION present (phi < 0.7)")
else:
    print("Interpretation: Consistent with binomial variation")

# ============================================================================
# 4. SAMPLE SIZE EFFECTS
# ============================================================================

print("\n" + "="*80)
print("SAMPLE SIZE EFFECTS ANALYSIS")
print("="*80)

# Correlation between n and success rate
corr_n_rate = data['n'].corr(data['success_rate'])
corr_p_value = stats.pearsonr(data['n'], data['success_rate'])[1]

print(f"\nCorrelation between sample size (n) and success rate:")
print(f"Pearson r: {corr_n_rate:.4f}")
print(f"P-value: {corr_p_value:.4f}")

# Spearman rank correlation (non-parametric)
spearman_corr = data['n'].corr(data['success_rate'], method='spearman')
spearman_p = stats.spearmanr(data['n'], data['success_rate'])[1]
print(f"Spearman rho: {spearman_corr:.4f}")
print(f"P-value: {spearman_p:.4f}")

# Split by median sample size
median_n = data['n'].median()
small_groups = data[data['n'] <= median_n]
large_groups = data[data['n'] > median_n]

print(f"\nComparison by sample size (median split at n={median_n:.0f}):")
print(f"Small groups (n <= {median_n:.0f}):")
print(f"  Count: {len(small_groups)}")
print(f"  Mean success rate: {small_groups['success_rate'].mean():.4f}")
print(f"  SD success rate: {small_groups['success_rate'].std():.4f}")

print(f"Large groups (n > {median_n:.0f}):")
print(f"  Count: {len(large_groups)}")
print(f"  Mean success rate: {large_groups['success_rate'].mean():.4f}")
print(f"  SD success rate: {large_groups['success_rate'].std():.4f}")

# T-test for difference in means
t_stat, t_pval = stats.ttest_ind(small_groups['success_rate'],
                                  large_groups['success_rate'])
print(f"\nT-test for difference in means:")
print(f"t-statistic: {t_stat:.3f}")
print(f"P-value: {t_pval:.4f}")

# ============================================================================
# 5. OUTLIER DETECTION
# ============================================================================

print("\n" + "="*80)
print("OUTLIER DETECTION")
print("="*80)

# Z-scores based on empirical distribution
data['z_score'] = (data['success_rate'] - data['success_rate'].mean()) / data['success_rate'].std()

# Standardized residuals (using pooled rate)
data['std_residual'] = (data['r'] - data['n'] * pooled_rate) / np.sqrt(data['n'] * pooled_rate * (1 - pooled_rate))

print("\nGroups ranked by standardized residual:")
outlier_check = data[['group', 'n', 'r', 'success_rate', 'std_residual']].copy()
outlier_check = outlier_check.sort_values('std_residual', key=abs, ascending=False)
print(outlier_check.to_string(index=False))

# Flag outliers
threshold = 2.0
outliers = data[np.abs(data['std_residual']) > threshold]
print(f"\nOutliers (|std_residual| > {threshold}):")
if len(outliers) > 0:
    print(outliers[['group', 'n', 'r', 'success_rate', 'std_residual']].to_string(index=False))
else:
    print("None detected")

# IQR method on success rates
Q1 = data['success_rate'].quantile(0.25)
Q3 = data['success_rate'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

iqr_outliers = data[(data['success_rate'] < lower_bound) | (data['success_rate'] > upper_bound)]
print(f"\nIQR method outliers (success rate outside [{lower_bound:.4f}, {upper_bound:.4f}]):")
if len(iqr_outliers) > 0:
    print(iqr_outliers[['group', 'n', 'r', 'success_rate']].to_string(index=False))
else:
    print("None detected")

# ============================================================================
# 6. EXCHANGEABILITY ASSESSMENT
# ============================================================================

print("\n" + "="*80)
print("EXCHANGEABILITY ASSESSMENT")
print("="*80)

# Test for ordering effects
print("\nTesting for ordering effects (by group number):")
corr_group_rate = data['group'].corr(data['success_rate'])
corr_group_p = stats.pearsonr(data['group'], data['success_rate'])[1]
print(f"Correlation with group number: {corr_group_rate:.4f} (p={corr_group_p:.4f})")

# Runs test for randomness
median_rate = data['success_rate'].median()
runs = (data['success_rate'] > median_rate).astype(int)
runs_diff = runs.diff().fillna(0)
n_runs = (runs_diff != 0).sum() + 1
n_above = (runs == 1).sum()
n_below = (runs == 0).sum()

# Expected runs under randomness
expected_runs = (2 * n_above * n_below) / (n_above + n_below) + 1
var_runs = (2 * n_above * n_below * (2 * n_above * n_below - n_above - n_below)) / ((n_above + n_below)**2 * (n_above + n_below - 1))
z_runs = (n_runs - expected_runs) / np.sqrt(var_runs) if var_runs > 0 else 0

print(f"\nRuns test (above/below median):")
print(f"Number of runs: {n_runs}")
print(f"Expected runs: {expected_runs:.1f}")
print(f"Z-score: {z_runs:.2f}")
print(f"Interpretation: {'Random order' if abs(z_runs) < 1.96 else 'Non-random pattern detected'}")

# Levene's test for homogeneity of variance
# Split into thirds
data['tercile'] = pd.qcut(data['n'], q=3, labels=['small', 'medium', 'large'])
groups_for_levene = [data[data['tercile'] == t]['success_rate'].values for t in ['small', 'medium', 'large']]
levene_stat, levene_p = stats.levene(*groups_for_levene)
print(f"\nLevene's test for variance homogeneity (by sample size tercile):")
print(f"Statistic: {levene_stat:.3f}")
print(f"P-value: {levene_p:.4f}")

# ============================================================================
# 7. VISUALIZATIONS
# ============================================================================

print("\n" + "="*80)
print("GENERATING VISUALIZATIONS")
print("="*80)

# Visualization 1: Success rate by group with error bars
fig, ax = plt.subplots(figsize=(12, 6))
x_pos = np.arange(len(data))

# Error bars based on binomial SE
se_binomial = np.sqrt(pooled_rate * (1 - pooled_rate) / data['n'])
bars = ax.bar(x_pos, data['success_rate'], yerr=se_binomial,
              capsize=5, alpha=0.7, edgecolor='black', linewidth=1.5)

# Color by outlier status
colors = ['red' if abs(r) > 2 else 'steelblue' for r in data['std_residual']]
for bar, color in zip(bars, colors):
    bar.set_facecolor(color)

ax.axhline(pooled_rate, color='darkgreen', linestyle='--', linewidth=2, label=f'Pooled rate: {pooled_rate:.3f}')
ax.set_xlabel('Group', fontsize=12, fontweight='bold')
ax.set_ylabel('Success Rate (r/n)', fontsize=12, fontweight='bold')
ax.set_title('Success Rate by Group with Binomial Standard Errors', fontsize=14, fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels(data['group'])
ax.legend()
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('/workspace/eda/analyst_1/visualizations/success_rate_by_group.png')
print("Saved: success_rate_by_group.png")
plt.close()

# Visualization 2: Success rate vs sample size scatter plot
fig, ax = plt.subplots(figsize=(10, 6))
scatter = ax.scatter(data['n'], data['success_rate'], s=100, alpha=0.7,
                     c=data['std_residual'], cmap='RdYlBu_r',
                     edgecolors='black', linewidth=1.5)
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('Standardized Residual', rotation=270, labelpad=20, fontweight='bold')

# Add group labels
for idx, row in data.iterrows():
    ax.annotate(f"{row['group']}", (row['n'], row['success_rate']),
                xytext=(5, 5), textcoords='offset points', fontsize=9)

# Add pooled rate line
ax.axhline(pooled_rate, color='darkgreen', linestyle='--', linewidth=2,
           label=f'Pooled rate: {pooled_rate:.3f}')

# Add regression line
z = np.polyfit(data['n'], data['success_rate'], 1)
p = np.poly1d(z)
x_line = np.linspace(data['n'].min(), data['n'].max(), 100)
ax.plot(x_line, p(x_line), "r--", alpha=0.5, linewidth=2,
        label=f'Trend line (r={corr_n_rate:.3f})')

ax.set_xlabel('Sample Size (n)', fontsize=12, fontweight='bold')
ax.set_ylabel('Success Rate (r/n)', fontsize=12, fontweight='bold')
ax.set_title('Success Rate vs Sample Size', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('/workspace/eda/analyst_1/visualizations/success_rate_vs_sample_size.png')
print("Saved: success_rate_vs_sample_size.png")
plt.close()

# Visualization 3: Distribution of success rates
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Histogram
ax = axes[0]
ax.hist(data['success_rate'], bins=10, edgecolor='black', alpha=0.7, color='steelblue')
ax.axvline(data['success_rate'].mean(), color='red', linestyle='--',
           linewidth=2, label=f'Mean: {data["success_rate"].mean():.3f}')
ax.axvline(data['success_rate'].median(), color='orange', linestyle='--',
           linewidth=2, label=f'Median: {data["success_rate"].median():.3f}')
ax.axvline(pooled_rate, color='darkgreen', linestyle='--',
           linewidth=2, label=f'Pooled: {pooled_rate:.3f}')
ax.set_xlabel('Success Rate', fontsize=12, fontweight='bold')
ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
ax.set_title('Distribution of Success Rates', fontsize=13, fontweight='bold')
ax.legend()
ax.grid(axis='y', alpha=0.3)

# Box plot
ax = axes[1]
bp = ax.boxplot([data['success_rate']], vert=True, patch_artist=True,
                labels=['All Groups'])
bp['boxes'][0].set_facecolor('lightblue')
bp['boxes'][0].set_edgecolor('black')
bp['medians'][0].set_color('red')
bp['medians'][0].set_linewidth(2)

ax.axhline(pooled_rate, color='darkgreen', linestyle='--',
           linewidth=2, label=f'Pooled rate: {pooled_rate:.3f}')
ax.set_ylabel('Success Rate', fontsize=12, fontweight='bold')
ax.set_title('Box Plot of Success Rates', fontsize=13, fontweight='bold')
ax.legend()
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('/workspace/eda/analyst_1/visualizations/success_rate_distribution.png')
print("Saved: success_rate_distribution.png")
plt.close()

# Visualization 4: Funnel plot
fig, ax = plt.subplots(figsize=(10, 6))

# Calculate precision (1/sqrt(n))
data['precision'] = 1 / np.sqrt(data['n'])

# Plot points
scatter = ax.scatter(data['precision'], data['success_rate'], s=100, alpha=0.7,
                     c=data['std_residual'], cmap='RdYlBu_r',
                     edgecolors='black', linewidth=1.5)
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('Standardized Residual', rotation=270, labelpad=20, fontweight='bold')

# Add group labels
for idx, row in data.iterrows():
    ax.annotate(f"{row['group']}", (row['precision'], row['success_rate']),
                xytext=(5, 5), textcoords='offset points', fontsize=9)

# Add pooled rate and confidence limits
ax.axhline(pooled_rate, color='darkgreen', linestyle='-', linewidth=2,
           label=f'Pooled rate: {pooled_rate:.3f}')

# 95% confidence limits (approximate)
x_funnel = np.linspace(0, data['precision'].max() * 1.1, 100)
n_funnel = 1 / x_funnel**2
se_funnel = np.sqrt(pooled_rate * (1 - pooled_rate) / n_funnel)
upper_95 = pooled_rate + 1.96 * se_funnel
lower_95 = pooled_rate - 1.96 * se_funnel

ax.plot(x_funnel, upper_95, 'r--', linewidth=1.5, alpha=0.7, label='95% CI')
ax.plot(x_funnel, lower_95, 'r--', linewidth=1.5, alpha=0.7)

# 99.8% limits (3 sigma)
upper_99 = pooled_rate + 3 * se_funnel
lower_99 = pooled_rate - 3 * se_funnel
ax.plot(x_funnel, upper_99, 'orange', linestyle=':', linewidth=1.5, alpha=0.7, label='99.8% CI')
ax.plot(x_funnel, lower_99, 'orange', linestyle=':', linewidth=1.5, alpha=0.7)

ax.set_xlabel('Precision (1/√n)', fontsize=12, fontweight='bold')
ax.set_ylabel('Success Rate', fontsize=12, fontweight='bold')
ax.set_title('Funnel Plot: Success Rate vs Precision', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('/workspace/eda/analyst_1/visualizations/funnel_plot.png')
print("Saved: funnel_plot.png")
plt.close()

# Visualization 5: Multi-panel diagnostic plot
fig = plt.figure(figsize=(14, 10))
gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

# Panel 1: Residual plot
ax1 = fig.add_subplot(gs[0, 0])
ax1.scatter(data['group'], data['std_residual'], s=80, alpha=0.7,
            edgecolors='black', linewidth=1.5)
ax1.axhline(0, color='black', linestyle='-', linewidth=1)
ax1.axhline(2, color='red', linestyle='--', linewidth=1, alpha=0.5)
ax1.axhline(-2, color='red', linestyle='--', linewidth=1, alpha=0.5)
ax1.set_xlabel('Group', fontweight='bold')
ax1.set_ylabel('Standardized Residual', fontweight='bold')
ax1.set_title('A) Standardized Residuals by Group', fontweight='bold')
ax1.grid(alpha=0.3)

# Panel 2: Q-Q plot for success rates
ax2 = fig.add_subplot(gs[0, 1])
stats.probplot(data['success_rate'], dist="norm", plot=ax2)
ax2.set_title('B) Q-Q Plot: Success Rates vs Normal', fontweight='bold')
ax2.grid(alpha=0.3)

# Panel 3: Sample size distribution
ax3 = fig.add_subplot(gs[1, 0])
ax3.bar(data['group'], data['n'], alpha=0.7, edgecolor='black', linewidth=1.5)
ax3.axhline(data['n'].mean(), color='red', linestyle='--', linewidth=2,
            label=f'Mean: {data["n"].mean():.0f}')
ax3.set_xlabel('Group', fontweight='bold')
ax3.set_ylabel('Sample Size (n)', fontweight='bold')
ax3.set_title('C) Sample Size by Group', fontweight='bold')
ax3.legend()
ax3.grid(axis='y', alpha=0.3)

# Panel 4: Success count vs expected
ax4 = fig.add_subplot(gs[1, 1])
expected = data['n'] * pooled_rate
ax4.scatter(expected, data['r'], s=80, alpha=0.7, edgecolors='black', linewidth=1.5)
lim = [0, max(expected.max(), data['r'].max()) * 1.1]
ax4.plot(lim, lim, 'r--', linewidth=2, label='y=x (perfect match)')
ax4.set_xlabel('Expected Successes (n × pooled rate)', fontweight='bold')
ax4.set_ylabel('Observed Successes (r)', fontweight='bold')
ax4.set_title('D) Observed vs Expected Successes', fontweight='bold')
ax4.legend()
ax4.grid(alpha=0.3)

# Panel 5: Variance comparison
ax5 = fig.add_subplot(gs[2, 0])
variance_data = pd.DataFrame({
    'Type': ['Empirical\n(observed)', 'Theoretical\n(binomial)'],
    'Variance': [empirical_var, expected_mean_var]
})
bars = ax5.bar(variance_data['Type'], variance_data['Variance'],
               alpha=0.7, edgecolor='black', linewidth=1.5,
               color=['steelblue', 'orange'])
ax5.set_ylabel('Variance', fontweight='bold')
ax5.set_title('E) Empirical vs Theoretical Variance', fontweight='bold')
ax5.grid(axis='y', alpha=0.3)
# Add values on bars
for bar, val in zip(bars, variance_data['Variance']):
    height = bar.get_height()
    ax5.text(bar.get_x() + bar.get_width()/2., height,
            f'{val:.6f}', ha='center', va='bottom', fontweight='bold')

# Panel 6: Cook's distance analog (influence)
ax6 = fig.add_subplot(gs[2, 1])
# Influence measure: leverage × residual^2
influence = (data['n'] / data['n'].sum()) * data['std_residual']**2
ax6.bar(data['group'], influence, alpha=0.7, edgecolor='black', linewidth=1.5)
ax6.set_xlabel('Group', fontweight='bold')
ax6.set_ylabel('Influence Measure', fontweight='bold')
ax6.set_title('F) Group Influence on Pooled Estimate', fontweight='bold')
ax6.grid(axis='y', alpha=0.3)

plt.suptitle('Diagnostic Plots for Binomial Data', fontsize=16, fontweight='bold', y=0.995)
plt.savefig('/workspace/eda/analyst_1/visualizations/diagnostic_panel.png')
print("Saved: diagnostic_panel.png")
plt.close()

# ============================================================================
# 8. SUMMARY STATISTICS TABLE
# ============================================================================

print("\n" + "="*80)
print("FINAL SUMMARY TABLE")
print("="*80)

summary_table = data[['group', 'n', 'r', 'success_rate', 'std_residual']].copy()
summary_table['se_binomial'] = np.sqrt(pooled_rate * (1 - pooled_rate) / summary_table['n'])
summary_table = summary_table.sort_values('group')
summary_table.columns = ['Group', 'Trials', 'Successes', 'Rate', 'Std_Residual', 'SE']
print(summary_table.to_string(index=False))

# Save data with calculations
data.to_csv('/workspace/eda/analyst_1/code/data_with_calculations.csv', index=False)
print("\nData with calculations saved to: data_with_calculations.csv")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
