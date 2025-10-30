"""
Clustering and Heterogeneity Analysis
Testing competing hypotheses:
H1: Groups cluster into distinct subpopulations (e.g., high/low performers)
H2: Groups are drawn from a continuous distribution (hierarchical)
H3: Groups are homogeneous (all from same population)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, linkage

# Setup
BASE_DIR = Path("/workspace/eda/analyst_2")
VIZ_DIR = BASE_DIR / "visualizations"
df = pd.read_csv(BASE_DIR / "code" / "hierarchical_analysis.csv")

pooled_rate = df['r_successes'].sum() / df['n_trials'].sum()

print("=" * 80)
print("CLUSTERING AND HETEROGENEITY ANALYSIS")
print("=" * 80)

# ============================================================================
# 1. Visual inspection: Histogram and density
# ============================================================================
print("\n" + "=" * 80)
print("1. DISTRIBUTION OF SUCCESS RATES")
print("=" * 80)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Top left: Histogram
ax1 = axes[0, 0]
ax1.hist(df['success_rate'], bins=10, alpha=0.7, color='steelblue', edgecolor='black')
ax1.axvline(pooled_rate, color='red', linestyle='--', linewidth=2, label=f'Pooled rate')
ax1.axvline(df['success_rate'].mean(), color='orange', linestyle='--', linewidth=2, label='Mean')
ax1.set_xlabel('Success Rate')
ax1.set_ylabel('Frequency')
ax1.set_title('Distribution of Group Success Rates', fontweight='bold')
ax1.legend()
ax1.grid(alpha=0.3)

# Top right: KDE with rug plot
ax2 = axes[0, 1]
kde = stats.gaussian_kde(df['success_rate'])
x_range = np.linspace(df['success_rate'].min() - 0.02, df['success_rate'].max() + 0.02, 200)
ax2.plot(x_range, kde(x_range), linewidth=2, color='darkblue')
ax2.fill_between(x_range, kde(x_range), alpha=0.3, color='steelblue')
ax2.scatter(df['success_rate'], np.zeros(len(df)), alpha=0.7, s=100, color='red', marker='|')
ax2.axvline(pooled_rate, color='red', linestyle='--', linewidth=2, alpha=0.5)
ax2.set_xlabel('Success Rate')
ax2.set_ylabel('Density')
ax2.set_title('Kernel Density Estimate with Rug Plot', fontweight='bold')
ax2.grid(alpha=0.3)

# Bottom left: Q-Q plot (test for normality)
ax3 = axes[1, 0]
stats.probplot(df['success_rate'], dist="norm", plot=ax3)
ax3.set_title('Q-Q Plot (Test for Normality)', fontweight='bold')
ax3.grid(alpha=0.3)

# Bottom right: Weighted by sample size
ax4 = axes[1, 1]
weights = df['n_trials'] / df['n_trials'].sum()
ax4.bar(range(len(df)), df['success_rate'], alpha=0.7, color='steelblue')
ax4.axhline(pooled_rate, color='red', linestyle='--', linewidth=2, label='Pooled rate')
ax4.set_xlabel('Group')
ax4.set_ylabel('Success Rate')
ax4.set_title('Success Rates by Group (bar height = rate)', fontweight='bold')
ax4.set_xticks(range(len(df)))
ax4.set_xticklabels([f"G{int(g)}" for g in df['group']], rotation=45)
ax4.legend()
ax4.grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(VIZ_DIR / "distribution_analysis.png", dpi=150, bbox_inches='tight')
print(f"\nSaved: {VIZ_DIR / 'distribution_analysis.png'}")
plt.close()

# Test for normality
shapiro_stat, shapiro_p = stats.shapiro(df['success_rate'])
print(f"\nShapiro-Wilk test for normality:")
print(f"  Statistic: {shapiro_stat:.4f}")
print(f"  P-value: {shapiro_p:.4f}")
if shapiro_p > 0.05:
    print("  → Cannot reject normality (p > 0.05)")
    print("  → Success rates could plausibly come from normal distribution")
else:
    print("  → Reject normality (p < 0.05)")
    print("  → Distribution is non-normal")

# ============================================================================
# 2. Simple clustering analysis (manual k-means-like approach)
# ============================================================================
print("\n" + "=" * 80)
print("2. SIMPLE CLUSTERING ANALYSIS")
print("=" * 80)
print("\nTesting if groups naturally cluster into 2 subpopulations...")

# Split at median
median_rate = df['success_rate'].median()
df['cluster_2'] = (df['success_rate'] > median_rate).astype(int)

print(f"\n2-cluster solution (split at median = {median_rate:.4f}):")
for cluster in range(2):
    cluster_data = df[df['cluster_2'] == cluster]
    print(f"\n  Cluster {cluster}: {len(cluster_data)} groups")
    print(f"    Groups: {cluster_data['group'].values}")
    print(f"    Mean success rate: {cluster_data['success_rate'].mean():.4f}")
    print(f"    Std dev: {cluster_data['success_rate'].std():.4f}")
    print(f"    Range: [{cluster_data['success_rate'].min():.4f}, {cluster_data['success_rate'].max():.4f}]")

# Check for natural gap/bimodality
sorted_rates = np.sort(df['success_rate'].values)
gaps = np.diff(sorted_rates)
mean_gap = gaps.mean()
std_gap = gaps.std()

# Look for unusually large gaps
print(f"\nGap analysis:")
print(f"  Mean gap: {mean_gap:.4f}")
print(f"  Std gap: {std_gap:.4f}")
print(f"  Max gap: {gaps.max():.4f}")

large_gaps = gaps > (mean_gap + 1.5 * std_gap)
if large_gaps.any():
    print(f"\n  Large gaps detected at:")
    for i, is_large in enumerate(large_gaps):
        if is_large:
            print(f"    Between {sorted_rates[i]:.4f} and {sorted_rates[i+1]:.4f} (gap={gaps[i]:.4f})")
else:
    print(f"\n  No unusually large gaps detected")
    print(f"  → Distribution appears continuous, not clustered")

# ============================================================================
# 3. Testing for multimodality
# ============================================================================
print("\n" + "=" * 80)
print("3. TESTING FOR MULTIMODALITY")
print("=" * 80)

# Compare variance of gaps
print(f"\nVariance of gaps: {gaps.var():.6f}")
print(f"Mean gap: {gaps.mean():.4f}")
print(f"CV of gaps: {gaps.std() / gaps.mean():.4f}")

# Test against uniform spacing (would indicate discrete clusters)
# Uniform would have constant gaps
if gaps.std() / gaps.mean() < 0.5:
    print("  → Low CV: Gaps are relatively uniform")
    print("  → Suggests continuous distribution")
else:
    print("  → High CV: Gaps are highly variable")
    print("  → Could suggest discrete clusters or outliers")

# ============================================================================
# 4. Visual clustering analysis
# ============================================================================
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Left: 2-cluster visualization
ax1 = axes[0]
colors = ['#1f77b4', '#ff7f0e']
for cluster in range(2):
    cluster_data = df[df['cluster_2'] == cluster]
    ax1.scatter(cluster_data['success_rate'], cluster_data['n_trials'],
               s=200, alpha=0.7, color=colors[cluster], label=f'Cluster {cluster}')
    for _, row in cluster_data.iterrows():
        ax1.annotate(f"G{int(row['group'])}", (row['success_rate'], row['n_trials']),
                    xytext=(5, 5), textcoords='offset points', fontsize=9)

ax1.axvline(pooled_rate, color='red', linestyle='--', linewidth=2, alpha=0.5, label='Pooled rate')
ax1.axvline(median_rate, color='green', linestyle=':', linewidth=2, alpha=0.5, label='Median (split)')
ax1.set_xlabel('Success Rate', fontsize=12)
ax1.set_ylabel('Sample Size (n_trials)', fontsize=12)
ax1.set_title('Simple 2-Cluster Split (at median)', fontweight='bold', fontsize=13)
ax1.legend()
ax1.grid(alpha=0.3)

# Right: Dendrogram (hierarchical clustering)
ax2 = axes[1]
# Perform hierarchical clustering
X = df[['success_rate']].values
Z = linkage(X, method='ward')
dendrogram(Z, labels=[f"G{int(g)}" for g in df['group']], ax=ax2)
ax2.set_xlabel('Group', fontsize=12)
ax2.set_ylabel('Distance', fontsize=12)
ax2.set_title('Hierarchical Clustering Dendrogram', fontweight='bold', fontsize=13)
ax2.grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(VIZ_DIR / "clustering_analysis.png", dpi=150, bbox_inches='tight')
print(f"\n\nSaved: {VIZ_DIR / 'clustering_analysis.png'}")
plt.close()

# ============================================================================
# 5. Sample size effect analysis
# ============================================================================
print("\n" + "=" * 80)
print("4. SAMPLE SIZE EFFECT ANALYSIS")
print("=" * 80)
print("\nDo smaller groups show more extreme rates? (regression to the mean)")

# Correlation between sample size and distance from pooled rate
df['distance_from_pooled'] = np.abs(df['success_rate'] - pooled_rate)
corr, p_value = stats.spearmanr(df['n_trials'], df['distance_from_pooled'])

print(f"\nSpearman correlation (sample size vs |rate - pooled|):")
print(f"  Correlation: {corr:.4f}")
print(f"  P-value: {p_value:.4f}")

if corr < 0 and p_value < 0.05:
    print("  → Significant NEGATIVE correlation")
    print("  → Smaller samples have more extreme rates (regression to mean)")
    print("  → Supports hierarchical modeling with shrinkage")
elif p_value >= 0.05:
    print("  → No significant correlation")
else:
    print("  → Positive correlation (unexpected)")

# Visualize
fig, ax = plt.subplots(figsize=(10, 7))
ax.scatter(df['n_trials'], df['distance_from_pooled'], s=150, alpha=0.7, color='purple')
for _, row in df.iterrows():
    ax.annotate(f"G{int(row['group'])}", (row['n_trials'], row['distance_from_pooled']),
                xytext=(5, 5), textcoords='offset points', fontsize=9, alpha=0.7)

# Add trend line
z = np.polyfit(df['n_trials'], df['distance_from_pooled'], 1)
p = np.poly1d(z)
x_trend = np.linspace(df['n_trials'].min(), df['n_trials'].max(), 100)
ax.plot(x_trend, p(x_trend), "r--", alpha=0.8, linewidth=2,
        label=f'Linear trend (r={corr:.3f}, p={p_value:.3f})')

ax.set_xlabel('Sample Size (n_trials)', fontsize=12)
ax.set_ylabel('|Success Rate - Pooled Rate|', fontsize=12)
ax.set_title('Regression to the Mean: Sample Size vs Deviation from Pooled Rate',
            fontweight='bold', fontsize=13)
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(VIZ_DIR / "regression_to_mean.png", dpi=150, bbox_inches='tight')
print(f"\nSaved: {VIZ_DIR / 'regression_to_mean.png'}")
plt.close()

# Save final data
df.to_csv(BASE_DIR / "code" / "clustering_analysis.csv", index=False)
print(f"\nSaved: {BASE_DIR / 'code' / 'clustering_analysis.csv'}")
