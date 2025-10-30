"""
Visualizations for Clustering Analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist, squareform
import warnings
warnings.filterwarnings('ignore')

# Load data
df = pd.read_csv('/workspace/eda/analyst_2/code/processed_data_with_clusters.csv')

# Standardize for visualization
features = ['n_trials', 'success_rate']
X = df[features].values
X_scaled = (X - X.mean(axis=0)) / X.std(axis=0)

sns.set_style("whitegrid")

# ============================================================================
# FIGURE 1: Hierarchical Clustering Dendrogram
# ============================================================================

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Hierarchical Clustering Analysis', fontsize=16, fontweight='bold', y=0.995)

# Compute linkage for different methods
methods = ['ward', 'complete', 'average', 'single']
colors_list = ['steelblue', 'coral', 'mediumseagreen', 'purple']

for idx, (method, color) in enumerate(zip(methods, colors_list)):
    ax = axes[idx // 2, idx % 2]

    Z = linkage(X_scaled, method=method)

    dendrogram(Z, labels=df['group_id'].values, ax=ax,
               color_threshold=0, above_threshold_color='black')

    ax.set_xlabel('Group ID', fontsize=11, fontweight='bold')
    ax.set_ylabel('Distance', fontsize=11, fontweight='bold')
    ax.set_title(f'{method.capitalize()} Linkage', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    # Add horizontal line for 2-cluster cutoff
    if method == 'ward':
        cutoff_2 = Z[-2, 2]
        ax.axhline(y=cutoff_2, color='red', linestyle='--', linewidth=2,
                   label=f'2 clusters cutoff ({cutoff_2:.2f})', alpha=0.7)
        ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig('/workspace/eda/analyst_2/visualizations/hierarchical_clustering.png', dpi=300, bbox_inches='tight')
print("Saved: hierarchical_clustering.png")
plt.close()

# ============================================================================
# FIGURE 2: K-means Clustering Results
# ============================================================================

fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
fig.suptitle('K-means Clustering Solutions', fontsize=16, fontweight='bold', y=0.995)

# K=2
ax1 = fig.add_subplot(gs[0, 0])
colors_k2 = ['steelblue', 'coral']
for i in range(2):
    cluster_data = df[df['cluster_k2'] == i]
    ax1.scatter(cluster_data['n_trials'], cluster_data['success_rate'],
               s=150, c=colors_k2[i], alpha=0.7, edgecolors='black', linewidth=2,
               label=f'Cluster {i} (n={len(cluster_data)})')

    # Add group labels
    for _, row in cluster_data.iterrows():
        ax1.annotate(int(row['group_id']),
                    (row['n_trials'], row['success_rate']),
                    fontsize=8, ha='center', va='center', fontweight='bold')

ax1.set_xlabel('Number of Trials', fontsize=11)
ax1.set_ylabel('Success Rate', fontsize=11)
ax1.set_title('A. K=2 Clusters', fontsize=12, fontweight='bold')
ax1.legend(fontsize=9, loc='upper right')
ax1.grid(True, alpha=0.3)

# K=3
ax2 = fig.add_subplot(gs[0, 1])
colors_k3 = ['steelblue', 'coral', 'mediumseagreen']
for i in range(3):
    cluster_data = df[df['cluster_k3'] == i]
    ax2.scatter(cluster_data['n_trials'], cluster_data['success_rate'],
               s=150, c=colors_k3[i], alpha=0.7, edgecolors='black', linewidth=2,
               label=f'Cluster {i} (n={len(cluster_data)})')

    for _, row in cluster_data.iterrows():
        ax2.annotate(int(row['group_id']),
                    (row['n_trials'], row['success_rate']),
                    fontsize=8, ha='center', va='center', fontweight='bold')

ax2.set_xlabel('Number of Trials', fontsize=11)
ax2.set_ylabel('Success Rate', fontsize=11)
ax2.set_title('B. K=3 Clusters', fontsize=12, fontweight='bold')
ax2.legend(fontsize=9, loc='upper right')
ax2.grid(True, alpha=0.3)

# K=4
ax3 = fig.add_subplot(gs[0, 2])
colors_k4 = ['steelblue', 'coral', 'mediumseagreen', 'purple']
for i in range(4):
    cluster_data = df[df['cluster_k4'] == i]
    ax3.scatter(cluster_data['n_trials'], cluster_data['success_rate'],
               s=150, c=colors_k4[i], alpha=0.7, edgecolors='black', linewidth=2,
               label=f'Cluster {i} (n={len(cluster_data)})')

    for _, row in cluster_data.iterrows():
        ax3.annotate(int(row['group_id']),
                    (row['n_trials'], row['success_rate']),
                    fontsize=8, ha='center', va='center', fontweight='bold')

ax3.set_xlabel('Number of Trials', fontsize=11)
ax3.set_ylabel('Success Rate', fontsize=11)
ax3.set_title('C. K=4 Clusters', fontsize=12, fontweight='bold')
ax3.legend(fontsize=9, loc='upper right')
ax3.grid(True, alpha=0.3)

# Cluster characteristics for K=3 (most interpretable)
ax4 = fig.add_subplot(gs[1, :])
cluster_stats = []
for i in range(3):
    cluster_data = df[df['cluster_k3'] == i]
    cluster_stats.append({
        'Cluster': i,
        'Size': len(cluster_data),
        'Groups': str(cluster_data['group_id'].values),
        'Mean n_trials': cluster_data['n_trials'].mean(),
        'Mean success_rate': cluster_data['success_rate'].mean(),
        'Std n_trials': cluster_data['n_trials'].std(),
        'Std success_rate': cluster_data['success_rate'].std()
    })

cluster_df = pd.DataFrame(cluster_stats)

# Create table
table_data = []
for i in range(3):
    row = [
        f"Cluster {i}",
        f"{cluster_df.loc[i, 'Size']}",
        cluster_df.loc[i, 'Groups'],
        f"{cluster_df.loc[i, 'Mean n_trials']:.1f} ± {cluster_df.loc[i, 'Std n_trials']:.1f}",
        f"{cluster_df.loc[i, 'Mean success_rate']:.4f} ± {cluster_df.loc[i, 'Std success_rate']:.4f}"
    ]
    table_data.append(row)

ax4.axis('tight')
ax4.axis('off')

table = ax4.table(cellText=table_data,
                 colLabels=['Cluster', 'Size', 'Group IDs', 'n_trials (mean ± std)', 'success_rate (mean ± std)'],
                 cellLoc='left',
                 loc='center',
                 colWidths=[0.1, 0.07, 0.3, 0.25, 0.28])

table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2)

# Style header
for i in range(5):
    table[(0, i)].set_facecolor('lightblue')
    table[(0, i)].set_text_props(weight='bold')

# Color rows by cluster
for i in range(1, 4):
    table[(i, 0)].set_facecolor(colors_k3[i-1])
    table[(i, 0)].set_alpha(0.3)

ax4.set_title('D. Cluster Characteristics (K=3 Solution)', fontsize=12, fontweight='bold', pad=20)

plt.savefig('/workspace/eda/analyst_2/visualizations/kmeans_clustering.png', dpi=300, bbox_inches='tight')
print("Saved: kmeans_clustering.png")
plt.close()

# ============================================================================
# FIGURE 3: Distance Matrix Heatmap
# ============================================================================

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Pairwise Distance Analysis', fontsize=16, fontweight='bold')

# Distance heatmap
ax1 = axes[0]
dist_matrix = squareform(pdist(X_scaled, metric='euclidean'))

im = ax1.imshow(dist_matrix, cmap='YlOrRd', aspect='auto')
ax1.set_xticks(range(12))
ax1.set_yticks(range(12))
ax1.set_xticklabels(df['group_id'].values, fontsize=10)
ax1.set_yticklabels(df['group_id'].values, fontsize=10)
ax1.set_xlabel('Group ID', fontsize=11, fontweight='bold')
ax1.set_ylabel('Group ID', fontsize=11, fontweight='bold')
ax1.set_title('A. Pairwise Distance Matrix', fontsize=13, fontweight='bold')

# Add colorbar
cbar = plt.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
cbar.set_label('Euclidean Distance', fontsize=10)

# Distance distribution
ax2 = axes[1]
triu_indices = np.triu_indices_from(dist_matrix, k=1)
triu_distances = dist_matrix[triu_indices]

ax2.hist(triu_distances, bins=15, color='steelblue', alpha=0.7, edgecolor='black', linewidth=1.5)
ax2.axvline(triu_distances.mean(), color='red', linestyle='--', linewidth=2,
           label=f'Mean = {triu_distances.mean():.2f}')
ax2.axvline(np.median(triu_distances), color='green', linestyle=':', linewidth=2,
           label=f'Median = {np.median(triu_distances):.2f}')

ax2.set_xlabel('Pairwise Distance', fontsize=11, fontweight='bold')
ax2.set_ylabel('Frequency', fontsize=11, fontweight='bold')
ax2.set_title('B. Distribution of Pairwise Distances', fontsize=13, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3, axis='y')

# Add statistics text
stats_text = f'Min: {triu_distances.min():.3f}\nMax: {triu_distances.max():.3f}\nStd: {triu_distances.std():.3f}'
ax2.text(0.7, 0.95, stats_text, transform=ax2.transAxes,
        fontsize=10, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

plt.tight_layout()
plt.savefig('/workspace/eda/analyst_2/visualizations/distance_matrix.png', dpi=300, bbox_inches='tight')
print("Saved: distance_matrix.png")
plt.close()

# ============================================================================
# FIGURE 4: Cluster Comparison (K=3)
# ============================================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Detailed Cluster Analysis (K=3 Solution)', fontsize=16, fontweight='bold', y=0.995)

# Panel 1: Boxplot of n_trials by cluster
ax1 = axes[0, 0]
cluster_data_list = [df[df['cluster_k3'] == i]['n_trials'].values for i in range(3)]
bp1 = ax1.boxplot(cluster_data_list, labels=['Cluster 0', 'Cluster 1', 'Cluster 2'],
                  patch_artist=True, medianprops=dict(color='red', linewidth=2))
for patch, color in zip(bp1['boxes'], colors_k3):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)

ax1.set_ylabel('Number of Trials', fontsize=11, fontweight='bold')
ax1.set_title('A. n_trials by Cluster', fontsize=12, fontweight='bold')
ax1.grid(True, alpha=0.3, axis='y')

# Panel 2: Boxplot of success_rate by cluster
ax2 = axes[0, 1]
cluster_data_list2 = [df[df['cluster_k3'] == i]['success_rate'].values for i in range(3)]
bp2 = ax2.boxplot(cluster_data_list2, labels=['Cluster 0', 'Cluster 1', 'Cluster 2'],
                  patch_artist=True, medianprops=dict(color='red', linewidth=2))
for patch, color in zip(bp2['boxes'], colors_k3):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)

ax2.set_ylabel('Success Rate', fontsize=11, fontweight='bold')
ax2.set_title('B. success_rate by Cluster', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3, axis='y')

# Panel 3: Cluster sizes
ax3 = axes[1, 0]
cluster_sizes = [len(df[df['cluster_k3'] == i]) for i in range(3)]
bars = ax3.bar(range(3), cluster_sizes, color=colors_k3, alpha=0.7, edgecolor='black', linewidth=2)
ax3.set_xticks(range(3))
ax3.set_xticklabels(['Cluster 0', 'Cluster 1', 'Cluster 2'])
ax3.set_ylabel('Number of Groups', fontsize=11, fontweight='bold')
ax3.set_title('C. Cluster Sizes', fontsize=12, fontweight='bold')
ax3.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for i, (bar, size) in enumerate(zip(bars, cluster_sizes)):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
            str(size), ha='center', va='bottom', fontsize=12, fontweight='bold')

# Panel 4: Interpretation text
ax4 = axes[1, 1]
ax4.axis('off')

interpretation = """
CLUSTER INTERPRETATIONS (K=3):

Cluster 0 (Large, Low Success):
  - Groups: 3, 4, 5, 6, 7, 9, 11, 12
  - Characteristics: High n_trials (mean~288)
                     Low success rate (mean~0.065)
  - Profile: Large sample, rare events

Cluster 1 (Small, Very Low Success):
  - Group: 10 only
  - Characteristics: Low n_trials (97)
                     Very low success rate (0.031)
  - Profile: Smallest sample, rarest events

Cluster 2 (Medium, High Success):
  - Groups: 1, 2, 8
  - Characteristics: Medium n_trials (mean~137)
                     High success rate (mean~0.132)
  - Profile: Moderate sample, common events

KEY INSIGHT:
Groups naturally separate based on sample size
and event frequency, suggesting heterogeneous
data generation processes.
"""

ax4.text(0.05, 0.95, interpretation, transform=ax4.transAxes,
        fontsize=10, verticalalignment='top', family='monospace',
        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

plt.tight_layout()
plt.savefig('/workspace/eda/analyst_2/visualizations/cluster_analysis_k3.png', dpi=300, bbox_inches='tight')
print("Saved: cluster_analysis_k3.png")
plt.close()

print("\nClustering visualization generation complete!")
