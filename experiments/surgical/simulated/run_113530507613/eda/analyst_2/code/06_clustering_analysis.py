"""
Clustering Analysis
Focus: Do groups cluster into distinct subpopulations?
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist, squareform
from scipy.stats import mannwhitneyu, ttest_ind
import warnings
warnings.filterwarnings('ignore')

# Load data
df = pd.read_csv('/workspace/eda/analyst_2/code/processed_data.csv')

print("="*80)
print("CLUSTERING ANALYSIS")
print("="*80)

# 1. PREPARE DATA FOR CLUSTERING
print("\n1. DATA PREPARATION")
print("-"*80)

# Select features for clustering
features = ['n_trials', 'success_rate']
X = df[features].values

print(f"Features for clustering: {features}")
print(f"Data shape: {X.shape}")

# Standardize features manually
def standardize(data):
    mean = data.mean(axis=0)
    std = data.std(axis=0)
    return (data - mean) / std, mean, std

X_scaled, mean_vals, std_vals = standardize(X)

print(f"\nOriginal data statistics:")
print(df[features].describe())

print(f"\nScaled data statistics:")
print(f"Mean: {X_scaled.mean(axis=0)}")
print(f"Std: {X_scaled.std(axis=0)}")

# 2. HIERARCHICAL CLUSTERING
print("\n2. HIERARCHICAL CLUSTERING")
print("-"*80)

# Calculate linkage for different methods
linkage_methods = ['ward', 'complete', 'average', 'single']
linkage_results = {}

for method in linkage_methods:
    Z = linkage(X_scaled, method=method)
    linkage_results[method] = Z
    print(f"\n{method.capitalize()} linkage:")
    print(f"  Last merge distance: {Z[-1, 2]:.4f}")

# Use Ward's method as primary
Z_ward = linkage_results['ward']

# Cut tree at different heights to get different numbers of clusters
for n_clusters in [2, 3, 4]:
    clusters = fcluster(Z_ward, n_clusters, criterion='maxclust')
    print(f"\n{n_clusters} clusters (Ward's method):")
    for i in range(1, n_clusters + 1):
        members = df.loc[clusters == i, 'group_id'].values
        print(f"  Cluster {i}: Groups {members}")

# Save 2-cluster and 3-cluster solutions
df['cluster_h2'] = fcluster(Z_ward, 2, criterion='maxclust')
df['cluster_h3'] = fcluster(Z_ward, 3, criterion='maxclust')

# 3. SIMPLE K-MEANS IMPLEMENTATION
print("\n3. K-MEANS CLUSTERING (Manual Implementation)")
print("-"*80)

def simple_kmeans(X, k, max_iter=100, random_seed=42):
    """Simple k-means implementation"""
    np.random.seed(random_seed)

    # Initialize centroids randomly
    n_samples = X.shape[0]
    random_indices = np.random.choice(n_samples, k, replace=False)
    centroids = X[random_indices].copy()

    for iteration in range(max_iter):
        # Assign points to nearest centroid
        distances = np.zeros((n_samples, k))
        for i in range(k):
            distances[:, i] = np.sqrt(np.sum((X - centroids[i])**2, axis=1))

        labels = np.argmin(distances, axis=1)

        # Update centroids
        new_centroids = np.zeros_like(centroids)
        for i in range(k):
            if np.sum(labels == i) > 0:
                new_centroids[i] = X[labels == i].mean(axis=0)
            else:
                new_centroids[i] = centroids[i]

        # Check convergence
        if np.allclose(centroids, new_centroids):
            break

        centroids = new_centroids

    # Calculate inertia
    inertia = 0
    for i in range(k):
        cluster_points = X[labels == i]
        inertia += np.sum((cluster_points - centroids[i])**2)

    return labels, centroids, inertia

# Try different k values
for k in [2, 3, 4]:
    labels, centroids, inertia = simple_kmeans(X_scaled, k)
    df[f'cluster_k{k}'] = labels

    print(f"\nK-means with k={k}:")
    print(f"  Inertia: {inertia:.4f}")
    for i in range(k):
        cluster_groups = df.loc[df[f'cluster_k{k}'] == i, 'group_id'].values
        cluster_n_trials = df.loc[df[f'cluster_k{k}'] == i, 'n_trials'].mean()
        cluster_success_rate = df.loc[df[f'cluster_k{k}'] == i, 'success_rate'].mean()
        print(f"  Cluster {i}: Groups {cluster_groups}")
        print(f"    Mean n_trials: {cluster_n_trials:.1f}")
        print(f"    Mean success_rate: {cluster_success_rate:.4f}")

# 4. DISTANCE MATRIX ANALYSIS
print("\n4. PAIRWISE DISTANCE ANALYSIS")
print("-"*80)

# Calculate pairwise distances
dist_matrix = squareform(pdist(X_scaled, metric='euclidean'))

print(f"Distance matrix shape: {dist_matrix.shape}")
print(f"Mean distance: {dist_matrix[np.triu_indices_from(dist_matrix, k=1)].mean():.4f}")
print(f"Min distance: {dist_matrix[np.triu_indices_from(dist_matrix, k=1)].min():.4f}")
print(f"Max distance: {dist_matrix[np.triu_indices_from(dist_matrix, k=1)].max():.4f}")

# Find closest pairs
triu_indices = np.triu_indices_from(dist_matrix, k=1)
triu_distances = dist_matrix[triu_indices]
closest_pairs_idx = np.argsort(triu_distances)[:5]

print("\nTop 5 closest group pairs:")
for idx in closest_pairs_idx:
    i, j = triu_indices[0][idx], triu_indices[1][idx]
    dist = dist_matrix[i, j]
    print(f"  Groups {int(df.iloc[i]['group_id'])} & {int(df.iloc[j]['group_id'])}: distance={dist:.4f}")

# Find most distant pairs
farthest_pairs_idx = np.argsort(triu_distances)[-5:][::-1]

print("\nTop 5 most distant group pairs:")
for idx in farthest_pairs_idx:
    i, j = triu_indices[0][idx], triu_indices[1][idx]
    dist = dist_matrix[i, j]
    print(f"  Groups {int(df.iloc[i]['group_id'])} & {int(df.iloc[j]['group_id'])}: distance={dist:.4f}")

# 5. CLUSTER CHARACTERIZATION (K=2)
print("\n5. CLUSTER CHARACTERIZATION (K=2)")
print("-"*80)

for i in range(2):
    cluster_data = df[df['cluster_k2'] == i]
    print(f"\nCluster {i} (n={len(cluster_data)}):")
    print(f"  Groups: {cluster_data['group_id'].values}")
    print(f"  n_trials: mean={cluster_data['n_trials'].mean():.1f}, "
          f"std={cluster_data['n_trials'].std():.1f}, "
          f"range=[{cluster_data['n_trials'].min()}, {cluster_data['n_trials'].max()}]")
    print(f"  r_successes: mean={cluster_data['r_successes'].mean():.1f}, "
          f"std={cluster_data['r_successes'].std():.1f}")
    print(f"  success_rate: mean={cluster_data['success_rate'].mean():.4f}, "
          f"std={cluster_data['success_rate'].std():.4f}, "
          f"range=[{cluster_data['success_rate'].min():.4f}, {cluster_data['success_rate'].max():.4f}]")

# 6. STATISTICAL TESTS FOR CLUSTER DIFFERENCES
print("\n6. STATISTICAL TESTS FOR CLUSTER DIFFERENCES (K=2)")
print("-"*80)

cluster0 = df[df['cluster_k2'] == 0]
cluster1 = df[df['cluster_k2'] == 1]

# Test for n_trials
u_trials, p_trials = mannwhitneyu(cluster0['n_trials'], cluster1['n_trials'], alternative='two-sided')
t_trials, p_t_trials = ttest_ind(cluster0['n_trials'], cluster1['n_trials'])

print(f"n_trials difference:")
print(f"  Mann-Whitney U: U={u_trials:.2f}, p={p_trials:.4f}")
print(f"  t-test: t={t_trials:.2f}, p={p_t_trials:.4f}")
print(f"  Cluster 0 mean: {cluster0['n_trials'].mean():.1f}")
print(f"  Cluster 1 mean: {cluster1['n_trials'].mean():.1f}")

# Test for success_rate
u_rate, p_rate = mannwhitneyu(cluster0['success_rate'], cluster1['success_rate'], alternative='two-sided')
t_rate, p_t_rate = ttest_ind(cluster0['success_rate'], cluster1['success_rate'])

print(f"\nsuccess_rate difference:")
print(f"  Mann-Whitney U: U={u_rate:.2f}, p={p_rate:.4f}")
print(f"  t-test: t={t_rate:.2f}, p={p_t_rate:.4f}")
print(f"  Cluster 0 mean: {cluster0['success_rate'].mean():.4f}")
print(f"  Cluster 1 mean: {cluster1['success_rate'].mean():.4f}")

# 7. CLUSTER STABILITY CHECK
print("\n7. CLUSTER STABILITY CHECK")
print("-"*80)

# Compare hierarchical and k-means clustering
agreement = (df['cluster_h2'] - 1 == df['cluster_k2']).sum() / len(df)
# Account for possible label swapping
agreement_swapped = (df['cluster_h2'] - 1 != df['cluster_k2']).sum() / len(df)
agreement_final = max(agreement, agreement_swapped)

print(f"Agreement between hierarchical and k-means (k=2): {agreement_final*100:.1f}%")

print("\nHierarchical clustering (Ward, k=2):")
for i in range(1, 3):
    groups = df[df['cluster_h2'] == i]['group_id'].values
    print(f"  Cluster {i}: {groups}")

print("\nK-means clustering (k=2):")
for i in range(2):
    groups = df[df['cluster_k2'] == i]['group_id'].values
    print(f"  Cluster {i}: {groups}")

print("\n" + "="*80)
print("CLUSTERING ANALYSIS COMPLETE")
print("="*80)

# Save updated data
df.to_csv('/workspace/eda/analyst_2/code/processed_data_with_clusters.csv', index=False)
print("\nData with cluster labels saved")
