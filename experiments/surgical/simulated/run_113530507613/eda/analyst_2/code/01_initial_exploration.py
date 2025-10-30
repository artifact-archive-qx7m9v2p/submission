"""
Initial Data Exploration - EDA Analyst 2
Focus: Temporal/Sequential Patterns and Group Relationships
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist, squareform
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

# Load data
df = pd.read_csv('/workspace/data/data_analyst_2.csv')

print("="*80)
print("INITIAL DATA EXPLORATION")
print("="*80)

print("\n1. BASIC DATA STRUCTURE")
print("-"*80)
print(f"Shape: {df.shape}")
print(f"\nColumn types:\n{df.dtypes}")
print(f"\nFirst few rows:\n{df.head()}")
print(f"\nLast few rows:\n{df.tail()}")

print("\n2. DESCRIPTIVE STATISTICS")
print("-"*80)
print(df.describe())

print("\n3. DATA QUALITY CHECKS")
print("-"*80)
print(f"Missing values:\n{df.isnull().sum()}")
print(f"\nDuplicate rows: {df.duplicated().sum()}")
print(f"\nGroup ID range: {df['group_id'].min()} to {df['group_id'].max()}")
print(f"Group ID sequential: {(df['group_id'] == range(1, len(df)+1)).all()}")

print("\n4. VARIABLE RANGES AND CONSTRAINTS")
print("-"*80)
print(f"n_trials range: [{df['n_trials'].min()}, {df['n_trials'].max()}]")
print(f"r_successes range: [{df['r_successes'].min()}, {df['r_successes'].max()}]")
print(f"success_rate range: [{df['success_rate'].min():.4f}, {df['success_rate'].max():.4f}]")
print(f"\nLogical constraint check (r_successes <= n_trials): {(df['r_successes'] <= df['n_trials']).all()}")
print(f"Success rate calculation check: {np.allclose(df['success_rate'], df['r_successes']/df['n_trials'])}")

print("\n5. DERIVED VARIABLES")
print("-"*80)
df['failure_count'] = df['n_trials'] - df['r_successes']
df['failure_rate'] = 1 - df['success_rate']
df['logit_success_rate'] = np.log((df['r_successes'] + 0.5) / (df['failure_count'] + 0.5))

print(f"Failure rate range: [{df['failure_rate'].min():.4f}, {df['failure_rate'].max():.4f}]")
print(f"Logit(success_rate) range: [{df['logit_success_rate'].min():.4f}, {df['logit_success_rate'].max():.4f}]")

print("\n6. DISTRIBUTIONAL CHARACTERISTICS")
print("-"*80)
for col in ['n_trials', 'r_successes', 'success_rate']:
    print(f"\n{col}:")
    print(f"  Mean: {df[col].mean():.4f}")
    print(f"  Median: {df[col].median():.4f}")
    print(f"  Std: {df[col].std():.4f}")
    print(f"  Skewness: {stats.skew(df[col]):.4f}")
    print(f"  Kurtosis: {stats.kurtosis(df[col]):.4f}")
    print(f"  CV (Coefficient of Variation): {df[col].std()/df[col].mean():.4f}")

print("\n7. INITIAL CORRELATION ANALYSIS")
print("-"*80)
corr_matrix = df[['n_trials', 'r_successes', 'success_rate']].corr()
print("Correlation Matrix:")
print(corr_matrix)

print("\n" + "="*80)
print("INITIAL EXPLORATION COMPLETE")
print("="*80)
