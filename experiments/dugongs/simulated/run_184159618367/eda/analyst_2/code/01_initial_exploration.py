"""
Initial Data Exploration - Analyst 2
Focus: Functional form and model class recommendations
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

# Load data
data = pd.read_csv('/workspace/data/data_analyst_2.csv')

print("="*60)
print("INITIAL DATA EXPLORATION - ANALYST 2")
print("="*60)

# Basic statistics
print("\n1. DATASET OVERVIEW")
print(f"Number of observations: {len(data)}")
print(f"\nData types:\n{data.dtypes}")
print(f"\nBasic statistics:\n{data.describe()}")

# Check for missing values
print(f"\n2. DATA QUALITY")
print(f"Missing values:\n{data.isnull().sum()}")
print(f"Duplicate rows: {data.duplicated().sum()}")

# Check for repeated x values
print(f"\n3. X VALUE DISTRIBUTION")
x_counts = data['x'].value_counts().sort_index()
print(f"Unique x values: {data['x'].nunique()}")
print(f"Repeated x values (shows variability at same x):")
print(x_counts[x_counts > 1])

# Range and spread
print(f"\n4. VARIABLE RANGES")
print(f"x range: [{data['x'].min():.2f}, {data['x'].max():.2f}]")
print(f"Y range: [{data['Y'].min():.2f}, {data['Y'].max():.2f}]")
print(f"x span: {data['x'].max() - data['x'].min():.2f}")
print(f"Y span: {data['Y'].max() - data['Y'].min():.2f}")

# Initial correlation
print(f"\n5. INITIAL CORRELATION ANALYSIS")
pearson_corr = data['x'].corr(data['Y'])
spearman_corr = data['x'].corr(data['Y'], method='spearman')
print(f"Pearson correlation: {pearson_corr:.4f}")
print(f"Spearman correlation: {spearman_corr:.4f}")
print(f"Difference (suggests non-linearity): {abs(pearson_corr - spearman_corr):.4f}")

# Check for outliers using IQR method
print(f"\n6. OUTLIER DETECTION (IQR method)")
for col in ['x', 'Y']:
    Q1 = data[col].quantile(0.25)
    Q3 = data[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = data[(data[col] < lower_bound) | (data[col] > upper_bound)]
    print(f"{col}: {len(outliers)} potential outliers")
    if len(outliers) > 0:
        print(f"  Values: {outliers[col].values}")

# Distribution tests
print(f"\n7. NORMALITY TESTS")
_, p_x = stats.shapiro(data['x'])
_, p_y = stats.shapiro(data['Y'])
print(f"Shapiro-Wilk test for x: p-value = {p_x:.4f}")
print(f"Shapiro-Wilk test for Y: p-value = {p_y:.4f}")
print(f"  (p < 0.05 suggests non-normal distribution)")

print("\n" + "="*60)
