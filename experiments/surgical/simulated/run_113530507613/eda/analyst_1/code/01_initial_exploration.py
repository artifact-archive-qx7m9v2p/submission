"""
Initial Data Exploration - Analyst 1
Focus: Distributional characteristics and variance patterns
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
data = pd.read_csv('/workspace/data/data_analyst_1.csv')

print("="*80)
print("INITIAL DATA EXPLORATION")
print("="*80)

print("\n1. DATASET SHAPE")
print(f"Rows: {len(data)}, Columns: {len(data.columns)}")

print("\n2. DATA TYPES")
print(data.dtypes)

print("\n3. FIRST FEW ROWS")
print(data.head(10))

print("\n4. LAST FEW ROWS")
print(data.tail(10))

print("\n5. MISSING VALUES")
print(data.isnull().sum())

print("\n6. BASIC SUMMARY STATISTICS")
print(data.describe())

print("\n7. ADDITIONAL STATISTICS FOR SUCCESS_RATE")
print(f"Median: {data['success_rate'].median():.6f}")
print(f"Mode: {data['success_rate'].mode().values if not data['success_rate'].mode().empty else 'No mode'}")
print(f"Skewness: {data['success_rate'].skew():.6f}")
print(f"Kurtosis: {data['success_rate'].kurtosis():.6f}")
print(f"CV (Coefficient of Variation): {data['success_rate'].std() / data['success_rate'].mean():.6f}")

print("\n8. ADDITIONAL STATISTICS FOR N_TRIALS")
print(f"Median: {data['n_trials'].median():.1f}")
print(f"Skewness: {data['n_trials'].skew():.6f}")
print(f"Kurtosis: {data['n_trials'].kurtosis():.6f}")

print("\n9. RANGE INFORMATION")
print(f"Success rate range: [{data['success_rate'].min():.6f}, {data['success_rate'].max():.6f}]")
print(f"N_trials range: [{data['n_trials'].min()}, {data['n_trials'].max()}]")
print(f"R_successes range: [{data['r_successes'].min()}, {data['r_successes'].max()}]")

print("\n10. QUARTILE INFORMATION")
for col in ['n_trials', 'r_successes', 'success_rate']:
    q25, q50, q75 = data[col].quantile([0.25, 0.50, 0.75])
    iqr = q75 - q25
    print(f"{col}: Q1={q25:.4f}, Q2={q50:.4f}, Q3={q75:.4f}, IQR={iqr:.4f}")

print("\n" + "="*80)
