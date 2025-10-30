"""
Round 1: Initial Exploration - Distributions and Basic Statistics
Focus: Understanding the distribution of effect sizes and standard errors
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300

# Load data
data = pd.read_csv('/workspace/data/data_analyst_1.csv')
print("="*70)
print("META-ANALYSIS DATA: INITIAL EXPLORATION")
print("="*70)

# Basic info
print("\n1. DATA STRUCTURE")
print("-"*70)
print(f"Number of studies: {len(data)}")
print(f"Columns: {list(data.columns)}")
print("\nFirst few rows:")
print(data.head())
print("\nData types:")
print(data.dtypes)
print("\nMissing values:")
print(data.isnull().sum())

# Descriptive statistics
print("\n2. DESCRIPTIVE STATISTICS")
print("-"*70)
print("\nEffect sizes (y):")
print(data['y'].describe())
print(f"  Range: [{data['y'].min():.2f}, {data['y'].max():.2f}]")
print(f"  IQR: {data['y'].quantile(0.75) - data['y'].quantile(0.25):.2f}")
print(f"  Skewness: {data['y'].skew():.3f}")
print(f"  Kurtosis: {data['y'].kurtosis():.3f}")

print("\nStandard errors (sigma):")
print(data['sigma'].describe())
print(f"  Range: [{data['sigma'].min():.2f}, {data['sigma'].max():.2f}]")
print(f"  IQR: {data['sigma'].quantile(0.75) - data['sigma'].quantile(0.25):.2f}")
print(f"  Skewness: {data['sigma'].skew():.3f}")
print(f"  Kurtosis: {data['sigma'].kurtosis():.3f}")
print(f"  Coefficient of variation: {(data['sigma'].std() / data['sigma'].mean()):.3f}")

# Calculate confidence intervals for each study
data['ci_lower'] = data['y'] - 1.96 * data['sigma']
data['ci_upper'] = data['y'] + 1.96 * data['sigma']
data['ci_width'] = data['ci_upper'] - data['ci_lower']

print("\n3. UNCERTAINTY CHARACTERISTICS")
print("-"*70)
print(f"Mean CI width: {data['ci_width'].mean():.2f}")
print(f"Range of CI widths: [{data['ci_width'].min():.2f}, {data['ci_width'].max():.2f}]")
print(f"Ratio of max to min CI width: {data['ci_width'].max() / data['ci_width'].min():.2f}x")

# Check for outliers using different methods
print("\n4. OUTLIER DETECTION (Effect Sizes)")
print("-"*70)

# Method 1: IQR method
Q1, Q3 = data['y'].quantile(0.25), data['y'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
outliers_iqr = data[(data['y'] < lower_bound) | (data['y'] > upper_bound)]
print(f"IQR method (1.5*IQR): {len(outliers_iqr)} outliers")
if len(outliers_iqr) > 0:
    print(f"  Studies: {list(outliers_iqr['study'])}")
    print(f"  Values: {list(outliers_iqr['y'])}")

# Method 2: Z-score method
z_scores = np.abs(stats.zscore(data['y']))
outliers_z = data[z_scores > 2]
print(f"Z-score method (|z| > 2): {len(outliers_z)} outliers")
if len(outliers_z) > 0:
    print(f"  Studies: {list(outliers_z['study'])}")
    print(f"  Values: {list(outliers_z['y'])}")

# Method 3: Modified Z-score (using median absolute deviation)
median = data['y'].median()
mad = np.median(np.abs(data['y'] - median))
modified_z_scores = 0.6745 * (data['y'] - median) / mad if mad != 0 else np.zeros(len(data))
outliers_mad = data[np.abs(modified_z_scores) > 3.5]
print(f"Modified Z-score method (MAD, |z| > 3.5): {len(outliers_mad)} outliers")
if len(outliers_mad) > 0:
    print(f"  Studies: {list(outliers_mad['study'])}")
    print(f"  Values: {list(outliers_mad['y'])}")

# Normality tests
print("\n5. DISTRIBUTION TESTS")
print("-"*70)
shapiro_y = stats.shapiro(data['y'])
print(f"Shapiro-Wilk test for y: W={shapiro_y.statistic:.4f}, p={shapiro_y.pvalue:.4f}")
print(f"  Interpretation: {'Normal' if shapiro_y.pvalue > 0.05 else 'Non-normal'} at alpha=0.05")

shapiro_sigma = stats.shapiro(data['sigma'])
print(f"Shapiro-Wilk test for sigma: W={shapiro_sigma.statistic:.4f}, p={shapiro_sigma.pvalue:.4f}")
print(f"  Interpretation: {'Normal' if shapiro_sigma.pvalue > 0.05 else 'Non-normal'} at alpha=0.05")

# Save processed data
data.to_csv('/workspace/eda/analyst_1/code/processed_data.csv', index=False)
print("\n" + "="*70)
print("Processed data saved to: /workspace/eda/analyst_1/code/processed_data.csv")
print("="*70)
