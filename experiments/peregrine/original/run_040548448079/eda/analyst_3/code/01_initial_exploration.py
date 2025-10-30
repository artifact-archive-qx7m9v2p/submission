"""
Initial exploration of the dataset to understand basic patterns
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Load data
df = pd.read_csv('/workspace/data/data_analyst_3.csv')

print("="*60)
print("INITIAL DATA EXPLORATION")
print("="*60)
print("\nDataset shape:", df.shape)
print("\nFirst few rows:")
print(df.head(10))
print("\nLast few rows:")
print(df.tail(10))

print("\n" + "="*60)
print("DESCRIPTIVE STATISTICS")
print("="*60)
print(df.describe())

print("\n" + "="*60)
print("DATA QUALITY CHECKS")
print("="*60)
print("Missing values:")
print(df.isnull().sum())
print("\nData types:")
print(df.dtypes)

print("\n" + "="*60)
print("BASIC RELATIONSHIP METRICS")
print("="*60)
print(f"Pearson correlation: {df['year'].corr(df['C']):.4f}")
print(f"Spearman correlation: {df['year'].corr(df['C'], method='spearman'):.4f}")

# Calculate growth rate
df_sorted = df.sort_values('year')
growth_ratio = df_sorted['C'].iloc[-1] / df_sorted['C'].iloc[0]
print(f"\nGrowth from first to last observation: {growth_ratio:.2f}x")

# Check variance relationship
print("\n" + "="*60)
print("VARIANCE STRUCTURE")
print("="*60)
# Split into thirds to check variance heterogeneity
n = len(df)
third = n // 3
df_sorted = df.sort_values('year').reset_index(drop=True)
var_low = df_sorted['C'].iloc[:third].var()
var_mid = df_sorted['C'].iloc[third:2*third].var()
var_high = df_sorted['C'].iloc[2*third:].var()
print(f"Variance in first third: {var_low:.2f}")
print(f"Variance in middle third: {var_mid:.2f}")
print(f"Variance in last third: {var_high:.2f}")
print(f"Ratio (high/low): {var_high/var_low:.2f}")
