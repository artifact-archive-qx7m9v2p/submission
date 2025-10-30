"""
Initial Data Exploration and Quality Assessment
================================================
Purpose: Load data, assess quality, generate basic statistics
Author: EDA Specialist Agent
Date: 2025-10-27
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Setup paths
DATA_PATH = Path("/workspace/data/data.csv")
OUTPUT_DIR = Path("/workspace/eda")
CODE_DIR = OUTPUT_DIR / "code"
VIZ_DIR = OUTPUT_DIR / "visualizations"

# Ensure directories exist
CODE_DIR.mkdir(parents=True, exist_ok=True)
VIZ_DIR.mkdir(parents=True, exist_ok=True)

def load_and_inspect_data(filepath):
    """
    Load data and perform initial inspection.

    Parameters:
    -----------
    filepath : Path
        Path to the CSV file

    Returns:
    --------
    pd.DataFrame
        Loaded dataframe
    dict
        Dictionary of quality metrics
    """
    print("="*80)
    print("DATA LOADING AND INITIAL INSPECTION")
    print("="*80)

    # Load data
    df = pd.read_csv(filepath)

    print(f"\n1. Dataset Shape: {df.shape[0]} observations, {df.shape[1]} variables")
    print(f"   Variables: {list(df.columns)}")

    # Display first and last rows
    print("\n2. First 5 rows:")
    print(df.head())
    print("\n3. Last 5 rows:")
    print(df.tail())

    # Data types
    print("\n4. Data Types:")
    print(df.dtypes)

    # Check for missing values
    print("\n5. Missing Values:")
    missing = df.isnull().sum()
    print(missing)
    missing_pct = (missing / len(df)) * 100
    print("\nMissing Values (%):")
    print(missing_pct)

    # Check for duplicates
    print("\n6. Duplicate Rows:")
    duplicates = df.duplicated().sum()
    print(f"   Total duplicates: {duplicates}")
    if duplicates > 0:
        print("   Duplicate rows:")
        print(df[df.duplicated(keep=False)])

    # Basic statistics
    print("\n7. Basic Statistics:")
    print(df.describe())

    # Quality metrics
    quality_metrics = {
        'n_obs': len(df),
        'n_vars': len(df.columns),
        'missing_total': missing.sum(),
        'duplicates': duplicates,
        'data_types': df.dtypes.to_dict()
    }

    return df, quality_metrics

def assess_data_quality(df):
    """
    Detailed data quality assessment.

    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe

    Returns:
    --------
    dict
        Quality assessment results
    """
    print("\n" + "="*80)
    print("DATA QUALITY ASSESSMENT")
    print("="*80)

    quality_report = {}

    # Check for numeric types
    print("\n1. Numeric Variables Check:")
    for col in df.columns:
        is_numeric = pd.api.types.is_numeric_dtype(df[col])
        print(f"   {col}: {'Numeric' if is_numeric else 'Non-numeric'}")
        quality_report[f'{col}_numeric'] = is_numeric

    # Range checks
    print("\n2. Data Range:")
    for col in df.select_dtypes(include=[np.number]).columns:
        min_val = df[col].min()
        max_val = df[col].max()
        range_val = max_val - min_val
        print(f"   {col}: [{min_val:.4f}, {max_val:.4f}], Range: {range_val:.4f}")
        quality_report[f'{col}_range'] = (min_val, max_val, range_val)

    # Check for zeros and negative values
    print("\n3. Special Values Check:")
    for col in df.select_dtypes(include=[np.number]).columns:
        n_zeros = (df[col] == 0).sum()
        n_negative = (df[col] < 0).sum()
        print(f"   {col}: Zeros={n_zeros}, Negative={n_negative}")
        quality_report[f'{col}_zeros'] = n_zeros
        quality_report[f'{col}_negative'] = n_negative

    # Check for potential outliers using IQR method
    print("\n4. Potential Outliers (IQR method):")
    for col in df.select_dtypes(include=[np.number]).columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]
        print(f"   {col}: {len(outliers)} potential outliers")
        if len(outliers) > 0:
            print(f"      Values: {outliers.values}")
        quality_report[f'{col}_outliers'] = len(outliers)

    # Check for exact duplicates in x values
    print("\n5. Duplicate x Values:")
    if 'x' in df.columns:
        x_counts = df['x'].value_counts()
        duplicates = x_counts[x_counts > 1]
        print(f"   Number of x values with duplicates: {len(duplicates)}")
        if len(duplicates) > 0:
            print("   x values with multiple observations:")
            for x_val, count in duplicates.items():
                print(f"      x={x_val}: {count} observations")
                y_vals = df[df['x'] == x_val]['Y'].values
                print(f"         Y values: {y_vals}")
        quality_report['x_duplicates'] = len(duplicates)

    return quality_report

def compute_advanced_statistics(df):
    """
    Compute advanced statistical measures.

    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe

    Returns:
    --------
    dict
        Advanced statistics
    """
    print("\n" + "="*80)
    print("ADVANCED STATISTICS")
    print("="*80)

    from scipy import stats

    advanced_stats = {}

    for col in df.select_dtypes(include=[np.number]).columns:
        print(f"\n{col}:")

        # Basic moments
        mean = df[col].mean()
        median = df[col].median()
        std = df[col].std()
        var = df[col].var()

        # Skewness and kurtosis
        skewness = stats.skew(df[col])
        kurtosis = stats.kurtosis(df[col])

        # Coefficient of variation
        cv = (std / mean) * 100 if mean != 0 else np.nan

        # Range and IQR
        q25 = df[col].quantile(0.25)
        q75 = df[col].quantile(0.75)
        iqr = q75 - q25

        print(f"  Mean: {mean:.4f}")
        print(f"  Median: {median:.4f}")
        print(f"  Std Dev: {std:.4f}")
        print(f"  Variance: {var:.4f}")
        print(f"  CV: {cv:.2f}%")
        print(f"  Skewness: {skewness:.4f} {'(right-skewed)' if skewness > 0 else '(left-skewed)' if skewness < 0 else '(symmetric)'}")
        print(f"  Kurtosis: {kurtosis:.4f} {'(heavy-tailed)' if kurtosis > 0 else '(light-tailed)'}")
        print(f"  IQR: {iqr:.4f}")
        print(f"  Q25: {q25:.4f}, Q75: {q75:.4f}")

        # Normality test
        shapiro_stat, shapiro_p = stats.shapiro(df[col])
        print(f"  Shapiro-Wilk test: statistic={shapiro_stat:.4f}, p-value={shapiro_p:.4f}")
        print(f"  {'Appears normally distributed' if shapiro_p > 0.05 else 'Deviates from normal distribution'} (α=0.05)")

        advanced_stats[col] = {
            'mean': mean,
            'median': median,
            'std': std,
            'variance': var,
            'cv': cv,
            'skewness': skewness,
            'kurtosis': kurtosis,
            'iqr': iqr,
            'q25': q25,
            'q75': q75,
            'shapiro_stat': shapiro_stat,
            'shapiro_p': shapiro_p
        }

    return advanced_stats

def main():
    """Main execution function."""
    print("\n" + "="*80)
    print("EXPLORATORY DATA ANALYSIS - PHASE 1: INITIAL EXPLORATION")
    print("="*80 + "\n")

    # Load and inspect
    df, quality_metrics = load_and_inspect_data(DATA_PATH)

    # Quality assessment
    quality_report = assess_data_quality(df)

    # Advanced statistics
    advanced_stats = compute_advanced_statistics(df)

    # Save cleaned data and metrics
    print("\n" + "="*80)
    print("SAVING RESULTS")
    print("="*80)

    # Save the dataframe for later use
    output_data_path = OUTPUT_DIR / "cleaned_data.csv"
    df.to_csv(output_data_path, index=False)
    print(f"\nCleaned data saved to: {output_data_path}")

    # Save statistics to file
    stats_file = OUTPUT_DIR / "initial_statistics.txt"
    with open(stats_file, 'w') as f:
        f.write("INITIAL DATA EXPLORATION SUMMARY\n")
        f.write("="*80 + "\n\n")
        f.write(f"Dataset: {DATA_PATH}\n")
        f.write(f"Shape: {df.shape}\n")
        f.write(f"Variables: {list(df.columns)}\n\n")
        f.write("QUALITY METRICS:\n")
        for key, value in quality_metrics.items():
            f.write(f"  {key}: {value}\n")
        f.write("\nQUALITY REPORT:\n")
        for key, value in quality_report.items():
            f.write(f"  {key}: {value}\n")
        f.write("\nADVANCED STATISTICS:\n")
        for var, stats in advanced_stats.items():
            f.write(f"\n{var}:\n")
            for stat_name, stat_value in stats.items():
                f.write(f"  {stat_name}: {stat_value}\n")

    print(f"Statistics saved to: {stats_file}")

    print("\n" + "="*80)
    print("PHASE 1 COMPLETE")
    print("="*80 + "\n")

    return df, quality_metrics, quality_report, advanced_stats

if __name__ == "__main__":
    df, quality_metrics, quality_report, advanced_stats = main()
