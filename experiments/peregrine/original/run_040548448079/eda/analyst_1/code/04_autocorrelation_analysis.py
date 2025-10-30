"""
Autocorrelation and Temporal Dependency Analysis
Analyst 1: Temporal Patterns and Trends
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import pickle
import warnings
warnings.filterwarnings('ignore')

# Load data and models
df = pd.read_csv('/workspace/data/data_analyst_1.csv')
df = df.sort_values('year').reset_index(drop=True)

with open('/workspace/eda/analyst_1/code/trend_models.pkl', 'rb') as f:
    results = pickle.load(f)

models = results['models']
X = results['X']
y = results['y']

print("=" * 80)
print("AUTOCORRELATION ANALYSIS")
print("=" * 80)

# Function to compute ACF
def compute_acf(data, nlags=15):
    """Compute autocorrelation function"""
    acf = np.zeros(nlags + 1)
    mean = np.mean(data)
    var = np.var(data)

    for lag in range(nlags + 1):
        if lag == 0:
            acf[lag] = 1.0
        else:
            c0 = np.sum((data[:-lag] - mean) * (data[lag:] - mean))
            acf[lag] = c0 / (len(data) * var)

    return acf

# Function to compute PACF (simplified)
def compute_pacf(data, nlags=15):
    """Compute partial autocorrelation function using Yule-Walker"""
    acf = compute_acf(data, nlags)
    pacf = np.zeros(nlags + 1)
    pacf[0] = 1.0

    if nlags > 0:
        pacf[1] = acf[1]

    for k in range(2, nlags + 1):
        # Yule-Walker equations
        numerator = acf[k]
        for j in range(1, k):
            numerator -= pacf[j] * acf[k - j]

        denominator = 1.0
        for j in range(1, k):
            denominator -= pacf[j] * acf[j]

        if abs(denominator) > 1e-10:
            pacf[k] = numerator / denominator
        else:
            pacf[k] = 0

    return pacf

# ============================================================================
# Analysis 1: Raw data autocorrelation
# ============================================================================
print("\nRAW DATA AUTOCORRELATION:")
print("-" * 80)

nlags = 15
acf_raw = compute_acf(y, nlags)
pacf_raw = compute_pacf(y, nlags)

print(f"\nACF (first 10 lags):")
for lag in range(1, min(11, len(acf_raw))):
    print(f"  Lag {lag}: {acf_raw[lag]:.4f}")

print(f"\nPACF (first 10 lags):")
for lag in range(1, min(11, len(pacf_raw))):
    print(f"  Lag {lag}: {pacf_raw[lag]:.4f}")

# Ljung-Box test for autocorrelation
def ljung_box(data, lags=10):
    """Simple Ljung-Box test"""
    n = len(data)
    acf = compute_acf(data, lags)

    Q = n * (n + 2) * np.sum([acf[i]**2 / (n - i) for i in range(1, lags + 1)])

    # Chi-square critical value (approximate, 95% confidence)
    from scipy.stats import chi2
    p_value = 1 - chi2.cdf(Q, lags)

    return Q, p_value

Q_stat, p_value = ljung_box(y, lags=10)
print(f"\nLjung-Box Test (10 lags):")
print(f"  Q-statistic: {Q_stat:.4f}")
print(f"  p-value: {p_value:.6f}")
print(f"  Conclusion: {'Significant autocorrelation' if p_value < 0.05 else 'No significant autocorrelation'}")

# ============================================================================
# Analysis 2: Residual autocorrelation for each model
# ============================================================================
print("\n" + "=" * 80)
print("RESIDUAL AUTOCORRELATION BY MODEL")
print("=" * 80)

residual_acf = {}
residual_pacf = {}

for name in ['Linear', 'Quadratic', 'Cubic', 'Exponential']:
    residuals = y - models[name]['predictions']

    acf = compute_acf(residuals, nlags)
    pacf = compute_pacf(residuals, nlags)

    residual_acf[name] = acf
    residual_pacf[name] = pacf

    Q_stat, p_value = ljung_box(residuals, lags=10)

    print(f"\n{name}:")
    print(f"  ACF(1): {acf[1]:.4f}")
    print(f"  ACF(2): {acf[2]:.4f}")
    print(f"  PACF(1): {pacf[1]:.4f}")
    print(f"  Ljung-Box Q: {Q_stat:.4f}, p-value: {p_value:.6f}")
    print(f"  Residual autocorrelation: {'YES' if p_value < 0.05 else 'NO'}")

# ============================================================================
# Analysis 3: First differences autocorrelation
# ============================================================================
print("\n" + "=" * 80)
print("FIRST DIFFERENCES AUTOCORRELATION")
print("=" * 80)

diff_y = np.diff(y)
acf_diff = compute_acf(diff_y, nlags)
pacf_diff = compute_pacf(diff_y, nlags)

print(f"\nFirst Differences ACF (first 5 lags):")
for lag in range(1, min(6, len(acf_diff))):
    print(f"  Lag {lag}: {acf_diff[lag]:.4f}")

Q_stat_diff, p_value_diff = ljung_box(diff_y, lags=10)
print(f"\nLjung-Box Test:")
print(f"  Q-statistic: {Q_stat_diff:.4f}")
print(f"  p-value: {p_value_diff:.6f}")

# Save for plotting
acf_data = {
    'raw': {'acf': acf_raw, 'pacf': pacf_raw},
    'diff': {'acf': acf_diff, 'pacf': pacf_diff},
    'residuals': {'acf': residual_acf, 'pacf': residual_pacf}
}

with open('/workspace/eda/analyst_1/code/acf_data.pkl', 'wb') as f:
    pickle.dump(acf_data, f)

print("\n" + "=" * 80)
print("Analysis complete. Results saved for visualization.")
print("=" * 80)
