================================================================================
META-ANALYSIS DATASET: INITIAL EXPLORATION
================================================================================

================================================================================
1. BASIC DATA STRUCTURE
================================================================================

Dataset shape: (8, 3)
Number of studies (J): 8

Columns: ['study', 'y', 'sigma']
Data types:
study    int64
y        int64
sigma    int64
dtype: object

--- First few rows ---
   study   y  sigma
0      1  28     15
1      2   8     10
2      3  -3     16
3      4   7     11
4      5  -1      9

--- Last few rows ---
   study   y  sigma
3      4   7     11
4      5  -1      9
5      6   1     11
6      7  18     10
7      8  12     18

--- Full dataset ---
   study   y  sigma
0      1  28     15
1      2   8     10
2      3  -3     16
3      4   7     11
4      5  -1      9
5      6   1     11
6      7  18     10
7      8  12     18

================================================================================
2. DATA QUALITY ASSESSMENT
================================================================================

--- Missing Values ---
Missing values per column:
study    0
y        0
sigma    0
dtype: int64
Total missing values: 0
Percentage missing: 0.00%

--- Duplicate Studies ---
Number of duplicate rows: 0

--- Study ID Continuity ---
Expected study IDs: [1, 2, 3, 4, 5, 6, 7, 8]
Actual study IDs: [np.int64(1), np.int64(2), np.int64(3), np.int64(4), np.int64(5), np.int64(6), np.int64(7), np.int64(8)]
IDs continuous and complete: True

--- Standard Error Validity ---
Minimum sigma: 9
Any sigma <= 0: False
Any sigma == 0: False

--- Infinite Values ---
Infinite values in y: 0
Infinite values in sigma: 0

================================================================================
3. DESCRIPTIVE STATISTICS
================================================================================

--- Summary Statistics ---
         study          y      sigma
count  8.00000   8.000000   8.000000
mean   4.50000   8.750000  12.500000
std    2.44949  10.443727   3.338092
min    1.00000  -3.000000   9.000000
25%    2.75000   0.500000  10.000000
50%    4.50000   7.500000  11.000000
75%    6.25000  13.500000  15.250000
max    8.00000  28.000000  18.000000

--- Additional Statistics ---

Effect Size (y):
  Mean: 8.75
  Median: 7.50
  Std Dev: 10.44
  Range: [-3.00, 28.00]
  IQR: 13.00
  Skewness: 0.66
  Kurtosis: -0.58

Standard Error (sigma):
  Mean: 12.50
  Median: 11.00
  Std Dev: 3.34
  Range: [9.00, 18.00]
  IQR: 5.25
  Skewness: 0.59
  Kurtosis: -1.23

================================================================================
4. EXTREME VALUE ANALYSIS
================================================================================

--- Z-scores (|z| > 2 considered extreme) ---

Effect Size Z-scores:
   study   y  y_zscore
0      1  28  1.843212
2      3  -3 -1.125077
4      5  -1 -0.933575
6      7  18  0.885699
5      6   1 -0.742072
7      8  12  0.311192
3      4   7 -0.167565
1      2   8 -0.071813

Standard Error Z-scores:
   study  sigma  sigma_zscore
7      8     18      1.647648
2      3     16      1.048503
4      5      9     -1.048503
0      1     15      0.748931
6      7     10     -0.748931
1      2     10     -0.748931
3      4     11     -0.449359
5      6     11     -0.449359

--- Studies with Extreme Values ---
Studies with |y_zscore| > 2: []
Studies with |sigma_zscore| > 2: []

--- Precision Analysis ---
   study  sigma  precision    weight
4      5      9   0.111111  0.012346
1      2     10   0.100000  0.010000
6      7     10   0.100000  0.010000
3      4     11   0.090909  0.008264
5      6     11   0.090909  0.008264
0      1     15   0.066667  0.004444
2      3     16   0.062500  0.003906
7      8     18   0.055556  0.003086

Most precise study (lowest sigma): Study 5
Least precise study (highest sigma): Study 8

================================================================================
5. STUDY ORDERING PATTERNS
================================================================================

Correlation between study ID and y: -0.162
Correlation between study ID and sigma: 0.035

Spearman correlation (study ID vs y): rho=0.000, p=1.000
Spearman correlation (study ID vs sigma): rho=0.036, p=0.932

--- Potential Temporal/Quality Trends ---
No strong evidence of trends in effect sizes by study order
No strong evidence of trends in standard errors by study order

================================================================================
6. RELATIONSHIP BETWEEN EFFECT SIZE AND STANDARD ERROR
================================================================================

Pearson correlation (y vs sigma): 0.213
Spearman correlation (y vs sigma): rho=0.108, p=0.798

No strong correlation between effect size and standard error

================================================================================
7. META-ANALYSIS CONTEXT & COMPARISON
================================================================================

Sample size characteristics:
  Number of studies (J): 8
  Classification: Small meta-analysis (5 <= J < 10)

Precision heterogeneity:
  Coefficient of variation (CV) for sigma: 0.27
  Low heterogeneity in study precision

Effect size heterogeneity:
  Approximate I²: 0.0%

Sign consistency:
  Positive effects: 6 (75.0%)
  Negative effects: 2 (25.0%)
  Zero effects: 0

================================================================================
8. CONFIDENCE INTERVAL ANALYSIS
================================================================================

--- 95% Confidence Intervals ---
 study  y  ci_lower  ci_upper  ci_width
     1 28     -1.40     57.40     58.80
     2  8    -11.60     27.60     39.20
     3 -3    -34.36     28.36     62.72
     4  7    -14.56     28.56     43.12
     5 -1    -18.64     16.64     35.28
     6  1    -20.56     22.56     43.12
     7 18     -1.60     37.60     39.20
     8 12    -23.28     47.28     70.56

Studies with CIs including zero: 8/8 (100.0%)

CI Width analysis:
  Mean CI width: 49.00
  Range: [35.28, 70.56]
  CV: 0.27

================================================================================
INITIAL EXPLORATION COMPLETE
================================================================================

Enhanced data saved to: /workspace/eda/analyst_3/code/data_with_diagnostics.csv
================================================================================
HYPOTHESIS TESTING & CONTEXTUAL ANALYSIS
================================================================================

================================================================================
HYPOTHESIS 1: PUBLICATION BIAS DETECTION
================================================================================

H1a: Small-study effects (correlation between effect size and SE)
----------------------------------------------------------------------
Pearson correlation (y vs sigma): r = 0.213
Spearman correlation (y vs sigma): rho = 0.108, p = 0.798

Result: No significant correlation (p >= 0.10)
Interpretation: No strong evidence of small-study effects

H1b: Egger's regression test for funnel plot asymmetry
----------------------------------------------------------------------
Egger's intercept: 0.917
Egger's p-value: 0.874
Standard error: 15.843

Result: No significant funnel asymmetry (p >= 0.10)
Interpretation: Limited evidence of publication bias

CAVEAT: With J=8, power to detect publication bias is very low.
These tests are unreliable with small meta-analyses.

================================================================================
HYPOTHESIS 2: HETEROGENEITY ASSESSMENT
================================================================================

H2: Cochran's Q test for heterogeneity
----------------------------------------------------------------------
Cochran's Q: 4.707
Degrees of freedom: 7
p-value: 0.696

Result: No significant heterogeneity detected (p >= 0.10)
Interpretation: Variation consistent with sampling error

I² statistic: 0.0%
Interpretation: Minimal heterogeneity
Tau² (between-study variance): 0.000

================================================================================
HYPOTHESIS 3: TEMPORAL/QUALITY TRENDS
================================================================================

H3a: Trend in effect sizes over study sequence
----------------------------------------------------------------------
Slope (effect per study): -0.690
R²: 0.026
p-value: 0.702

Result: No significant trend (p >= 0.10)
Interpretation: No evidence of temporal/quality effects

H3b: Trend in precision over study sequence
----------------------------------------------------------------------
Slope (precision per study): 0.00033
R²: 0.002
p-value: 0.926

Result: No significant trend (p >= 0.10)
Interpretation: No evidence of quality changes over time

================================================================================
HYPOTHESIS 4: OVERALL EFFECT TESTING
================================================================================

H4: Fixed-effect meta-analysis test
----------------------------------------------------------------------
Fixed-effect estimate: 7.686
Standard error: 4.072
95% CI: [-0.295, 15.667]
Z-score: 1.887
p-value: 0.0591

Result: Non-significant effect (p >= 0.05)
Interpretation: Cannot reject null hypothesis of no effect

H4b: Random-effects meta-analysis test
----------------------------------------------------------------------
Random-effect estimate: 7.686
Standard error: 4.072
95% CI: [-0.295, 15.667]
Z-score: 1.887
p-value: 0.0591

Result: Non-significant effect (p >= 0.05)
Interpretation: Cannot reject null hypothesis of no effect

================================================================================
HYPOTHESIS 5: INFLUENTIAL STUDIES & OUTLIERS
================================================================================

H5: Leave-one-out sensitivity analysis
----------------------------------------------------------------------

Fixed-effect estimates with each study removed:
Study    Estimate   SE         95% CI                    Change    
----------------------------------------------------------------------
1        6.070      4.231      [-2.223, 14.362]    -1.616
2        7.623      4.458      [-1.115, 16.361]    -0.062
3        8.426      4.211      [ 0.173, 16.678]     0.740
4        7.794      4.383      [-0.797, 16.386]     0.109
5        9.921      4.566      [ 0.972, 18.870]     2.236
6        8.747      4.383      [ 0.156, 17.338]     1.062
7        5.636      4.458      [-3.103, 14.374]    -2.050
8        7.453      4.180      [-0.740, 15.646]    -0.233

Most influential study: Study 5
Maximum influence: 2.236
As % of fixed effect: 29.1%

================================================================================
HYPOTHESIS 6: SAMPLE SIZE & POWER CONSIDERATIONS
================================================================================

H6: Is J=8 adequate for this meta-analysis?
----------------------------------------------------------------------
Number of studies (J): 8

General guidelines:
  - J < 5: Too small for reliable meta-analysis
  - 5 <= J < 10: Small, limited power for heterogeneity tests
  - 10 <= J < 20: Medium, reasonable for basic meta-analysis
  - J >= 20: Large, good power for most analyses

Current classification: Small meta-analysis (J=8)

Implications:
  1. Low power to detect heterogeneity (Q test unreliable)
  2. Low power to detect publication bias (Egger test unreliable)
  3. Random-effects estimates may be unstable
  4. Subgroup analyses not recommended
  5. Meta-regression not feasible
  6. Sensitivity analyses limited

Precision characteristics:
  Total weight (sum of 1/sigma²): 0.060
  Average precision: 0.085
  Effective sample size: ~0.1 (precision-adjusted)

================================================================================
SUMMARY OF HYPOTHESIS TESTING
================================================================================

1. PUBLICATION BIAS:
   - No strong evidence detected

2. HETEROGENEITY:
   - No significant heterogeneity (Q=4.71, p=0.696)
   - I²=0.0%, variation consistent with sampling error

3. TEMPORAL/QUALITY TRENDS:
   - No evidence of temporal or quality trends

4. OVERALL EFFECT:
   - Non-significant overall effect (Fixed: 7.69, p=0.0591)

5. INFLUENTIAL STUDIES:
   - No single study dominates the meta-analysis

6. SAMPLE SIZE:
   - J=8 is small but acceptable for basic meta-analysis
   - Limited power for complex analyses
   - Results should be interpreted cautiously

================================================================================
HYPOTHESIS TESTING COMPLETE
================================================================================
