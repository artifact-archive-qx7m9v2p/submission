================================================================================
BINOMIAL DATA EDA - ANALYST #2
================================================================================

Dataset shape: (12, 3)

First few rows:
   group    n   r
0      1   47   6
1      2  148  19
2      3  119   8
3      4  810  34
4      5  211  12

Data summary:
           group           n          r
count  12.000000   12.000000  12.000000
mean    6.500000  234.500000  16.333333
std     3.605551  198.391486   9.838083
min     1.000000   47.000000   3.000000
25%     3.750000  140.750000   8.750000
50%     6.500000  201.500000  14.500000
75%     9.250000  225.250000  21.000000
max    12.000000  810.000000  34.000000

================================================================================
GROUP-LEVEL STATISTICS
================================================================================
 group   n  r    p_hat  ci_lower  ci_upper  ci_width
     1  47  6 0.127660  0.059846  0.251739  0.191893
     2 148 19 0.128378  0.083750  0.191811  0.108061
     3 119  8 0.067227  0.034456  0.127065  0.092609
     4 810 34 0.041975  0.030191  0.058083  0.027892
     5 211 12 0.056872  0.032829  0.096762  0.063933
     6 196 13 0.066327  0.039168  0.110158  0.070990
     7 148  9 0.060811  0.032319  0.111524  0.079205
     8 215 30 0.139535  0.099520  0.192205  0.092685
     9 207 16 0.077295  0.048135  0.121857  0.073722
    10  97  3 0.030928  0.010573  0.087020  0.076447
    11 256 19 0.074219  0.048026  0.113001  0.064975
    12 360 27 0.075000  0.052054  0.106921  0.054867

================================================================================
1. POOLING ASSESSMENT
================================================================================

Completely Pooled Model:
  Total trials: 2814
  Total successes: 196
  Pooled success rate: 0.0697
  95% CI: [0.0608, 0.0797]

Completely Unpooled Model:
  Individual group rates range: [0.0309, 0.1395]
  Mean of individual rates: 0.0789
  Std of individual rates: 0.0348
  Coefficient of variation: 0.4412

Deviation from Pooled Rate:
  Mean absolute deviation: 0.0248
  Max deviation: 0.0699 (Group 8)
  Groups above pooled rate: 6
  Groups below pooled rate: 6

Chi-square test for homogeneity:
  Chi-square statistic: 39.52
  Degrees of freedom: 11
  P-value: 0.000043
  Conclusion: Strong evidence for heterogeneity (reject null of equal rates)

================================================================================
2. HIERARCHICAL STRUCTURE EVIDENCE
================================================================================

Variance Components (Logit Scale):
  Total variance in logit(p): 0.2354
  Mean within-group variance: 0.1044
  Estimated between-group variance (tau^2): 0.1311
  Estimated tau: 0.3620
  Approximate ICC: 0.5567
  Interpretation: Moderate to high between-group variation - hierarchical model recommended

Shrinkage Analysis:
  Mean shrinkage toward group mean: 39.1%
  Range of shrinkage: [19.0%, 72.4%]

  Groups by shrinkage (highest to lowest):
 group   n  r    p_hat  shrinkage_pct
    10  97  3 0.030928      72.408589
     1  47  6 0.127660      59.310893
     3 119  8 0.067227      50.554183
     7 148  9 0.060811      47.440496
     5 211 12 0.056872      40.267373
     6 196 13 0.066327      38.596521
     9 207 16 0.077295      34.071111
     2 148 19 0.128378      31.539362
    11 256 19 0.074219      30.252449
    12 360 27 0.075000      23.400040
     8 215 30 0.139535      22.813032
     4 810 34 0.041975      18.977662

Effect of Partial Pooling:
  Mean absolute change in rates: 0.0105
  Max change: 0.0359 (Group 1)

================================================================================
3. PRIOR ELICITATION INSIGHTS
================================================================================

Success Rate Prior (p ~ Beta(alpha, beta)):
  Observed rate range: [0.0309, 0.1395]
  Median rate: 0.0707
  IQR: [0.0598, 0.0899]

  Weakly Informative Prior Suggestions:
    Option 1 - Uniform: Beta(1, 1) [completely flat]
    Option 2 - Jeffreys: Beta(0.5, 0.5) [non-informative]
    Option 3 - Weak: Beta(2, 2) [mild peak at 0.5]
    Option 4 - Data-informed: Beta(5, 50) [mean ~ 0.091]
    Option 5 - Method of Moments: Beta(4.65, 54.36)
      [matches empirical mean=0.079, var=0.0012]

Hierarchical Variance Prior (tau ~ Half-Cauchy or Half-Normal):
  Estimated tau (logit scale): 0.3620
  Suggested Half-Cauchy scale: 1.00
  Suggested Half-Normal scale: 1.00

================================================================================
4. EXTREME GROUPS
================================================================================

Extreme Success Rates:
 group   n  r    p_hat  z_score_rate
     8 215 30 0.139535      1.744281

Extreme Sample Sizes:
 group   n  r    p_hat  z_score_n
     4 810 34 0.041975    2.90083

High Influence Groups (large n AND extreme rate):
 group   n  r    p_hat  influence_score
     4 810 34 0.041975         0.305117
     8 215 30 0.139535         0.133270
     2 148 19 0.128378         0.074873

Small Sample Groups (potential instability):
 group   n  r    p_hat  ci_width
     1  47  6 0.127660  0.191893
     3 119  8 0.067227  0.092609
    10  97  3 0.030928  0.076447
  Note: These groups have wide confidence intervals and may benefit most from pooling

================================================================================
5. TEMPORAL/SPATIAL PATTERNS
================================================================================

Correlation with Group Number:
  Success rate vs group: rho=-0.112, p=0.7292
  Sample size vs group: rho=0.452, p=0.1403
  Interpretation: No significant temporal/ordering trend

Runs Test for Randomness:
  Number of runs: 5
  Expected runs: 7.0
  Z-score: -1.211
  Interpretation: Data appears random (no clustering pattern)

================================================================================
6. DATA QUALITY CHECKS
================================================================================

Missing Values:
group                    0
n                        0
r                        0
p_hat                    0
failures                 0
ci_lower                 0
ci_upper                 0
ci_width                 0
deviation_from_pooled    0
abs_deviation            0
relative_deviation       0
logit_p                  0
se_logit                 0
shrinkage_factor         0
shrinkage_pct            0
logit_p_pooled           0
p_hat_pooled             0
change_from_pooling      0
z_score_rate             0
z_score_n                0
influence_score          0
dtype: int64

Data Validity:
  All trials > 0: True
  All successes valid (0 <= r <= n): True

Potential Issues:
  No major data quality issues detected

================================================================================
ANALYSIS COMPLETE - Proceeding to visualizations
================================================================================

Processed data saved to: /workspace/eda/analyst_2/code/processed_data.csv
