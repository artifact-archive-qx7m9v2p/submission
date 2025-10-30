# EDA Log - Analyst 1: Distributional Properties and Count Characteristics

## Round 1: Initial Exploration

### Data Overview
- **Sample size**: 40 observations
- **Variables**: 2 (year, C)
- **Data quality**: Clean - no missing values, no negative values, all counts are integers
- **Unique count values**: 34 out of 40 observations

### Key Findings from Round 1

#### 1. Distribution Shape and Central Tendency
**Evidence**: See `01_distribution_overview.png`

- **Mean**: 109.4
- **Median**: 67.0
- **Standard deviation**: 87.78
- **Range**: 21 to 269 (span of 248)

**Key Insight**: The mean (109.4) is substantially larger than the median (67.0), indicating a **right-skewed distribution**. This is confirmed by:
- Positive skewness coefficient: 0.64
- Platykurtic distribution (negative excess kurtosis: -1.13), suggesting lighter tails than normal

**Interpretation**: The histogram shows a bimodal-like pattern with concentration in lower values (20-70) and higher values (160-270), with fewer observations in the middle range. This is NOT a typical count distribution.

#### 2. CRITICAL FINDING: Severe Overdispersion
**Evidence**: Numerical output and `03_variance_mean_relationship.png`

- **Variance**: 7,704.66
- **Mean**: 109.4
- **Variance-to-Mean Ratio**: **70.43**

This is **extreme overdispersion** - the variance is more than 70 times the mean!

- Chi-square test for overdispersion: p < 0.000001 (highly significant)
- **Poisson distribution is completely inappropriate** for this data

**Interpretation**: The variance-mean plot clearly shows all groups fall far above the Poisson identity line. This level of overdispersion suggests:
1. Negative Binomial distribution may be suitable, but even this may struggle
2. The data generation process involves substantial heterogeneity
3. Alternative explanations: mixing of populations, time trends, or unmeasured covariates

#### 3. Temporal Trend Dominates the Data
**Evidence**: `02_temporal_pattern.png` and regression output

- **Pearson correlation**: 0.939 (very strong positive)
- **Spearman correlation**: 0.954 (even stronger, suggesting monotonic relationship)
- **Linear regression R²**: 0.88 (88% of variance explained by year)
- **Slope**: 82.4 counts per standardized year unit

**Critical Insight**: The overwhelming majority of variance in C is explained by the temporal trend. This has major implications:
- The "overdispersion" may actually be **trend-induced heterogeneity**
- Count models should incorporate year as a covariate
- Examining C alone without accounting for year is misleading

**Hypothesis**: Much of the apparent overdispersion comes from pooling data across different time periods with different expected counts.

#### 4. No Outliers Detected
**Evidence**: Numerical output and `01_distribution_overview.png` boxplot

- IQR method: 0 outliers (all values within 1.5*IQR fences)
- Z-score method: 0 outliers (no |z| > 3)

**Interpretation**: The wide range of values (21-269) represents genuine variation in the data generation process, not measurement errors or anomalies. The temporal trend explains this variation.

#### 5. Zero-Inflation Assessment
**Evidence**: Numerical output

- **Number of zeros**: 0 (0%)
- **Minimum value**: 21

**Interpretation**: No zero-inflation present. All counts are at least 21, which is unusual for typical count data. This suggests:
- The process being measured has a guaranteed minimum level
- Or the data represents counts from an already-established process (not starting from zero)

### Theoretical Distribution Assessment

**Evidence**: `04_theoretical_distributions.png`

**Poisson fit**: Completely inadequate. The Poisson distribution with λ=109.4 is far too narrow to capture the spread in the data. This confirms the overdispersion finding.

**Negative Binomial fit**: Using method of moments:
- Estimated r (size parameter): 1.56
- Estimated p (probability parameter): 0.014

The NB distribution fits somewhat better than Poisson but still shows poor fit. The empirical distribution appears flatter and more bimodal than the theoretical NB.

### Questions for Round 2

1. **How does the variance change across time?** Is overdispersion constant or does it increase with the mean?
2. **What happens if we detrend the data?** Will residuals still show overdispersion?
3. **Is there evidence of different regimes?** The bimodal appearance suggests potential structural breaks
4. **What is the coefficient of variation across different time periods?**
5. **Can we identify a better parametric family by examining detrended residuals?**

### Hypotheses to Test in Round 2

**Hypothesis 1**: The overdispersion is primarily due to the temporal trend. After accounting for year, residuals will show much less overdispersion.

**Hypothesis 2**: The distribution changes systematically over time - both mean and variance increase with year.

**Hypothesis 3**: There may be a structural break or regime change in the middle of the time series.

---

## Round 2: Detrending and Deep Dive Analysis

### Hypothesis Testing Results

#### HYPOTHESIS 1: CONFIRMED (with important nuance)
**Evidence**: Detrending analysis output and `05_residual_diagnostics.png`

**Finding**: Detrending dramatically reduces variance, but substantial residual variation remains.

- **Original variance**: 7,704.66
- **Residual variance after linear detrending**: 915.05
- **Variance explained by trend**: **88.12%**

**Critical Insight**: While the temporal trend explains most of the variance, the residual variance (915.05) is still quite large relative to typical count data. The residual variance-to-predicted-mean ratio is 8.36, indicating moderate overdispersion remains even after accounting for the trend.

**Interpretation**:
1. The extreme apparent overdispersion (Var/Mean = 70.43) was indeed largely driven by pooling across different time periods
2. However, even after detrending, there's still meaningful overdispersion (ratio ≈ 8.4)
3. This suggests TWO sources of overdispersion:
   - **Between-time heterogeneity** (trend-induced) - explains 88% of variance
   - **Within-time variability** (residual) - explains 12% of variance but still substantial

#### HYPOTHESIS 2: CONFIRMED
**Evidence**: `08_temporal_distribution_changes.png` and heteroscedasticity analysis

**Finding**: Distribution systematically shifts over time:

**Early Period** (observations 1-13):
- Mean: 28.3
- SD: 4.3
- Coefficient of Variation (CV): 0.151

**Middle Period** (observations 14-26):
- Mean: 70.7
- SD: 22.5
- CV: 0.318

**Late Period** (observations 27-40):
- Mean: 215.6
- SD: 35.6
- CV: 0.165

**Critical Insight**: The coefficient of variation is NOT constant - it peaks in the middle period and is lower in early and late periods. This suggests:
- The data generation process is non-stationary
- Simple scaling relationships may not hold throughout
- There may be distinct regimes or a transition period

#### HYPOTHESIS 3: PARTIALLY CONFIRMED
**Evidence**: Heteroscedasticity analysis in `05_residual_diagnostics.png`

The variance structure is not uniform, but it's more complex than a simple break:

**Residual variance by quartile of predicted values**:
- Q1 (lowest predictions): Var = 525.49
- Q2: Var = 89.15 (much lower!)
- Q3: Var = 372.13
- Q4 (highest predictions): Var = 514.96

**Interpretation**: The middle range (Q2) shows much lower residual variance than the extremes. This creates a **U-shaped pattern** rather than monotonic heteroscedasticity. However, the Breusch-Pagan test shows no significant linear relationship between squared residuals and year (r = -0.15, p = 0.35).

### Major Discovery: Power Law Relationship

**Evidence**: `06_mean_variance_relationship.png` and power law analysis

**Finding**: Variance scales as the square of the mean!

**Power law fit**: Variance = 0.057 × Mean^2.01
- R² = 0.843 (excellent fit)
- p < 0.000001 (highly significant)

**Critical Interpretation**: This is a **quadratic mean-variance relationship**, which has profound implications:

1. **Not Poisson**: Poisson has Var = Mean (power = 1)
2. **Not standard Negative Binomial**: NB typically has Var = Mean + α×Mean²
3. **Suggests quasi-Poisson or NB with small dispersion parameter**

This relationship is characteristic of:
- Processes with multiplicative (rather than additive) variability
- Heterogeneous populations being aggregated
- Log-normal or gamma-distributed rates in an underlying Poisson process

### Model Comparison: Linear vs Log-Linear

**Evidence**: `07_model_comparison.png` and model comparison output

**Log-linear model**: log(C) = 4.33 + 0.86 × year
- R² = 0.937 (better than linear's 0.881)
- MSE = 482.25 (much better than linear's 892.18)
- Implied growth rate: 137% per standardized year unit

**Winner**: Log-linear model is superior by multiple criteria.

**Key Insight**: The data exhibits **exponential growth** over time, not linear growth. This is characteristic of:
- Population growth processes
- Compound accumulation
- Epidemic/diffusion processes

**Implication for modeling**: A Poisson or Negative Binomial GLM with log link and year as a covariate would be very natural for this data.

### Residual Analysis

**Evidence**: `05_residual_diagnostics.png`

**Normality of residuals** (from linear model):
- Shapiro-Wilk test: W = 0.955, p = 0.112
- **Cannot reject normality** (p > 0.05)

The Q-Q plot shows residuals are approximately normally distributed with:
- Good fit in the middle
- Slight deviation in the tails (but not severe)

**Residual patterns**:
- Residuals vs Fitted plot shows no clear systematic pattern
- Some suggestion of higher spread at extremes (U-shape mentioned earlier)
- Scale-Location plot confirms non-constant variance but no strong trend

### Summary of Round 2 Key Findings

1. **Overdispersion has two components**: 88% from temporal trend, 12% residual
2. **Power law relationship**: Var ∝ Mean² (quadratic scaling)
3. **Exponential growth**: Log-linear model superior to linear
4. **Non-stationary process**: Distribution characteristics change over time
5. **Residuals approximately normal**: After linear detrending, residuals pass normality test
6. **Complex variance structure**: U-shaped pattern across predicted values

---

## Final Recommendations for Modeling

Based on comprehensive analysis across two rounds, the following modeling approaches are recommended (in order of preference):

### Recommended Model Classes

#### 1. **Generalized Linear Model (GLM) with Negative Binomial distribution** (PREFERRED)
**Rationale**:
- Accounts for overdispersion (even after accounting for covariates)
- Natural log link matches the exponential growth pattern
- Can incorporate year as covariate to capture trend
- Variance function Var = μ + μ²/θ can accommodate the quadratic mean-variance relationship

**Model specification**:
- Response: C (count)
- Distribution: Negative Binomial
- Link: log
- Covariates: year (possibly year²)

**Advantages**:
- Handles overdispersion flexibly
- Widely supported in statistical software
- Allows for inference on growth rate

**Concerns to address**:
- Check if single dispersion parameter θ is adequate
- Validate assumptions on held-out data

#### 2. **Quasi-Poisson GLM**
**Rationale**:
- Relaxes variance = mean assumption while keeping Poisson-like structure
- Simpler than NB, appropriate when exact distribution is uncertain
- Can handle the observed mean-variance relationship

**Model specification**:
- Response: C
- Family: quasi-Poisson
- Link: log
- Covariates: year

**Advantages**:
- Robust to distribution misspecification
- Accounts for overdispersion via dispersion parameter
- Standard errors properly adjusted

**Concerns**:
- No full likelihood (can't use AIC for model selection)
- Less efficient than NB if NB is correct

#### 3. **Log-Linear Regression with Gaussian errors**
**Rationale**:
- Round 2 showed log(C) has approximately normal residuals
- Simpler to interpret than count models
- Best empirical fit in model comparison (lowest MSE)

**Model specification**:
- Response: log(C)
- Distribution: Normal
- Covariates: year

**Advantages**:
- Best fit among tested models
- Familiar interpretation
- Residuals approximately normal

**Concerns**:
- Back-transformation introduces bias (need smearing estimator)
- Doesn't respect count nature of data
- May predict non-integer or negative values on original scale

### Key Features Any Model Should Include

1. **Year as primary covariate**: Captures 88% of variance
2. **Consider non-linear time effects**: Possibly year² or splines
3. **Accommodate overdispersion**: Via NB, quasi-likelihood, or robust SEs
4. **Use log link**: Matches exponential growth pattern

### Diagnostics to Perform

1. **Residual plots**: Check for remaining patterns
2. **Dispersion parameter**: Estimate and interpret
3. **Influence diagnostics**: Though no outliers detected, check for influential points
4. **Cross-validation**: Assess predictive performance
5. **Rootogram**: For count models, check distributional fit

### Data Quality Issues to Address Before Modeling

**NONE IDENTIFIED**. The data appears clean and high quality:
- No missing values
- No impossible values (negatives, non-integers)
- No extreme outliers
- Temporal sequence appears intact
- No evidence of data entry errors

### Open Questions for Further Investigation

1. **What does the count variable represent?** Understanding the domain would help validate model choice
2. **Are there additional covariates?** Other variables might explain residual overdispersion
3. **Is the time trend expected to continue?** Extrapolation beyond observed range may be risky
4. **What explains the U-shaped variance pattern?** Domain knowledge might clarify

---

## Robust vs Tentative Findings

### ROBUST (High Confidence)
- Strong positive temporal trend (r = 0.94, p < 0.001)
- Severe overdispersion relative to Poisson
- Exponential growth pattern (log-linear superior to linear)
- No outliers or data quality issues
- Power law mean-variance relationship (power ≈ 2)

### TENTATIVE (Medium Confidence)
- U-shaped residual variance pattern (based on quartile analysis with small sample sizes)
- Distinct regimes in early/middle/late periods (CV differences could be sampling variation)
- Specific parameter values for NB distribution (based on method of moments, not MLE)

### REQUIRES FURTHER INVESTIGATION
- Whether non-linear time effects (year²) would improve fit
- Presence of additional unmeasured covariates
- Predictive performance on held-out data
- Whether dispersion parameter is constant over time

---

**Analysis complete. Ready for modeling phase.**
