# Implementation Priority Guide - Time-Series Models

**Designer 3 - Quick Reference**

## Priority Order (Start Here)

### 1. Model 1: Dynamic Linear Model (PRIMARY)
**Why first**: Best alignment with EDA findings (discrete regime change + strong autocorrelation)

**Quick Start**:
```bash
# File: model_1_dlm.stan
# Estimated runtime: 15-30 minutes
# Critical parameters: φ (velocity persistence), τ (changepoint), Δδ (regime shift)
```

**First Implementation**: Fix τ=17 (from EDA) to avoid discrete parameter sampling complexity. This simplifies to:
- 7 continuous parameters (instead of 8)
- 2N state variables (η_t, δ_t for t=1:40)
- Total: 87 parameters

**Key Diagnostic**: After fitting, check if posterior for φ is away from 0 (if φ ≈ 0, velocity AR is unnecessary, pivot to simpler model).

### 2. Model 3: Gaussian Process (VALIDATION)
**Why second**: Tests whether discrete changepoint is necessary or if smooth transition fits better

**Quick Start**:
```bash
# File: model_3_gp.stan
# Estimated runtime: 20-40 minutes (slower due to Cholesky decomposition)
# Critical parameters: length_scale (smoothness), σ_f (signal strength)
```

**Key Diagnostic**: Plot posterior mean f(year) trajectory. Does it show a visible kink/acceleration at year ≈ -0.2? If smooth, changepoint may be artifact.

### 3. Model 2: NB-AR (BASELINE)
**Why last**: Simplest, but may conflate AR term with polynomial trend. Good sanity check.

**Quick Start**:
```bash
# File: model_2_nbar.stan
# Estimated runtime: 10-20 minutes (fastest)
# Critical parameters: γ (AR coefficient), β_3 (post-break slope)
```

**Key Diagnostic**: Check posterior correlation between γ and β_2. If |corr| > 0.7, identifiability problem.

---

## Falsification Test Checklist

Run these tests on ALL models after fitting:

### Test 1: Autocorrelation Capture
```python
# Compute Pearson residuals
r_t = (C - mu) / sqrt(mu + alpha * mu^2)

# Calculate ACF
acf_1 = acf(r_t, lag=1)

# PASS if acf_1 < 0.4
# FAIL if acf_1 > 0.5
```

### Test 2: Structural Break in Posterior Predictive
```python
# Generate 1000 datasets from posterior predictive
for i in 1:1000:
    C_rep = posterior_predictive_draw(i)
    p_value[i] = chow_test(C_rep, breakpoint=17)

# PASS if mean(p_value < 0.05) > 0.80
# FAIL if mean(p_value < 0.05) < 0.60
```

### Test 3: Overdispersion Adequacy
```python
# Observed ratio
obs_ratio = var(C) / mean(C)  # = 67.99

# Posterior predictive ratios
pp_ratios = []
for i in 1:1000:
    C_rep = posterior_predictive_draw(i)
    pp_ratios[i] = var(C_rep) / mean(C_rep)

# PASS if obs_ratio in quantile(pp_ratios, [0.05, 0.95])
```

### Test 4: Out-of-Sample Prediction
```python
# Fit on t=1:30, predict t=31:40
model_train = fit_model(C[1:30], year[1:30])
pred_interval = predict(model_train, year[31:40], interval=0.90)

# Count misses
misses = sum(C[31:40] < pred_interval[:, 0] or C[31:40] > pred_interval[:, 1])

# PASS if misses <= 3/10
# FAIL if misses >= 5/10
```

---

## Red Flag Triggers (Stop and Reassess)

### Computational Red Flags
- **Divergences > 5%**: Model geometry pathological, reparameterize or abandon
- **R-hat > 1.05**: Chains haven't converged, increase iterations or check initialization
- **Runtime > 1 hour**: For N=40, this is excessive. Consider approximate methods.

### Statistical Red Flags
- **All models fail Test 1 (ACF)**: Autocorrelation is more complex than AR(1)/GP/DLM
- **All models fail Test 2 (Break)**: Structural break may be illusory
- **All models fail Test 3 (Variance)**: Need more flexible distribution than NB

### Model-Specific Red Flags

**Model 1 (DLM)**:
- φ posterior near 0: Velocity AR unnecessary
- φ posterior near 1: Non-stationary, model misspecified
- σ_η >> σ_δ: State-space adds no value

**Model 2 (NB-AR)**:
- γ posterior near 0: AR term unnecessary (autocorrelation is spurious)
- |corr(γ, β_2)| > 0.7: Identifiability problem
- First observation has huge uncertainty: Boundary condition unstable

**Model 3 (GP)**:
- length_scale posterior near 0: Overfitting to noise
- length_scale posterior > 5: GP degenerates to linear trend
- σ_n / σ_f > 0.5: Nugget dominates, not capturing structure

---

## Decision Tree

```
START
  |
  v
Fit Model 1 (DLM with τ=17)
  |
  +-- Passes all 4 tests? --> DONE, use Model 1
  |
  +-- Fails Test 1 (ACF)? --> Pivot to ARMA(p,q) models
  |
  +-- Fails Test 2 (Break)? --> Fit Model 3 (GP) to check if smooth
  |
  +-- Computational failure? --> Try Model 2 (simpler)
  |
  v
Fit Model 3 (GP)
  |
  +-- Passes all 4 tests? --> Compare Model 1 vs 3 via LOO
  |
  +-- GP trajectory shows discrete break? --> Confirms Model 1
  |
  +-- GP trajectory smooth? --> Structural break is artifact
  |
  v
Fit Model 2 (NB-AR)
  |
  +-- Passes all 4 tests? --> Compare all 3 via LOO
  |
  +-- All 3 models fail? --> STOP, reconsider model classes
  |
  v
LOO Comparison (among survivors)
  |
  +-- ΔLOO > 2*SE? --> Use best model
  |
  +-- ΔLOO < 2*SE? --> Model stacking or averaging
  |
  v
Sensitivity Analysis (winning model)
  |
  +-- Prior sensitivity: 2x wider/narrower priors
  +-- Changepoint sensitivity: τ = 15, 17, 19
  +-- Subsampling: Remove 10% observations
  |
  v
FINAL MODEL SELECTION
```

---

## Expected Outcomes (Predictions)

Based on EDA evidence, my predictions are:

### Model 1 (DLM):
- **Will likely pass** all tests
- φ posterior will be high (0.7-0.9), confirming velocity persistence
- Δδ posterior will be large and positive (0.6-1.2), confirming regime shift
- Computational challenges with discrete τ, but fixable by fixing τ=17
- **Predicted LOO rank**: 1st

### Model 3 (GP):
- **Will likely pass** Tests 1, 3, 4 but struggle with Test 2
- length_scale posterior will be moderate (0.8-2.0)
- GP trajectory will show gradual acceleration, not sharp break
- May over-smooth the structural break
- **Predicted LOO rank**: 2nd (close to Model 1)

### Model 2 (NB-AR):
- **May fail** Test 1 (residual ACF still high)
- γ posterior will be positive but may not fully explain ACF(1)=0.944
- Identifiability issues with γ vs β_2
- **Predicted LOO rank**: 3rd

### If All Three Fail:
- Likely culprit: Autocorrelation is more complex (ARMA, long memory, or regime-dependent)
- Next step: Fit ARMA(2,1) or Hidden Markov Model

---

## Computational Tips

### Stan Compilation
```bash
# Use cmdstan for faster compilation
install_cmdstan()

# Or use pystan with caching
model = pystan.StanModel(file='model_1_dlm.stan', model_name='dlm_cached')
```

### Non-Centered Parameterization (Critical for Model 1 & 3)
```stan
// BAD (centered):
eta[t] ~ normal(eta[t-1] + delta[t], sigma_eta);

// GOOD (non-centered):
eta_raw[t] ~ std_normal();
eta[t] = eta[t-1] + delta[t] + eta_raw[t] * sigma_eta;
```

### Initialization
```python
# Model 1 (DLM)
init_dict = {
    'eta_0': 3.0,
    'delta_0': 0.3,
    'phi': 0.8,
    'Delta_delta': 0.8,
    'sigma_eta': 0.1,
    'sigma_delta': 0.1,
    'alpha': 0.6
}

# Model 3 (GP)
init_dict = {
    'beta_0': 4.3,
    'sigma_f': 1.0,
    'length_scale': 1.5,
    'sigma_n': 0.1,
    'alpha': 0.6
}
```

### Sampling Parameters
```python
# Start conservative
fit = model.sampling(
    data=data_dict,
    iter=2000,
    warmup=1000,
    chains=4,
    thin=1,
    control={
        'adapt_delta': 0.95,  # High for GP and DLM
        'max_treedepth': 12   # Increase if hitting limit
    },
    init=init_dict
)

# If divergences persist, increase adapt_delta to 0.99
```

---

## File Organization

```
experiments/designer_3/
├── proposed_models.md              (main document)
├── implementation_priority.md      (this file)
├── models/
│   ├── model_1_dlm.stan           (Dynamic Linear Model)
│   ├── model_2_nbar.stan          (NB-AR)
│   ├── model_3_gp.stan            (Gaussian Process)
├── scripts/
│   ├── fit_model_1.py             (Fitting script for DLM)
│   ├── fit_model_2.py             (Fitting script for NB-AR)
│   ├── fit_model_3.py             (Fitting script for GP)
│   ├── falsification_tests.py     (All 4 tests)
│   ├── posterior_predictive.py    (PPC plots)
│   └── compare_models.py          (LOO comparison)
├── results/
│   ├── model_1_diagnostics.html   (R-hat, ESS, trace plots)
│   ├── model_1_posterior.csv      (Posterior draws)
│   ├── model_1_loo.pkl            (LOO object)
│   └── ...
└── figures/
    ├── acf_comparison.png         (Residual ACF for all models)
    ├── posterior_trajectories.png (η_t or f(year) draws)
    ├── pp_check_break.png         (Posterior predictive Chow test)
    └── loo_comparison.png         (Model comparison plot)
```

---

## Timeline Estimate

**Week 1**: Model Implementation & Initial Fitting
- Day 1: Implement Model 1 Stan code
- Day 2: Fit Model 1, diagnose convergence
- Day 3: Implement Model 3 Stan code
- Day 4: Fit Model 3, diagnose convergence
- Day 5: Implement Model 2, fit, diagnose

**Week 2**: Falsification & Comparison
- Day 1-2: Run all 4 falsification tests on all models
- Day 3: LOO comparison among survivors
- Day 4-5: Sensitivity analysis on leading model

**Week 3**: Validation & Reporting
- Day 1: Out-of-sample prediction (time-series CV)
- Day 2: Posterior inference and interpretation
- Day 3: Create publication-quality figures
- Day 4-5: Write-up and documentation

**Total**: 15 working days (3 weeks)

---

## Contact for Help

**If stuck on**:
- Stan syntax errors: Check Stan documentation, use Stan forums
- Divergences: Try non-centered parameterization, increase adapt_delta
- Slow sampling: Consider Hilbert space GP, or fix τ in Model 1
- All models fail: Contact main modeling strategist, propose alternative model classes

**Remember**: The goal is truth, not task completion. If all models fail, that's a scientific result worth reporting.

---

**Quick Start Command** (when ready to implement):

```bash
cd /workspace/experiments/designer_3
mkdir -p models scripts results figures
# Start with Model 1
vim models/model_1_dlm.stan
python scripts/fit_model_1.py
# Check diagnostics before proceeding
```

---

**End of Implementation Priority Guide**
