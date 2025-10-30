# Bayesian Modeling Experiment Plan
**Date:** 2024-10-27
**Dataset:** Y vs x (n=27)
**Synthesis:** Combined recommendations from 3 independent model designers

---

## Executive Summary

Three independent model designers proposed 9 total models across different modeling paradigms. After synthesis and removal of duplicates, we have **4 distinct model classes** to evaluate, prioritized by theoretical justification and computational feasibility.

### Key Insights from Parallel Design Process

**Convergent Findings (all 3 designers agreed):**
- Logarithmic functional form is primary candidate (all 3 ranked it #1)
- Robust likelihood desirable due to outlier at x=31.5
- Change point at x≈7 warrants investigation
- Homoscedastic errors reasonable starting assumption
- With n=27, models must balance flexibility vs parsimony

**Divergent Findings (unique perspectives):**
- Designer #1: Emphasized falsification criteria and stress testing
- Designer #2: Focused on mechanistic interpretation (saturation models)
- Designer #3: Prioritized robustness features (Student-t likelihood)

**Models Proposed by Multiple Designers:**
- Logarithmic model: All 3 designers (PRIMARY consensus)
- Change-point/segmented model: Designers #1 and #3
- Michaelis-Menten/rational function: Designers #1 and #2 (different parameterizations)

---

## Experiment Prioritization

### Minimum Attempt Policy
- **MUST attempt:** Model 1 (unless pre-fit validation fails)
- **MUST attempt:** Model 2 (unless Model 1 fails pre-fit validation)
- **MAY attempt:** Models 3-4 depending on results and time

### Model Selection Rationale

Based on:
1. **EDA support:** Strong evidence from exploratory analysis
2. **Designer consensus:** Agreement across independent reviewers
3. **Computational feasibility:** Likelihood of successful fitting with n=27
4. **Scientific interpretability:** Mechanistic meaning of parameters
5. **Falsifiability:** Clear criteria for rejection

---

## Model 1: Robust Logarithmic Regression ⭐ PRIMARY

**Priority:** 1 (MUST ATTEMPT)
**Proposed by:** All 3 designers (unanimous)
**Estimated Success:** 80%

### Mathematical Specification

```
Likelihood:
Y_i ~ StudentT(ν, μ_i, σ)

Mean function:
μ_i = α + β·log(x_i + c)

Parameters: α (intercept), β (slope), c (shift), ν (degrees of freedom), σ (scale)
```

### Prior Distributions

```stan
// Synthesis of all three designer recommendations
alpha ~ normal(2.0, 0.5);        // Conservative: Y centered ~2.3
beta ~ normal(0.3, 0.3);         // Positive slope, EDA suggests ~0.27-0.30
c ~ gamma(2, 2);                 // Mean=1, constrains to [0.1, 5]
nu ~ gamma(2, 0.1);              // Mean=20, allows adaptation to outliers
sigma ~ half_cauchy(0, 0.2);     // Expect small residuals given R²=0.89
```

**Prior Justification:**
- **α:** Data mean Y=2.33, prior allows ±1σ coverage of range [1.8, 2.8]
- **β:** OLS estimate ~0.27, prior centered slightly higher with uncertainty
- **c:** Standard log(x+1) has c=1; learning c allows data to refine
- **ν:** Student-t df; if ν>30 recovers Gaussian, if ν<10 indicates heavy tails
- **σ:** Residual scale; should be less than marginal SD(Y)=0.27

### Why This Model Should Work

**Empirical Evidence:**
- EDA R²=0.888 (best simple functional form)
- Log transformation linearizes relationship
- Only 5 parameters with n=27 (conservative)

**Theoretical Justification:**
- Diminishing returns pattern (common in learning curves, dose-response)
- No unbounded growth (log increases slowly at large x)
- Robust to outlier at x=31.5 via Student-t likelihood

**Computational Advantages:**
- Well-behaved geometry (nearly conjugate)
- Fast sampling expected
- Robust to initialization

### Falsification Criteria

**Abandon this model if:**

1. **Posterior Predictive Check fails**
   - Test: Systematic residual patterns (U-shape, trend vs x)
   - Threshold: Runs test p < 0.05 or visual clustering
   - Action: Move to Model 3 (Splines)

2. **Student-t df indicates extreme tails**
   - Test: Posterior ν < 5
   - Interpretation: Multiple outliers or systematic misspecification
   - Action: Investigate data quality; try mixture model

3. **Change-point model strongly preferred**
   - Test: ΔWAIC(Model 2 - Model 1) > 6
   - Interpretation: x≈7 breakpoint is real, not log curvature artifact
   - Action: Accept Model 2

4. **Log shift parameter at boundary**
   - Test: Posterior c > 4 or c < 0.2
   - Interpretation: Log transformation fundamentally wrong
   - Action: Try power law or Model 3

5. **Replicate prediction failure**
   - Test: Coverage < 60% on replicated x values
   - Interpretation: Model missing systematic variance structure
   - Action: Consider heteroscedastic model

### Computational Plan

**Stan Implementation:**
- 4 chains × 2000 iterations (1000 warmup)
- Target: Rhat < 1.01, ESS > 400 for all parameters
- Expected: 0-5% divergent transitions

**Stress Tests:**
1. Outlier sensitivity (refit dropping x=31.5)
2. Prior sensitivity (double all prior SDs)
3. Leave-one-out cross-validation (LOO-CV)

**Expected Runtime:** < 5 minutes

---

## Model 2: Change-Point / Segmented Regression ⭐ SECONDARY

**Priority:** 2 (MUST ATTEMPT)
**Proposed by:** Designers #1 and #3
**Estimated Success:** 60%

### Mathematical Specification

```
Likelihood:
Y_i ~ StudentT(ν, μ_i, σ)

Mean function (continuous piecewise linear):
μ_i = α + β₁·x_i                      if x_i ≤ τ
μ_i = α + β₁·τ + β₂·(x_i - τ)        if x_i > τ

Parameters: α, β₁ (steep slope), β₂ (flat slope), τ (change point), ν, σ
```

### Prior Distributions

```stan
alpha ~ normal(1.8, 0.3);        // Low x intercept
beta_1 ~ normal(0.15, 0.1);      // Steep initial slope
beta_2 ~ normal(0.02, 0.05);     // Near-flat plateau slope
tau ~ uniform(5, 12);            // Constrained around EDA x≈7
nu ~ gamma(2, 0.1);              // Robustness to outliers
sigma ~ half_cauchy(0, 0.2);     // Residual scale
```

**Prior Justification:**
- **τ:** EDA found 66% RSS improvement at x≈7; prior constrains to [5,12]
- **β₁ vs β₂:** Enforce expectation that initial slope steeper than plateau
- **Continuity:** Model enforces continuous function at τ

### Why This Model Might Win

**EDA Evidence:**
- 66% RSS reduction with breakpoint at x=7 (strongest signal in EDA!)
- Visual inspection shows potential regime change
- Diminishing returns could be two linear phases

**Theoretical Plausibility:**
- Phase transition in underlying process
- Threshold effect (resource depletion, saturation onset)
- Two distinct regimes with sharp boundary

### Falsification Criteria

**Abandon this model if:**

1. **Change point posterior is diffuse**
   - Test: 95% credible interval for τ spans [5,12] (full prior range)
   - Interpretation: Data don't identify τ
   - Action: Accept Model 1 (smooth log)

2. **Slopes not significantly different**
   - Test: P(β₁ > β₂) < 0.80
   - Interpretation: No evidence for regime change
   - Action: Use single-regime model

3. **Change point at boundary**
   - Test: Posterior τ < 5.5 or τ > 11.5
   - Interpretation: τ degenerate, seeking boundary
   - Action: Model inappropriate

4. **WAIC worse than log model**
   - Test: ΔWAIC(Model 2 - Model 1) > 0
   - Interpretation: Extra complexity not justified
   - Action: Reject segmented model

5. **Few points define change point**
   - Test: < 10 observations on either side of τ
   - Interpretation: Unstable estimate
   - Action: Insufficient data for this model

### Computational Challenges

**Known Issues:**
- Change point models have challenging geometry
- τ parameter can have low ESS (sticky sampling)
- Discontinuous derivative at τ complicates HMC

**Mitigation:**
- Use Student-t to robustify against influential points near τ
- Tight prior on τ (uniform [5,12] vs wider range)
- Longer warmup (1500 iterations)
- Accept ESS > 200 for τ (lower threshold)

**Expected Runtime:** 5-10 minutes

---

## Model 3: Adaptive B-Spline (Flexible Non-Parametric) 🔍 TERTIARY

**Priority:** 3 (if Models 1-2 inadequate)
**Proposed by:** Designer #3
**Estimated Success:** 70% (if needed)

### Mathematical Specification

```
Likelihood:
Y_i ~ StudentT(ν, μ_i, σ)

Mean function:
μ_i = Σ_k w_k · B_k(x_i)    (cubic B-spline basis)

Hierarchical smoothness prior:
w_k ~ normal(w_{k-1}, τ)     (random walk on weights)
τ ~ half_cauchy(0, 0.1)      (learned smoothness)

Parameters: w₁,...,w₈ (spline weights), τ (smoothness), ν, σ
```

### Prior Distributions

```stan
// Spline weights (8 basis functions for 4 internal knots)
w[1] ~ normal(2.0, 0.5);              // First weight near data mean
for (k in 2:8):
    w[k] ~ normal(w[k-1], tau);       // Random walk prior

// Smoothness parameter (critical for preventing overfitting)
tau ~ half_cauchy(0, 0.1);            // Small = smooth, large = wiggly

// Robustness and error
nu ~ gamma(2, 0.1);
sigma ~ half_cauchy(0, 0.2);
```

**Knot Placement:**
- Boundary knots: [1.0, 31.5] (data range)
- Internal knots: At quartiles of x [5.0, 9.5, 13.0, 15.5]
- Cubic basis: C² continuous (smooth curves)

### Why Use This Model

**When to Deploy:**
- Models 1-2 show systematic residual patterns
- Posterior predictive checks fail for parametric models
- Functional form fundamentally uncertain

**Advantages:**
- No functional form assumption (data-driven shape)
- Hierarchical prior prevents overfitting despite 8 parameters
- Can capture change point smoothly without explicit τ
- Student-t provides outlier robustness

**Limitations:**
- Less interpretable (no single β parameter)
- Higher computational cost
- Effective df depends on learned τ (may be ~4-5, not 8)

### Falsification Criteria

**Abandon this model if:**

1. **Effective parameters > 8**
   - Test: Compute effective sample size from τ posterior
   - Interpretation: Overfitting with n=27
   - Action: Too flexible; constrain further or reject

2. **ΔWAIC vs Model 1 < 2**
   - Interpretation: Flexibility not justified by data
   - Action: Use simpler Model 1

3. **Posterior predictive checks still fail**
   - Interpretation: Problem is not functional form (maybe heteroscedasticity)
   - Action: Investigate error structure

4. **Computational pathologies**
   - Test: >10% divergent transitions or Rhat > 1.02
   - Interpretation: Model too complex for n=27
   - Action: Reduce basis functions or use GP

### Computational Plan

**Stan Implementation:**
- B-spline basis precomputed in R/Python
- 4 chains × 3000 iterations (1500 warmup)
- Monitor effective parameters: p_eff = σ² / τ²

**Expected Runtime:** 10-15 minutes

---

## Model 4: Michaelis-Menten Saturation (Mechanistic) 🔬 EXPLORATORY

**Priority:** 4 (optional, if saturation interpretation critical)
**Proposed by:** Designers #1 and #2
**Estimated Success:** 50%

### Mathematical Specification

```
Likelihood:
Y_i ~ Normal(μ_i, σ²)    [or StudentT if needed]

Mean function:
μ_i = Y_min + Δ · x_i/(K + x_i)

Parameters: Y_min (baseline), Δ (dynamic range), K (half-saturation), σ
```

### Prior Distributions

```stan
Y_min ~ normal(1.8, 0.2);        // Tight around observed minimum
Delta ~ normal(1.0, 0.5);        // Y_max - Y_min ≈ 0.95
K ~ gamma(5, 0.5);               // Mean=10, mode≈8
sigma ~ half_cauchy(0, 0.2);     // Small residuals expected
```

### Why Consider This Model

**Mechanistic Interpretation:**
- Explicit asymptote: Y_max = Y_min + Δ
- Half-saturation constant K (where μ = midpoint)
- Common in enzyme kinetics, dose-response, resource saturation

**When to Use:**
- If estimating maximum response (Y_max) is scientifically important
- If process is known to saturate (biological, chemical systems)
- If log model criticized for unbounded growth

### Falsification Criteria

**Abandon this model if:**

1. **Y_max posterior unbounded**
   - Test: 95% CI for (Y_min + Δ) > 1.5 units wide
   - Interpretation: No true asymptote in data range
   - Action: Use log model

2. **K posterior degenerate**
   - Test: K < 1 or K > 50
   - Interpretation: Saturation not evident in data
   - Action: Model unidentifiable

3. **ΔWAIC vs log model > 10**
   - Interpretation: Extra complexity unjustified
   - Action: Use simpler log model

4. **Computational failure**
   - Test: Convergence issues, high divergences
   - Interpretation: Non-linear optimization difficult
   - Action: Try reparameterization or abandon

### Computational Challenges

**Non-linear model difficulties:**
- Parameter correlations (K and Δ)
- Weak identification if max(x) not near asymptote
- Requires good initialization

**Mitigation:**
- Informative priors essential
- Non-centered parameterization: μ = Y_min + Δ/(1 + K/x)
- Careful initial values from EDA

**Expected Runtime:** 5-10 minutes

---

## Model Comparison Strategy

### Workflow

```
1. FIT ALL MODELS (1-2 required, 3-4 optional)
   ↓
2. CONVERGENCE DIAGNOSTICS (Rhat, ESS, divergences)
   ↓  [If any model fails: document and move to next]
   ↓
3. POSTERIOR PREDICTIVE CHECKS (residuals, replicates)
   ↓
4. COMPUTE WAIC / LOO-CV
   ↓
5. APPLY DECISION RULES (below)
```

### Decision Rules

**If ΔWAIC < 2:**
- Models equivalent by parsimony principle
- **Choose simplest model** (fewest parameters)
- Report uncertainty across model space

**If 2 ≤ ΔWAIC ≤ 6:**
- Weak preference for best model
- **Report both models** with caveats
- Conduct sensitivity analysis

**If ΔWAIC > 6:**
- Strong preference for best model
- **Select best model** as primary
- Secondary models relegated to supplement

### Falsification Framework

**Global red flags (reconsider entire model class):**

1. **All models fail diagnostics**
   → Data pathological or priors fundamentally wrong

2. **All models show systematic residuals**
   → Missing crucial feature (heteroscedasticity, non-monotonicity)

3. **All models equivalent by WAIC**
   → Data insufficient to distinguish; report uncertainty

4. **All posterior predictive checks fail**
   → Wrong likelihood family or error structure

5. **Extreme computational issues across all models**
   → Geometry problems suggest model class inappropriate

**Escape routes if all 4 models fail:**
- Gaussian Process (fully non-parametric)
- Generalized Additive Model (GAM)
- Heteroscedastic models (if variance structure issue)
- Transformation of Y (if Y scale inappropriate)

---

## Success Criteria

### Model-Level (each model evaluated independently)

**Pre-fit validation:**
- [ ] Prior predictive check: simulations plausible
- [ ] Simulation-based calibration: model can recover parameters

**Fit diagnostics:**
- [ ] Rhat < 1.01 for all parameters
- [ ] ESS > 400 (or ESS > 200 for change point τ)
- [ ] Divergent transitions < 5% (or < 10% for complex models)
- [ ] Trace plots show good mixing

**Post-fit validation:**
- [ ] Posterior predictive checks pass (p-value ∈ [0.05, 0.95])
- [ ] Residuals show no systematic patterns
- [ ] Replicate prediction coverage ≥ 80%
- [ ] LOO diagnostics: Pareto k < 0.7 for all observations

### Project-Level (overall adequacy)

**Minimum requirement:**
- At least 1 model passes all diagnostics and validation

**Adequate solution:**
- Best model has scientifically interpretable parameters
- Uncertainty quantification is well-calibrated
- Predictions are robust to prior specifications
- Model limitations are clearly documented

**Stopping criteria:**
- Adequate model found → Proceed to reporting
- All models fail → Refine or try alternative approaches
- Models equivalent → Report model uncertainty

---

## Timeline and Resource Allocation

### Phase 3A: Model 1 (Robust Log) - Day 1
- Prior predictive check: 1 hour
- Simulation validation: 2 hours
- Model fitting: 1 hour
- Posterior predictive check: 2 hours
- **Decision point:** ACCEPT / REVISE / REJECT

### Phase 3B: Model 2 (Change Point) - Day 2
- Prior predictive check: 1 hour
- Simulation validation: 2 hours
- Model fitting: 2 hours (longer due to complexity)
- Posterior predictive check: 2 hours
- **Decision point:** ACCEPT / REVISE / REJECT

### Phase 3C: Model 3-4 (Optional) - Day 3
- Only if Models 1-2 both fail or show inadequacy
- Same validation pipeline

### Phase 4: Model Comparison - Day 4
- WAIC/LOO comparison
- Sensitivity analyses
- Model selection decision

### Phase 5: Adequacy Assessment - Day 4
- Overall evaluation
- Refinement planning if needed

### Phase 6: Reporting - Day 5
- Final report synthesis
- Key visualizations
- Recommendations

**Total estimated time:** 4-5 days

---

## Key Uncertainties and Risks

### Scientific Uncertainties

1. **Is the change point at x≈7 real or artifact?**
   - Models 1 vs 2 directly test this
   - Strong EDA evidence (66% RSS) but n=27 small

2. **Does Y truly saturate or continue growing?**
   - Log model (unbounded) vs MM model (asymptote)
   - Data max(x)=31.5 may not extend far enough to tell

3. **Is the outlier at x=31.5 genuine?**
   - Student-t likelihood provides automatic down-weighting
   - Sensitivity analysis will quantify influence

### Computational Risks

1. **Change point model may not converge**
   - Mitigation: Tight prior, longer warmup
   - Backup: Accept Model 1 if Model 2 fails

2. **Non-linear models (MM) may have identification issues**
   - Mitigation: Informative priors
   - Backup: Report as exploratory

3. **Small sample (n=27) limits model complexity**
   - Mitigation: Start with simplest models
   - Conservative: Prefer parsimony when ΔWAIC < 2

### Decision Risks

1. **Type I error:** Accepting inadequate model
   - Mitigation: Rigorous falsification criteria
   - Multiple validation steps (PPC, LOO, residuals)

2. **Type II error:** Rejecting adequate model prematurely
   - Mitigation: Clear thresholds (ΔWAIC > 6 for strong rejection)
   - Sensitivity analyses before abandoning

---

## Summary

**Primary Strategy:**
1. Fit robust logarithmic model (Model 1) - 80% likely sufficient
2. Test change-point hypothesis (Model 2) - addresses strongest EDA signal
3. Deploy flexible models (3-4) only if 1-2 inadequate

**Expected Outcome:**
- Model 1 likely adequate with Student-t capturing outlier
- Model 2 may reveal change point is real
- Model comparison quantifies functional form uncertainty

**Contingency Plans:**
- All models have explicit falsification criteria
- Clear escape routes if model class inappropriate
- Adequacy assessment prevents premature acceptance

**Next Step:** Begin Phase 3 with Model 1 prior predictive checking.
