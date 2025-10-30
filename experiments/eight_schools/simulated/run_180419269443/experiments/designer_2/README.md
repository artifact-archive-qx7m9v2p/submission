# Designer 2: Robust Bayesian Models for Meta-Analysis

**Designer:** Model Designer 2 (Independent)
**Focus:** Distributional robustness and sensitivity to assumptions
**Date:** 2025-10-28
**Status:** Design phase complete, ready for implementation

---

## Overview

This directory contains **adversarial model designs** that challenge the assumptions underlying standard meta-analysis. While EDA suggests low heterogeneity and normality, these models explore:

1. What if outliers emerge in future studies?
2. What if there are hidden subpopulations?
3. What if our distributional assumptions are fundamentally wrong?

**Philosophy:** Plan for failure, design for discovery, pivot quickly when evidence contradicts assumptions.

---

## File Structure

```
designer_2/
├── README.md                          # This file
├── proposed_models.md                 # Comprehensive model design document
├── model_1_t_distribution.stan        # Heavy-tailed hierarchical model
├── model_2_mixture.py                 # Two-component mixture model
├── model_3_dirichlet_process.py       # Non-parametric DP mixture
└── (to be created during implementation)
    ├── analysis_script.py             # Main analysis pipeline
    ├── validation_tests.py            # Falsification tests
    ├── sensitivity_analyses.py        # Prior/influence sensitivity
    └── results/                       # Outputs
        ├── convergence_diagnostics/
        ├── posterior_summaries/
        ├── model_comparisons/
        └── visualizations/
```

---

## Three Model Classes

### Model 1: Heavy-Tailed Hierarchical (t-Distribution)

**File:** `model_1_t_distribution.stan`

**Key idea:** Replace normal likelihood with Student-t to robust-ify against outliers.

**When to use:**
- Uncertainty about data quality
- Concern about extreme observations
- Want automatic downweighting of outliers

**Falsification criteria:**
- If nu > 50: Normal distribution adequate, don't need this
- If nu < 3: Over-explaining tails, check data
- If LOO worse than normal: Complexity not helping

**Expected outcome (given EDA):**
- Likely nu ≈ 30-50 (near normal)
- If so, validates standard approach
- If nu < 20, important robustness finding

---

### Model 2: Mixture Model (Heterogeneous Heterogeneity)

**File:** `model_2_mixture.py`

**Key idea:** Allow for two subpopulations with different means and variances.

**When to use:**
- Study 5's negative effect might signal different population
- Suspicion of hidden subgroups
- Want to test homogeneity assumption directly

**Falsification criteria:**
- If pi → 0 or 1: Single cluster, use hierarchical normal
- If mu_1 ≈ mu_2: Clusters not distinguishable
- If label switching unsolvable: Fundamental ID problem

**Expected outcome (given EDA):**
- Likely collapse to single cluster (pi ≈ 0 or 1)
- If so, validates low heterogeneity finding
- If genuine mixture, major discovery

---

### Model 3: Dirichlet Process (Non-Parametric)

**File:** `model_3_dirichlet_process.py`

**Key idea:** Don't fix number of clusters, let data decide.

**When to use:**
- Fundamental uncertainty about structure
- Sensitivity analysis to model choice
- Want minimal distributional assumptions

**Falsification criteria:**
- If K_eff = 1: Use hierarchical normal
- If K_eff = J: No pooling, reconsider approach
- If LOO no better than simpler models: Too complex

**Expected outcome (given EDA):**
- Likely K_eff ≈ 1-2
- If so, confirms simple structure
- If K_eff > 3, questions EDA conclusions

---

## Implementation Priority

### Phase 1: Model 1 (Recommended starting point)

**Rationale:**
- Most conservative extension
- Well-studied, computationally stable
- Nests normal model (if nu large)

**Timeline:** Week 1

**Deliverables:**
- Stan code (✓ already created)
- Convergence diagnostics
- Posterior summaries
- LOO-CV comparison to normal
- Decision: Continue to Model 2 or stop here?

---

### Phase 2: Model 2 (Conditional)

**Trigger:** If Model 1 shows heavy tails (nu < 20) OR clear evidence of subgroups

**Timeline:** Week 2

**Deliverables:**
- PyMC code (✓ already created)
- Cluster assignment analysis
- Comparison to Model 1
- Decision: Is mixture supported by data?

---

### Phase 3: Model 3 (Advanced/Optional)

**Trigger:** If Model 2 suggests >2 clusters OR fundamental uncertainty

**Timeline:** Week 3

**Deliverables:**
- PyMC code (✓ already created)
- Effective cluster count analysis
- Comprehensive model comparison
- Decision: Final model choice

---

## Falsification Strategy

### Global Stopping Rules

**Abort entire approach if:**

1. All models show same pathology (suggests data issue)
2. Computational issues across all implementations
3. Posterior predictive checks fail for all models
4. Extreme sensitivity (removing any study changes estimate >50%)
5. Models give qualitatively different conclusions

### Model-Specific Red Flags

**Model 1:**
- Divergent transitions >5% after tuning
- R-hat > 1.05 for any parameter
- nu estimates <3 or >100

**Model 2:**
- Label switching despite constraints
- pi posterior uniform on [0,1]
- Cluster assignments nonsensical

**Model 3:**
- Each study in separate cluster
- K_eff unstable across chains
- Extreme concentration parameter

---

## Comparison Metrics

### Predictive Performance

1. **Leave-One-Out Cross-Validation (LOO-CV)**
   - Primary metric for model comparison
   - Prefer model with highest ELPD
   - If ΔELPD < 2 SE: Models equivalent, choose simpler

2. **Posterior Predictive Checks**
   - Test statistics: mean, SD, min, max, range
   - Good calibration: p-values in [0.05, 0.95]
   - Check for systematic biases

3. **Prediction Interval Coverage**
   - Generate future study predictions
   - Check if observed studies within 95% PI
   - Expect ≈95% coverage if well-calibrated

### Computational Diagnostics

1. **Convergence:** R-hat < 1.01, ESS > 400
2. **Efficiency:** Divergences <1%, high effective sample size
3. **Stability:** Consistent results across chains/initializations

### Scientific Interpretability

1. **Parameters make sense:** No extreme values
2. **Prior-posterior coherence:** Posteriors update reasonably
3. **Robustness:** Conclusions stable to small changes

---

## Sensitivity Analyses

### Required for Each Model

1. **Prior sensitivity**
   - Vary all hyperpriors by factor of 2-3
   - Check if posteriors change >10%
   - Report range of estimates

2. **Influence analysis**
   - Remove Study 4 (most influential)
   - Remove Study 5 (only negative)
   - Remove both
   - Check if model class changes

3. **Computational validation**
   - Run 4 chains from different seeds
   - Check convergence metrics
   - Verify reproducibility

---

## Expected Results (Predictions)

Based on EDA showing I²=2.9% and no outliers:

### Model 1 (t-distribution)
- **Prediction:** nu ≈ 30-50 (near-normal)
- **Implication:** Normal model adequate
- **Action if wrong:** If nu < 20, heavy tails are real concern

### Model 2 (Mixture)
- **Prediction:** pi → 0 or 1 (single cluster)
- **Implication:** Homogeneity confirmed
- **Action if wrong:** If genuine mixture, investigate covariates

### Model 3 (Dirichlet Process)
- **Prediction:** K_eff ≈ 1-2
- **Implication:** Simple structure
- **Action if wrong:** If K_eff > 3, reconsider pooling

### Overall
- **Most likely:** All models converge to similar conclusions
- **Best outcome:** Model 1 sufficient, validates standard approach
- **Surprising outcome:** Any model shows complex heterogeneity

---

## Success Criteria

### Scientific Success
- Find model that genuinely explains data
- Understand why alternatives fail
- Quantify uncertainty honestly
- Make reliable predictions

### Statistical Success
- Posterior predictive p-values: [0.05, 0.95]
- LOO pareto-k < 0.7 for all studies
- R-hat < 1.01, ESS > 400
- Convergence across chains

### Practical Success
- Clear recommendation with justification
- Transparent about limitations
- Actionable for decision-makers
- Reproducible analysis

---

## Comparison to Designer 1

**Designer 1 (expected):** Classical models
- Hierarchical normal
- Common effect
- Fixed/random effects comparison

**Designer 2 (this):** Robustness focus
- Heavy-tailed distributions
- Mixture models
- Non-parametric approaches

**Complementarity:**
- D1 establishes baseline
- D2 tests robustness
- Together: Comprehensive assessment

**Expected agreement:**
- If data truly homogeneous: All models agree
- If complex heterogeneity: D2 models reveal structure
- If outliers present: D2 models more robust

---

## How to Run

### Prerequisites

```bash
# Install required packages
pip install pymc arviz cmdstanpy pandas numpy scipy matplotlib seaborn

# Compile Stan model
cmdstan_model = cmdstanpy.CmdStanModel(stan_file='model_1_t_distribution.stan')
```

### Step 1: Fit Model 1 (t-distribution)

```python
# Using cmdstanpy
import cmdstanpy as cs

model = cs.CmdStanModel(stan_file='model_1_t_distribution.stan')
data = {
    'J': 8,
    'y': [20.02, 15.30, ...],  # from data.csv
    'sigma': [15, 10, ...]
}
fit = model.sample(data=data, chains=4, iter_sampling=2000)
```

### Step 2: Diagnostics and Comparison

```python
# Check convergence
fit.diagnose()

# Compute LOO
import arviz as az
idata = az.from_cmdstan(fit)
loo = az.loo(idata)

# Posterior predictive checks
# (implement based on model output)
```

### Step 3: Conditional on Results

- If nu > 30: Stop here, normal model adequate
- If nu < 20: Proceed to Model 2
- If evidence of mixture: Proceed to Model 3

---

## Documentation

### For Each Model Run

Document:
1. Convergence diagnostics (R-hat, ESS, traces)
2. Posterior summaries (mean, SD, quantiles)
3. LOO-CV results
4. Posterior predictive checks
5. Sensitivity analyses
6. Decision and rationale

### Final Report

Include:
1. Model comparison table
2. Recommendation with justification
3. Limitations and caveats
4. Comparison to Designer 1
5. Future directions

---

## Contact and Collaboration

**Designer 2 Focus:** Robustness and distributional assumptions

**Parallel Work:**
- Designer 1: Classical models
- Designer 2: Robust alternatives
- Synthesis: Compare and recommend

**Independent Analysis:**
- No coordination during design/implementation
- Compare results afterward
- Discuss disagreements
- Converge on recommendation

---

## References

### Methodological

- Gelman et al. (2013). *Bayesian Data Analysis* (3rd ed.)
- Kruschke (2014). *Doing Bayesian Data Analysis*
- McElreath (2020). *Statistical Rethinking* (2nd ed.)

### Meta-Analysis Specific

- Higgins & Thompson (2002). Quantifying heterogeneity in meta-analysis
- Riley et al. (2011). Evaluation of statistical methods for meta-analysis
- Röver et al. (2020). Bayesian random-effects meta-analysis using the bayesmeta R package

### Robust Methods

- Chung et al. (2013). A nondegenerate penalized likelihood estimator for variance parameters in multilevel models
- Gelman (2006). Prior distributions for variance parameters in hierarchical models

### Dirichlet Processes

- Ferguson (1973). A Bayesian analysis of some nonparametric problems
- Escobar & West (1995). Bayesian density estimation and inference using mixtures

---

**Last Updated:** 2025-10-28
**Status:** Ready for implementation
**Next Step:** Fit Model 1 and evaluate results
