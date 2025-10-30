# Experiment 1: Standard Hierarchical Model (Partial Pooling)

**Date**: 2025-10-29
**Model Class**: Hierarchical Normal with Random Effects
**Status**: Validation in progress

---

## Model Specification

### Mathematical Form

**Likelihood**:
```
y_i ~ Normal(theta_i, sigma_i)   for i = 1, ..., 8
```
where:
- y_i = observed treatment effect for school i
- sigma_i = known standard error (given in data)
- theta_i = true underlying effect for school i

**School-level model** (partial pooling):
```
theta_i ~ Normal(mu, tau)
```
where:
- mu = population mean effect
- tau = between-school standard deviation

**Hyperpriors**:
```
mu ~ Normal(0, 50)           # Weakly informative on population mean
tau ~ HalfCauchy(0, 25)      # Gelman's recommendation for group-level SD
```

### Stan Implementation

Uses **non-centered parameterization** to avoid funnel geometry when tau is small:
```stan
parameters {
  real mu;
  real<lower=0> tau;
  vector[J] theta_raw;  // ~ Normal(0, 1)
}
transformed parameters {
  vector[J] theta = mu + tau * theta_raw;
}
```

---

## Theoretical Rationale

**Why this model?**
1. **Canonical approach**: Standard reference for hierarchical data with known measurement error
2. **Data-adaptive pooling**: Posterior of tau determines shrinkage strength
   - If tau → 0: converges to complete pooling (all schools identical)
   - If tau large: minimal pooling (schools independent)
3. **Exchangeability**: Assumes schools are random samples from common superpopulation
4. **Known measurement error**: Properly accounts for different precision across schools

**EDA support**:
- Very low heterogeneity (I² = 1.6%) suggests tau will be small
- Variance paradox (observed < expected) consistent with minimal between-school variation
- Normality assumption validated (all tests p > 0.67)
- High individual uncertainty justifies information sharing via pooling

---

## Prior Justification

### mu ~ Normal(0, 50)

**Rationale**:
- Centered at 0 (no prior bias toward positive/negative treatment effects)
- SD = 50 allows effects from -100 to +100 at 95% prior credibility
- Observed effects range from -5 to 26, so prior is weakly informative
- With n=8 schools, likelihood will dominate prior
- More conservative than half-Cauchy alternatives

**Domain context**:
- Educational interventions typically have effect sizes |Cohen's d| < 1
- On many scales, this translates to effects of ±10-20 points
- Prior encompasses this range while remaining vague

### tau ~ HalfCauchy(0, 25)

**Rationale**:
- Gelman's (2006) recommendation for hierarchical standard deviations
- Scale = 25 based on typical effect size range
- Heavy tails allow large tau if data demand it
- Median ≈ 18, but probability mass extends to small values
- Performs well with small number of groups (n=8)

**Why not alternatives?**:
- HalfNormal(0, 25): Lighter tails, less conservative
- Exponential(1/25): Too strong regularization toward 0
- Inverse-Gamma(ε, ε): Improper, sensitive to ε choice
- Uniform(0, 100): Improper unless truncated, no regularization

**EDA-informed alternative** (Experiment 2):
- HalfNormal(0, 5) if we strongly believe tau < 10 based on I² = 1.6%

---

## Parameters to Estimate

**Total**: 10 parameters

1. **mu** (scalar): Population mean effect
2. **tau** (scalar): Between-school SD
3. **theta[1:8]** (vector): School-specific true effects

**Derived quantities** (generated quantities block):
- Shrinkage factors: 1 - tau² / (tau² + sigma_i²)
- Posterior predictive: y_rep ~ Normal(theta_i, sigma_i)
- New school prediction: theta_new ~ Normal(mu, tau)

---

## Falsification Criteria

**I will REJECT this model if**:

1. **Computational failure**:
   - R-hat > 1.01 for any parameter (non-convergence)
   - ESS < 400 for any parameter (poor mixing)
   - Divergent transitions > 0 (geometry problems)

2. **Prior-posterior conflict**:
   - Posterior tau > 15 despite prior median ≈ 18 (fighting data)
   - Extreme posterior values: mu outside [-50, 50]

3. **Poor predictive performance**:
   - Posterior predictive checks systematically fail
   - LOO Pareto-k > 0.7 for multiple observations
   - Cannot replicate key features of observed data

4. **Extreme prior sensitivity**:
   - Results flip dramatically with minor prior changes
   - Suggests data too weak to overcome prior

**I will REVISE this model if**:
- Specific schools persistently mispredicted → Try Horseshoe (Exp 3)
- Bimodal residuals → Try Mixture (Exp 4)
- Systematic PPC failures → Try Measurement Error (Exp 5)

**I will ACCEPT this model if**:
- All computational diagnostics pass
- Posterior predictive checks show good fit
- LOO-CV competitive with alternatives
- Substantive interpretation makes sense

---

## Expected Outcomes

Based on EDA (I² = 1.6%, variance ratio = 0.75):

**Posterior expectations**:
- **mu**: ≈ 10-12 (near observed mean of 12.5)
- **tau**: ≈ 3-8 (small due to low heterogeneity)
- **theta_i**: Strongly shrunk toward mu, especially School 5

**Shrinkage pattern**:
- Schools with high sigma (e.g., School 8, sigma=18): More shrinkage
- Schools with low sigma (e.g., School 5, sigma=9): Less shrinkage
- Extreme effects (Schools 3, 4, 5) shrink toward mu

**Uncertainty**:
- Wide posterior credible intervals for theta_i (small n=8)
- tau posterior may be diffuse (hard to estimate with 8 groups)
- Potential funnel geometry if tau very small

**Model comparison**:
- Should outperform complete pooling (fails to capture any heterogeneity)
- Should outperform no pooling (ignores information sharing)
- Likely comparable to Experiment 2 (near-complete pooling)

---

## Validation Plan

### 1. Prior Predictive Check
- Generate datasets from prior: y_sim ~ prior
- Check if plausible ranges include observed data
- Verify prior doesn't exclude reasonable values

**Success**: Prior generates diverse datasets spanning observed range

### 2. Simulation-Based Calibration (SBC)
- Simulate data from model with known parameters
- Fit model and check if posteriors contain true values
- Assess calibration and computational reliability

**Success**: Uniform rank statistics, no simulation failures

### 3. Fit to Real Data
- Run MCMC (4 chains, 2000 iterations, 1000 warmup)
- Monitor R-hat, ESS, divergences, energy diagnostics
- Extract posterior samples

**Success**: R-hat < 1.01, ESS > 400, zero divergences

### 4. Posterior Predictive Check
- Generate replicated data: y_rep ~ posterior
- Compare to observed data via test statistics
- Visual checks: histograms, Q-Q plots, forest plots

**Success**: y_obs within bulk of y_rep distribution, p-values ∈ [0.05, 0.95]

### 5. Model Critique
- Assess adequacy for scientific question
- Identify weaknesses and potential improvements
- Decide: ACCEPT / REVISE / REJECT

**Success**: Model adequate for inference, or clear path to improvement

---

## Implementation Notes

**Software**: CmdStanPy (primary) or PyMC (fallback)

**Computational settings**:
- Chains: 4 (for convergence assessment)
- Iterations: 2000 per chain (1000 warmup, 1000 sampling)
- Adapt delta: 0.95 (reduce divergences)
- Max tree depth: 12 (allow deeper trajectories if needed)

**File outputs**:
- Stan model: `experiments/experiment_1/*/code/model.stan`
- Posterior samples: `experiments/experiment_1/posterior_inference/diagnostics/posterior_inference.netcdf`
- Plots: `experiments/experiment_1/*/plots/`
- Reports: `experiments/experiment_1/*/findings.md`

---

## References

- Gelman, A., & Hill, J. (2006). *Data Analysis Using Regression and Multilevel/Hierarchical Models*. Section 5.6.
- Rubin, D. B. (1981). Estimation in parallel randomized experiments. *Journal of Educational Statistics*, 6(4), 377-401.
- Gelman, A. (2006). Prior distributions for variance parameters in hierarchical models. *Bayesian Analysis*, 1(3), 515-534.

---

**Next step**: Prior Predictive Check
