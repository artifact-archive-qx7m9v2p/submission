# Experiment 2: Hierarchical Logit Model (Non-Centered)

## Model Specification

**Model Class**: Hierarchical binomial with logit-normal random effects
**Implementation**: Stan (CmdStanPy) with non-centered parameterization
**Date**: 2025-10-30

### Likelihood

```
r_i ~ Binomial(n_i, θ_i)
logit(θ_i) = μ_logit + σ·η_i
η_i ~ Normal(0, 1)
```

**Non-centered parameterization**: More efficient for HMC sampling than centered version.

### Interpretation

- **μ_logit**: Population-level log-odds of success
- **σ**: Standard deviation of trial-specific effects on logit scale
- **η_i**: Standardized trial-specific deviations (N=12)
- **θ_i**: Trial-specific success probabilities (transformed)

**Overdispersion mechanism**: Gaussian variation on log-odds scale creates probability heterogeneity.

### Priors

```
μ_logit ~ Normal(logit(0.074), 1)   # Centers on pooled proportion
σ ~ Normal(0, 1) truncated [0, ∞)   # Half-normal for scale
η_i ~ Normal(0, 1)                  # Standard normal (non-centered)
```

**Prior Justification**:

- **μ_logit ~ Normal(-2.53, 1)**:
  - logit(0.074) ≈ -2.53
  - SD=1 allows μ to range roughly from 0.02 to 0.25 on probability scale
  - Regularizes but allows data to dominate

- **σ ~ HalfNormal(0, 1)**:
  - E[σ] ≈ 0.8, SD ≈ 0.6
  - σ=1 implies moderate heterogeneity on logit scale
  - Prior mass concentrated in [0, 2], allowing σ up to 3-4 if needed

- **η_i ~ Normal(0, 1)**:
  - Standard normal for non-centered parameterization
  - Separates hierarchy from trial-specific effects
  - Improves HMC geometry (reduces funnel)

### Transformation

```
θ_i = logistic(μ_logit + σ·η_i) = 1/(1 + exp(-(μ_logit + σ·η_i)))
```

### Expected Behavior

**Scale Interpretation**:
- Small σ (< 0.5): Minimal heterogeneity, trials similar
- Medium σ (0.5-2): Moderate variation, expected for this data
- Large σ (> 2): High heterogeneity across trials

**Advantages over Beta-Binomial**:
1. Logit scale is natural for multiplicative effects
2. Non-centered parameterization aids HMC convergence
3. Variation is scale-dependent (less extreme at boundaries)
4. May be more identifiable with N=12 trials

**Potential Issues**:
- Trial 1 (r=0/47): May push θ_1 to extreme low values
- Trial 8 (r=31/215): May push θ_8 to higher values
- Wider posteriors than Beta-Binomial (less informative priors)

### Falsification Criteria

Will REJECT this model if:
1. Divergent transitions > 1% of samples (parameterization issue)
2. Posterior for η_i strongly non-normal (Shapiro p < 0.01)
3. σ posterior has mode > 3 (implausibly high variation)
4. Trial 1 (0/47) causes extreme θ_1 → 0 (model cannot handle)
5. Cannot recover known parameters in simulation-based validation
6. Posterior predictive checks fail systematically

### Success Criteria

Will ACCEPT this model if:
1. Convergence: Rhat < 1.01, ESS > 400
2. No or minimal divergences (< 1%)
3. Posterior predictive checks capture key features
4. LOO diagnostics acceptable (Pareto k < 0.7 for most trials)
5. Parameters scientifically interpretable
6. Comparable or better LOO than other viable models

### Computational Approach

- **Software**: CmdStanPy with Stan
- **Sampling**: 4 chains, 2000 iterations (1000 warmup, 1000 sampling)
- **Adapt delta**: 0.95 (increased from default 0.8 to reduce divergences)
- **Max treedepth**: 12 (allow deeper trees if needed)
- **Diagnostics**: ArviZ for convergence and model checking
- **Log-likelihood**: Save pointwise log_lik for LOO-CV

### Non-Centered vs Centered

**Why Non-Centered?**

Centered: logit(θ_i) ~ Normal(μ_logit, σ)
- Creates funnel geometry when σ is uncertain
- HMC struggles when σ near 0 (posterior geometry changes)

Non-centered: η_i ~ Normal(0, 1), logit(θ_i) = μ_logit + σ·η_i
- Separates hierarchy parameter (σ) from trial effects (η_i)
- More uniform posterior geometry
- Better HMC performance, especially with small N

### Comparison to Beta-Binomial

**Structural Differences**:
- Beta-Binomial: Continuous mixing on probability scale
- Hierarchical Logit: Gaussian mixing on log-odds scale

**Expected Outcomes**:
- If σ < 1.5: Both models should give similar fits
- If σ > 2: Models may diverge (different scale assumptions)
- LOO comparison will reveal which scale is more appropriate

### Implementation Status

- [ ] Prior predictive check
- [ ] Simulation-based validation
- [ ] Model fitting
- [ ] Posterior predictive check
- [ ] Model critique
- [ ] Decision: ACCEPT/REVISE/REJECT
