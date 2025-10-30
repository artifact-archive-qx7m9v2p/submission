# Bayesian Model Experiment Plan
## Synthesis of Designer Proposals

**Date**: 2025-10-30
**Problem**: Binomial overdispersion (φ = 3.51, p < 0.001)
**Data**: 12 trials with varying sample sizes, 208 total successes out of 2,814 trials

---

## Designer Proposals Summary

### Designer 1 (Variance Modeling Focus)
1. Beta-Binomial (conjugate, continuous variation)
2. Dirichlet Process Mixture (non-parametric, unknown K groups)
3. Logistic-Normal Hierarchical (Gaussian on logit scale)

### Designer 2 (Hierarchical & Group-Based Focus)
1. Hierarchical Beta-Binomial (partial pooling)
2. Finite Mixture Model (K=2 or 3 discrete groups)
3. Non-Centered Hierarchical Logit (efficient parameterization)

### Designer 3 (Alternative Approaches)
1. Finite Mixture of Binomials (discrete regimes)
2. Robust Contamination Model (outlier mechanism)
3. Structured Outlier Detection (data-driven outlier probability)

---

## Synthesized Model Classes (Duplicates Removed)

After removing duplicates and grouping similar approaches, we have **5 distinct model classes**:

### **Model Class 1: Beta-Binomial (Standard)**
- **Source**: Designer 1 (Model 1), Designer 2 (Model 1)
- **Type**: Continuous probability variation, conjugate
- **Hypothesis**: Smooth continuous heterogeneity in success probabilities
- **Key Feature**: Marginalized likelihood for numerical stability
- **Overdispersion Mechanism**: Low concentration parameter (φ or κ)
- **Priority**: **HIGH** - Standard approach, well-understood, numerically stable

### **Model Class 2: Hierarchical Logit (Non-Centered)**
- **Source**: Designer 1 (Model 3), Designer 2 (Model 3)
- **Type**: Gaussian variation on log-odds scale
- **Hypothesis**: Multiplicative effects with scale-dependent variation
- **Key Feature**: Non-centered parameterization for HMC efficiency
- **Overdispersion Mechanism**: Gaussian variance in logit(θ)
- **Priority**: **HIGH** - Complements Beta-Binomial, different scale assumptions

### **Model Class 3: Finite Mixture Model (K=2 or 3)**
- **Source**: Designer 2 (Model 2), Designer 3 (Model 1)
- **Type**: Discrete probability regimes
- **Hypothesis**: Trials cluster into 2-3 distinct probability groups
- **Key Feature**: Ordered constraints for identifiability
- **Overdispersion Mechanism**: Between-group heterogeneity
- **Priority**: **MEDIUM** - Matches EDA evidence but risk of overfitting with N=12

### **Model Class 4: Robust Contamination Model**
- **Source**: Designer 3 (Model 2)
- **Type**: Beta-Binomial + explicit outlier mechanism
- **Hypothesis**: Clean baseline process with contamination/outliers
- **Key Feature**: Mixture of Beta-Binomial and uniform outlier component
- **Overdispersion Mechanism**: Combination of smooth variation + outliers
- **Priority**: **LOW** - More complex, may be overparameterized for N=12

### **Model Class 5: Dirichlet Process Mixture**
- **Source**: Designer 1 (Model 2)
- **Type**: Non-parametric clustering, unknown K
- **Hypothesis**: Unknown number of discrete probability groups
- **Key Feature**: Data learns number of clusters
- **Overdispersion Mechanism**: Non-parametric mixture
- **Priority**: **LOW** - High complexity, label switching issues, may overfit

---

## Prioritized Experiment Sequence

Based on theoretical justification, computational feasibility, and parsimony:

### **Experiment 1: Beta-Binomial Model** ⭐ PRIMARY
**Rationale**:
- Standard approach for binomial overdispersion
- Conjugate structure → numerically stable
- Well-understood theory and interpretation
- Both designers prioritized this approach

**Specification**:
```
Likelihood: r_i ~ BetaBinomial(n_i, α, β)
Priors:
  μ ~ Beta(2, 25)        # E[μ] ≈ 0.074
  φ ~ Gamma(2, 2)        # Concentration parameter
  α = μ·φ, β = (1-μ)·φ
```

**Falsification Criteria**:
- Posterior predictive p-value outside [0.05, 0.95]
- Concentration φ < 0.5 or > 50 (extreme values)
- Funnel plot violations persist in posterior samples
- LOO Pareto k > 0.7 for multiple observations

**Success Criteria**:
- Rhat < 1.01 for all parameters
- ESS > 400 for all parameters
- Posterior predictive checks pass (p ∈ [0.05, 0.95])
- LOO diagnostics acceptable (all Pareto k < 0.7)

---

### **Experiment 2: Hierarchical Logit Model (Non-Centered)** ⭐ PRIMARY
**Rationale**:
- Tests different scale assumption (log-odds vs probability)
- Non-centered parameterization → efficient HMC sampling
- Natural for multiplicative effects
- Complements Beta-Binomial (different structural assumptions)

**Specification**:
```
Likelihood: r_i ~ Binomial(n_i, θ_i)
           logit(θ_i) = μ_logit + σ·η_i
           η_i ~ Normal(0, 1)
Priors:
  μ_logit ~ Normal(logit(0.074), 1)
  σ ~ Normal(0, 1) truncated to [0, ∞)
```

**Falsification Criteria**:
- Posterior η_i strongly non-normal (Shapiro p < 0.01)
- σ > 3 (implausible variation on logit scale)
- Divergent transitions > 1% of samples
- Poor fit for trial 1 (r=0/47) - check with posterior predictive

**Success Criteria**:
- Rhat < 1.01, ESS > 400
- No divergences or < 1% divergences
- Posterior predictive checks pass
- Comparable or better LOO than Beta-Binomial

---

### **Experiment 3: Finite Mixture Model (K=3)** ⭐ SECONDARY
**Rationale**:
- EDA shows evidence for 2-3 groups (tercile split p=0.012)
- Tests discrete vs continuous heterogeneity hypothesis
- Both Designers 2 and 3 proposed this
- Risk of overfitting but worth testing

**Specification**:
```
Likelihood: z_i ~ Categorical(π)
           r_i | z_i ~ Binomial(n_i, p_{z_i})
Priors:
  π ~ Dirichlet(2, 2, 2)
  p_1, p_2, p_3 ~ Beta(...) with ordering: p_1 < p_2 < p_3
```

**Falsification Criteria**:
- Component separation < 0.6 (poorly separated groups)
- Any π_k < 0.05 (component with < 5% of data)
- Posterior overlap > 80% between adjacent components
- K=1 in posterior (no clustering) or K>5 (overfitting)

**Success Criteria**:
- Clear component separation (> 0.6)
- All components have reasonable membership (π_k > 0.1)
- Better LOO than Beta-Binomial (ΔLOO > 2)
- Stable classification (posterior classification probability > 0.7)

---

### **Experiment 4: Robust Contamination Model** (IF NEEDED)
**Rationale**:
- Only pursue if Experiments 1-3 show persistent outlier issues
- Trials 1 (0/47) and 8 (31/215) flagged as potential outliers
- More complex, may be overparameterized

**Specification**:
```
Likelihood: w_i ~ Bernoulli(ω)  # outlier indicator
           If w_i = 0: r_i ~ BetaBinomial(n_i, α, β)
           If w_i = 1: r_i ~ Binomial(n_i, 0.5)  # uninformative outlier
Priors:
  ω ~ Beta(1, 11)  # E[ω] ≈ 0.08 (expect ~1 outlier)
  α, β as in Experiment 1
```

**Falsification Criteria**:
- ω > 0.5 (implausibly high outlier rate)
- Same outliers flagged as in simple residual analysis
- No improvement in LOO over Beta-Binomial

---

### **Experiment 5: Dirichlet Process Mixture** (LOW PRIORITY)
**Rationale**:
- Only if finite mixtures (Exp 3) succeed but K is uncertain
- Computational challenges (label switching)
- May overfit with N=12

**Will Skip Unless**: Experiment 3 shows strong mixture evidence but K=2 vs K=3 is ambiguous

---

## Minimum Attempt Policy

Per workflow guidelines:
- **Must attempt**: Experiments 1 and 2 (unless Exp 1 fails pre-fit validation)
- **Conditional**: Experiment 3 if Experiments 1-2 show inadequacies
- **Optional**: Experiments 4-5 based on evidence from earlier experiments

---

## Model Comparison Strategy

After fitting viable models:

1. **LOO-CV Comparison**: Use `az.compare()` to rank models by ELPD
2. **Parsimony Rule**: If |ΔELPD| < 2×SE, prefer simpler model
3. **Posterior Predictive Checks**: All models must capture key data features
4. **Substantive Interpretation**: Prefer model with clearer scientific interpretation

**Decision Tree**:
```
If ΔLOO(BetaBin, Logit) < 2*SE:
  → Choose based on interpretability (likely Beta-Binomial)
If Mixture >> Beta-Binomial (ΔLOO > 4):
  → Evidence for discrete groups, report Mixture
If all models similar (ΔLOO < 2*SE):
  → Report uncertainty, consider model averaging
```

---

## Universal Falsification Criteria

**Abandon ALL models and reconsider if**:
1. None achieve acceptable convergence (Rhat > 1.01)
2. All show LOO Pareto k > 0.7 for majority of observations
3. All fail posterior predictive checks (systematic misfit)
4. Posterior distributions are degenerate or multimodal without scientific interpretation
5. Cannot recover known parameters in simulation-based validation

**Pivot Plan**: If all models fail → consider simpler pooled model or collect more data

---

## Implementation Notes

- **PPL**: Use Stan (CmdStanPy) as primary implementation
- **Fallback**: PyMC if Stan shows numerical issues (with documented errors)
- **Log-Likelihood**: All models must save pointwise log_lik for LOO-CV
- **Diagnostics**: ArviZ for all convergence and model comparison diagnostics
- **Reproducibility**: Set random seeds, save all code and outputs

---

## Next Steps

1. ✅ Complete: EDA and model design
2. **Next**: Experiment 1 - Beta-Binomial Model
   - Prior predictive checks
   - Simulation-based validation
   - Model fitting
   - Posterior predictive checks
   - Model critique
3. **Then**: Experiment 2 - Hierarchical Logit Model
4. **Evaluate**: Need for Experiment 3+ based on results

---

## Success Metrics

**Adequate Solution Achieved If**:
- At least one model passes all validation stages
- LOO diagnostics acceptable (Pareto k < 0.7)
- Posterior predictive checks show good calibration
- Scientific interpretation is clear and defensible
- Uncertainty quantification is honest and appropriate

**Continue Iteration If**:
- Models show systematic deficiencies but have clear improvement path
- Evidence suggests specific model refinements
- Computational issues can be resolved

**Stop and Report Limitations If**:
- Diminishing returns (improvements < 2*SE)
- Data insufficient for reliable inference (very wide posteriors)
- Fundamental issues cannot be resolved with available data
