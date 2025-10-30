# Experiment 1: Logarithmic Regression

**Created**: 2025-01-XX
**Status**: In Progress - Prior Predictive Check
**Priority**: HIGHEST (Required minimum attempt)

---

## Model Specification

### Scientific Hypothesis
The relationship between Y and x follows an unbounded logarithmic growth pattern, consistent with Weber-Fechner law or diminishing returns phenomena. This implies Y continues to increase with x, but at a decreasing rate.

### Mathematical Form

**Likelihood**:
```
Y_i ~ Normal(μ_i, σ)
μ_i = α + β·log(x_i)
```

**Parameters**:
- α: Intercept (Y value when x=1, since log(1)=0)
- β: Logarithmic slope (change in Y per unit increase in log(x))
- σ: Residual standard deviation (observation noise)

### Prior Specifications

Based on EDA findings (Y ∈ [1.71, 2.63], x ∈ [1, 31.5], exploratory fit: α≈1.75, β≈0.27):

```
α ~ Normal(1.75, 0.5)
  # Center at EDA estimate
  # SD=0.5 allows α ∈ [0.75, 2.75] with 95% probability
  # Weakly informative: constrains to reasonable Y-scale

β ~ Normal(0.27, 0.15)
  # Center at EDA estimate
  # SD=0.15 allows β ∈ [-0.03, 0.57] with 95% probability
  # Positive mode but allows negative (data can override)
  # Weakly informative: prevents extreme slopes

σ ~ HalfNormal(0.2)
  # Mode at 0, scale=0.2
  # Allows σ ∈ [0, 0.4] with ~95% probability
  # Consistent with EDA residual SD ≈ 0.12
  # Weakly informative: prevents overfitting
```

### Prior Justification

1. **α prior**: Centered at Y(x=1) from EDA fit. SD=0.5 is ~50% of observed Y range, allowing substantial deviation while preventing unreasonable values (e.g., negative Y or Y>4).

2. **β prior**: Centered at EDA slope. SD=0.15 is ~50% of point estimate, allowing data to substantially revise. Mode is positive (consistent with increasing trend) but includes 0 in tails (data could show flat relationship).

3. **σ prior**: HalfNormal(0.2) is weakly informative for residual variation. Mode at 0 encourages parsimony, but scale=0.2 easily accommodates observed residual SD≈0.12. Would need strong data evidence for σ>0.5.

All priors are **weakly informative**: they guide the model toward reasonable parameter space but can be easily overridden by data.

---

## Falsification Criteria

This model will be REJECTED if:

1. **Systematic residual pattern**:
   - PPC test statistic (max Y residual) shows p-value < 0.05
   - Visual: Residuals vs x plot shows clear curvature
   - Interpretation: Logarithmic form is misspecified

2. **Inferior predictive performance**:
   - LOO-ELPD worse than best competing model by >4
   - Interpretation: A different model class fits data better

3. **Influential point dominance**:
   - Pareto k > 0.7 for >5 observations
   - Removing x=31.5 changes β by >30%
   - Interpretation: Model is overly sensitive to few points

4. **Poor calibration**:
   - 95% posterior predictive intervals contain <85% or >99% of observations
   - Interpretation: Uncertainty quantification is miscalibrated

5. **Prior-posterior conflict**:
   - Prior predictive and posterior predictive distributions diverge dramatically (KS test p<0.01)
   - Interpretation: Priors fighting data; need different model or priors

---

## Expected Outcomes

### Most Likely Scenario (Baseline Expectation)

**Model Performance**:
- LOO-RMSE: 0.11 - 0.13 (similar to EDA fit)
- R²: 0.80 - 0.85
- Pareto k: <0.5 for most observations, possibly 0.5-0.7 for x=31.5

**Parameter Posteriors**:
- α: 1.6 - 1.9 (narrower than prior)
- β: 0.20 - 0.34 (narrower than prior, clearly positive)
- σ: 0.10 - 0.14 (consistent with EDA)

**Posterior Predictive Checks**:
- Pass most checks
- Possible slight underprediction at highest x values
- Gap region x∈[23,29] will have wide credible intervals

**Decision**: ACCEPT as baseline model, compare with alternatives

---

### Alternative Scenario: Model Inadequate

**Symptoms**:
- Systematic residuals at high x (curvature)
- Poor PPC (p-value < 0.05 for several test statistics)
- High influential point sensitivity

**Action**:
- If residuals show plateau: Try Michaelis-Menten (Experiment 4)
- If residuals show acceleration: Try quadratic (deferred model)
- If outliers dominate: Try robust-t (Experiment 3)

**Decision**: REVISE or REJECT, escalate to alternative model

---

### Surprising Scenario: Model Perfect

**Symptoms**:
- LOO-RMSE < 0.10
- All PPC tests pass with flying colors
- Low influential point sensitivity
- 95% intervals contain exactly 95% of data

**Action**:
- Still fit Experiment 2 (hierarchical) to test replicate structure
- May skip Experiments 3-5 if this is clearly adequate

**Decision**: ACCEPT, declare success early if no alternative hypotheses remain

---

## Computational Considerations

### Implementation: Stan via CmdStanPy

**Reasons for Stan**:
- Simple linear model with log transform: Stan's strength
- No numerical integration needed
- Fast sampling expected (< 1 min for 4 chains × 2000 iterations)
- Stable HMC for this problem

**Potential Issues**:
- None expected for this simple model
- If divergences occur: Likely data/prior issue, not computational

**Fallback**: PyMC if Stan mysteriously fails (unlikely)

### Sampling Strategy

- **Chains**: 4 parallel chains
- **Iterations**: 2000 (1000 warmup + 1000 sampling)
- **Total samples**: 4000 post-warmup
- **Thinning**: None (not needed for simple model)
- **Adapt delta**: 0.95 (default, increase to 0.99 if divergences)

### Convergence Criteria

- Rhat < 1.01 for all parameters
- ESS_bulk > 400 (10% of samples)
- ESS_tail > 400
- Divergences < 1% (<40 out of 4000 transitions)

---

## Success Criteria

### Pre-fit Validation (Prior Predictive)
- [ ] Prior predictive samples cover plausible data range
- [ ] No extreme or impossible predictions
- [ ] Prior on β produces monotonic increasing functions (mostly)

### Pre-fit Validation (Simulation-Based)
- [ ] Can recover true parameters from synthetic data
- [ ] 95% credible intervals contain true values ≥90% of simulations
- [ ] No biased parameter estimates

### Fitting
- [ ] All chains converge (Rhat < 1.01)
- [ ] Adequate ESS (>400)
- [ ] No divergences or <1% rate
- [ ] Reasonable runtime (<5 min)
- [ ] InferenceData saved with log_likelihood

### Post-fit Validation (Posterior Predictive)
- [ ] 95% intervals contain ~95% of observations
- [ ] No systematic residual patterns
- [ ] Test statistics (mean, SD, max, min) well-calibrated

### Model Critique
- [ ] LOO diagnostics: Pareto k < 0.7 for >90% observations
- [ ] Sensitivity: Results stable when excluding x=31.5
- [ ] Interpretability: Parameters scientifically sensible

---

## Connection to EDA

**EDA Found** (exploratory fit):
- Logarithmic R² = 0.829, RMSE = 0.115
- Coefficients: α ≈ 1.75, β ≈ 0.27
- Logarithmic outperformed linear, sqrt, asymptotic
- Slight concern about saturation at high x

**This Experiment Tests**:
- Is logarithmic form justified under Bayesian framework with proper uncertainty?
- Are priors reasonable and not overly influential?
- Can model handle influential points and gaps?
- Does it satisfy rigorous posterior predictive checks?

**Key Difference from EDA**:
- EDA used frequentist least squares (point estimates)
- Bayesian approach provides full uncertainty quantification
- Rigorous validation pipeline (prior/posterior checks, LOO, calibration)
- Explicit falsification criteria

---

## Timeline

| Stage | Duration | Output |
|-------|----------|--------|
| Prior predictive check | 30 min | Findings + plots |
| Simulation-based validation | 30 min | Recovery metrics |
| Model fitting | 5-10 min | Posterior samples |
| Posterior predictive check | 30 min | PPC plots + findings |
| Model critique | 30 min | Decision document |
| **Total** | **~2.5 hrs** | **Complete experiment** |

---

## Outputs

### Code
- `prior_predictive_check/code/prior_predictive.py`
- `simulation_based_validation/code/simulate_recover.py`
- `posterior_inference/code/fit_model.py`
- `posterior_inference/code/logarithmic_model.stan`
- `posterior_predictive_check/code/posterior_checks.py`

### Plots
- Prior predictive: Sampled functions, marginal distributions
- Simulation: Parameter recovery, calibration
- Posterior: Trace plots, pair plots, posterior distributions
- PPC: Y_rep vs Y_obs, residuals, test statistics

### Reports
- `prior_predictive_check/findings.md`
- `simulation_based_validation/recovery_metrics.md`
- `posterior_inference/inference_summary.md`
- `posterior_predictive_check/ppc_findings.md`
- `model_critique/decision.md` (ACCEPT/REVISE/REJECT)

### Data
- `posterior_inference/diagnostics/posterior_inference.netcdf` (ArviZ InferenceData with log_lik)

---

## Notes

- This is the PRIMARY model based on EDA findings
- Success of this experiment determines whether more complex models are needed
- Even if ACCEPTed, will still fit Experiment 2 (hierarchical) per Minimum Attempt Policy
- Serves as baseline for all model comparisons

---

**Status**: Ready for prior-predictive-checker agent
**Next Step**: Launch prior-predictive-checker for validation
