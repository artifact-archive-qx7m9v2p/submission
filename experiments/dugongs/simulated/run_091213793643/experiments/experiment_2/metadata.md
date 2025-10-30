# Experiment 2: Hierarchical Replicate Model

**Created**: 2025-01-XX
**Status**: In Progress - Prior Predictive Check
**Priority**: HIGH (Required minimum attempt #2)

---

## Model Specification

### Scientific Hypothesis
The data contains replicate observations at 6 x-values (14 total observations with replicates). This hierarchical model tests whether there is meaningful between-group variance beyond measurement noise. If replicates were taken under different conditions (batches, times, instruments), between-group variance would capture this structure.

### Replicate Structure in Data
- Total observations: N = 27
- Unique x values: 20
- x-values with replicates: 6 (x = 1.5, 5.0, 9.5, 12.0, 13.0, 15.5)
- Replicate counts: 3, 2, 2, 2, 2, 2 (respectively)
- Total observations with replicates: 14
- Singleton observations: 13

### Mathematical Form

**Likelihood**:
```
Y_ij ~ Normal(μ_j + δ_j, σ_within)
μ_j = α + β·log(x_j)
δ_j ~ Normal(0, σ_between)
```

**Structure**:
- **i**: Observation index within group j
- **j**: Group index (unique x values)
- **μ_j**: Population-level trend at x_j
- **δ_j**: Group-specific deviation from trend
- **σ_within**: Within-group (measurement) noise
- **σ_between**: Between-group variation

**Parameters**:
- α: Intercept
- β: Logarithmic slope
- σ_within: Within-group standard deviation
- σ_between: Between-group standard deviation
- δ_j: Group effects (20 values, one per unique x)

**Key Quantity**:
- ICC (Intraclass Correlation) = σ²_between / (σ²_between + σ²_within)
- Interpretation: Proportion of variance due to between-group differences

### Prior Specifications

```
α ~ Normal(1.75, 0.5)
  # Same as Experiment 1 (intercept)

β ~ Normal(0.27, 0.15)
  # Same as Experiment 1 (logarithmic slope)

σ_within ~ HalfNormal(0.15)
  # Within-group noise, slightly tighter than Exp 1
  # Expect σ_within < total σ from Exp 1 (0.125)

σ_between ~ HalfNormal(0.1)
  # Between-group variation, mode at 0 (hierarchical shrinkage)
  # Scale=0.1 allows ICC up to ~0.3
  # Weakly informative: allows data to determine magnitude

δ_j ~ Normal(0, σ_between) for j=1,...,20
  # Group-specific deviations
  # Centered at 0 (no systematic group effects)
  # Partial pooling: groups share information via σ_between
```

### Prior Justification

1. **α, β priors**: Identical to Experiment 1 for consistency. Hierarchical structure doesn't change mean function.

2. **σ_within prior**: Slightly tighter than Experiment 1's σ (HalfNormal(0.2)) because we expect within-group variance to be smaller than total variance. Mode=0, scale=0.15 allows σ_within ∈ [0, 0.3] with 95% probability.

3. **σ_between prior**: Mode at 0 encourages parsimony (ICC→0 if data doesn't support hierarchy). Scale=0.1 is weakly informative:
   - If σ_between ≈ 0: ICC ≈ 0, model reduces to Experiment 1
   - If σ_between ≈ 0.1: ICC ≈ 0.4 (moderate grouping)
   - Prior allows full range but favors simpler structure

4. **δ_j prior**: Hierarchical centered parameterization. Groups are exchangeable (no prior belief about which groups differ). Partial pooling: Groups with few observations borrow strength from global mean; groups with many observations contribute more to σ_between estimation.

---

## Falsification Criteria

This model will be REJECTED if:

1. **ICC ≈ 0 (no hierarchy needed)**:
   - 95% posterior for ICC includes 0 and mode < 0.05
   - Interpretation: Between-group variance is negligible
   - Action: Revert to Experiment 1 (simpler model sufficient)

2. **σ_between >> σ_within (replicates aren't true replicates)**:
   - Ratio σ_between / σ_within > 2
   - Interpretation: "Replicates" have more variance between than within
   - Action: Investigate data provenance; replicates may be from different conditions

3. **No LOO improvement over Experiment 1**:
   - LOO-ELPD(Exp2) - LOO-ELPD(Exp1) < 2
   - Interpretation: Additional complexity not justified
   - Action: Prefer simpler Experiment 1

4. **Poor convergence for group effects**:
   - Rhat > 1.01 for >5 group effects (δ_j)
   - Interpretation: Hierarchical structure causes identifiability issues
   - Action: Try non-centered parameterization or abandon hierarchy

5. **Posterior predictive failure**:
   - Model fails to predict held-out replicates (cross-validation)
   - Coverage < 85% when predicting one replicate given others
   - Interpretation: Model doesn't capture replicate structure correctly

---

## Expected Outcomes

### Most Likely Scenario: Small Hierarchy

**Model Performance**:
- ICC: 0.02 - 0.15 (small but non-zero)
- σ_within: 0.10 - 0.13 (similar to Experiment 1 total σ)
- σ_between: 0.02 - 0.05 (small group effects)
- LOO improvement: 0 - 3 (marginal, may not be decisive)

**Parameter Posteriors**:
- α, β: Very similar to Experiment 1 (trend unchanged)
- Group effects δ_j: Mostly small, shrunk toward 0

**Interpretation**:
- Replicates show slight between-group variance
- Could be measurement conditions, temporal effects, or random
- Hierarchy is scientifically interesting but not predictively essential

**Decision**: ACCEPT but acknowledge may not improve over Experiment 1

---

### Alternative Scenario: No Hierarchy (ICC ≈ 0)

**Symptoms**:
- σ_between posterior concentrated at ~0
- ICC posterior mode < 0.01
- δ_j all shrunk to nearly 0
- LOO identical to Experiment 1

**Interpretation**:
- Replicates are truly independent measurements
- No hidden grouping structure
- Between-group variance indistinguishable from noise

**Decision**: REJECT, recommend using Experiment 1 (simpler)

---

### Surprising Scenario: Large Hierarchy (ICC > 0.3)

**Symptoms**:
- σ_between comparable to or larger than σ_within
- Some δ_j clearly non-zero
- LOO substantially better than Experiment 1 (>5)

**Interpretation**:
- Strong evidence of group structure
- "Replicates" may represent different experimental conditions
- Need metadata to understand what groups represent

**Action**:
- Investigate data provenance
- Consider batch effects or temporal trends
- Model may need additional structure (time, batch, etc.)

**Decision**: ACCEPT with caveat; recommend investigating group structure

---

## Computational Considerations

### Implementation: Stan via CmdStanPy (or PyMC fallback)

**Challenges**:
- More parameters than Experiment 1 (23 vs 3)
- Group effects δ_j may have funnel geometry (centered parameterization)
- Potential solution: Non-centered parameterization if divergences occur

**Centered parameterization** (current):
```stan
δ_j ~ Normal(0, σ_between)
```

**Non-centered alternative** (if needed):
```stan
δ_raw_j ~ Normal(0, 1)
δ_j = σ_between * δ_raw_j
```

**Expected Runtime**:
- Longer than Experiment 1 (more parameters)
- Estimate: 5-15 minutes for 4 chains × 2000 iterations

### Sampling Strategy

- **Chains**: 4 parallel chains
- **Iterations**: 2000 (1000 warmup + 1000 sampling)
- **adapt_delta**: 0.95 (increase to 0.99 if divergences > 1%)
- **max_treedepth**: 10
- **Watch for**: Divergences in funnel region (small σ_between, large δ_j)

### Convergence Criteria

- Rhat < 1.01 for all parameters (including all 20 δ_j)
- ESS_bulk > 400 for all parameters
- Divergences < 1%
- Check δ_j mixing: Should show good exploration

---

## Success Criteria

### Pre-fit Validation (Prior Predictive)
- [ ] Prior on ICC covers [0, 0.5] range
- [ ] Prior predictive replicates show reasonable spread
- [ ] Group effects δ_j not too extreme

### Pre-fit Validation (Simulation-Based)
- [ ] Can recover σ_between and σ_within separately
- [ ] ICC recovery calibrated
- [ ] Group effects δ_j properly shrunk

### Fitting
- [ ] All chains converge (Rhat < 1.01 for all parameters)
- [ ] Adequate ESS (>400) for all parameters
- [ ] No excessive divergences (<1%)
- [ ] InferenceData saved with log_likelihood

### Post-fit Validation (Posterior Predictive)
- [ ] Can predict held-out replicates
- [ ] 95% intervals contain ~95% of observations
- [ ] Group-specific predictions reasonable

### Model Critique
- [ ] ICC estimate interpretable
- [ ] LOO comparison with Experiment 1
- [ ] Sensitivity to prior on σ_between
- [ ] Group effects δ_j make scientific sense

---

## Connection to Experiment 1

**Key Differences**:
- Exp 1: Treats all observations as independent
- Exp 2: Models correlation within groups (same x value)

**Relationship**:
- If σ_between = 0: Exp 2 reduces to Exp 1
- If σ_between > 0: Exp 2 accounts for replicate structure

**Comparison Metric**: LOO-ELPD difference
- ΔELPD < 2: Models equivalent, prefer simpler (Exp 1)
- ΔELPD = 2-5: Moderate evidence for hierarchy
- ΔELPD > 5: Strong evidence for hierarchy

**Scientific Question**:
Does accounting for replicate structure improve predictions or just add complexity?

---

## Timeline

| Stage | Duration | Output |
|-------|----------|--------|
| Prior predictive check | 30 min | Findings + plots |
| Simulation-based validation | 45 min | Recovery metrics (harder with hierarchy) |
| Model fitting | 10-20 min | Posterior samples (more parameters) |
| Posterior predictive check | 45 min | PPC plots + findings |
| Model critique | 45 min | Decision document |
| **Total** | **~3 hrs** | **Complete experiment** |

---

## Outputs

### Code
- `prior_predictive_check/code/hierarchical_model.stan`
- `simulation_based_validation/code/simulate_recover_hierarchical.py`
- `posterior_inference/code/fit_hierarchical.py`
- `posterior_predictive_check/code/ppc_hierarchical.py`

### Plots
- Prior predictive: ICC distribution, group effect distributions
- Simulation: σ_between and σ_within recovery, ICC recovery
- Posterior: Trace plots (including δ_j), ICC posterior, variance decomposition
- PPC: Group-specific predictions, replicate prediction cross-validation

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

- This model explicitly addresses Designer 3's hierarchical/compositional perspective
- Tests a key data structure feature (replicates) that Experiment 1 ignores
- May not improve predictions but provides scientific insight about replicate variance
- Success is learning whether hierarchy matters, not necessarily improving LOO
- Serves as methodological complement to Experiment 1's simpler structure

---

**Status**: Ready for prior-predictive-checker agent
**Next Step**: Launch prior-predictive-checker for validation
