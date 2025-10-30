# Implementation Note: PPL Selection

**Date**: 2025-10-30
**Experiment**: Experiment 1 - Negative Binomial GLM with Quadratic Trend

---

## PPL Used: PyMC (Not Stan)

This validation was implemented in **PyMC 5.26.1** rather than the planned Stan/CmdStanPy.

### Reason for Change

CmdStan compilation requires the `make` build tool, which is not available in this environment:

```
ValueError: No CmdStan installation found, run command "install_cmdstan"
or (re)activate your conda environment!
```

Attempted `cmdstanpy.install_cmdstan()` which downloads CmdStan but fails at compilation:

```
Command "make build" failed
failed with error [Errno 2] No such file or directory: 'make'
```

### Why This Is Acceptable

1. **PyMC implements identical model**:
   - Negative Binomial likelihood with same parameterization
   - Same prior specifications
   - Same NUTS sampler (via PyTensor)

2. **Validation results are PPL-agnostic**:
   - Testing parameter recovery, not software implementation
   - MCMC diagnostics (R-hat, ESS) have same interpretation
   - Goal is to validate model specification, not specific PPL

3. **Stan model preserved for reference**:
   - `/workspace/experiments/experiment_1/simulation_based_validation/code/model.stan`
   - Includes `generated quantities` block with `log_lik` for LOO-CV
   - Can be compiled later if `make` becomes available

### PyMC vs Stan Equivalence

**Stan parameterization**:
```stan
C ~ neg_binomial_2(mu, phi)
// mean = mu, variance = mu + mu^2/phi
```

**PyMC parameterization**:
```python
pm.NegativeBinomial('obs', mu=mu, alpha=phi)
# mean = mu, variance = mu + mu^2/alpha
```

Both use the **NB2 (mean-dispersion)** parameterization where:
- E[C] = μ
- Var[C] = μ + μ²/φ

**Prior specifications** are identical:
- beta_0 ~ Normal(4.5, 1.0)
- beta_1 ~ Normal(0.9, 0.5)
- beta_2 ~ Normal(0, 0.3)
- phi ~ Gamma(2, 0.1)

### Performance Comparison

**PyMC results**:
- Sampling time: 40.4 seconds
- Convergence: R-hat = 1.000 for all parameters
- ESS: >2400 for all parameters (excellent)
- Divergences: 0

**Expected Stan performance** (based on similar models):
- Likely faster (5-15 seconds)
- Similar or slightly better ESS per second
- Comparable convergence

**Conclusion**: PyMC provided fully adequate performance for this validation task.

---

## Files

### Stan Model (Preserved for Reference)
`/workspace/experiments/experiment_1/simulation_based_validation/code/model.stan`

Key features:
- Complete data/parameters/model blocks
- Generated quantities with `log_lik` for LOO-CV
- Ready for compilation if make becomes available

### PyMC Implementation (Used)
`/workspace/experiments/experiment_1/simulation_based_validation/code/simulation_validation_pymc.py`

Key features:
- Complete simulation-based calibration pipeline
- Identical model specification to Stan version
- Comprehensive diagnostics and visualization
- Produces same validation metrics

---

## Impact on Workflow

**For this validation**: None. Results are scientifically equivalent.

**For real data fitting**:
- Continue with PyMC for consistency
- OR attempt Stan if build environment is configured
- Recommendation: Use PyMC for Experiment 1, reassess for later experiments

**For reproducibility**:
- Document PyMC version (5.26.1) in methods
- Note Stan model as "planned but unavailable due to environment"
- Provide both implementations in supplementary materials

---

## Recommendation

**Proceed with PyMC for Experiment 1 real data fitting** because:
1. Validation successful with PyMC
2. Model specification proven correct
3. No need to switch PPL mid-experiment
4. Can revisit Stan for future experiments if needed

If Stan becomes critical for specific features (e.g., certain generated quantities, C++ speedups), we can:
1. Install make via system package manager
2. Recompile Stan model
3. Verify equivalence with PyMC results
4. Switch PPL if substantial benefits

---

**Decision**: Use PyMC for Experiment 1. Stan model preserved for optional future use.
