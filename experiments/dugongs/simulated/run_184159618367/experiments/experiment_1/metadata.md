# Experiment 1: Asymptotic Exponential Model

## Model Specification

**Hypothesis**: Smooth saturation via exponential approach to asymptote (mechanistic process like enzyme kinetics, learning curves).

### Functional Form
```
Y_i ~ Normal(μ_i, σ)
μ_i = α - β * exp(-γ * x_i)
```

### Parameters
- `α`: Asymptote (upper limit as x → ∞)
- `β`: Amplitude (difference from minimum to asymptote)
- `γ`: Rate parameter (speed of saturation, units: 1/x)
- `σ`: Residual standard deviation

### Priors

```
α ~ Normal(2.55, 0.1)      # Plateau observed at 2.5-2.6
β ~ Normal(0.9, 0.2)       # Range from min (~1.65) to plateau (~2.55)
γ ~ Gamma(4, 20)           # E[γ]=0.2, transition over ~10 x-units
σ ~ Half-Cauchy(0, 0.15)   # Pure error ~0.075-0.12
```

**Prior Justification** (from EDA):
- α: Y plateaus at 2.5-2.6 in observed data
- β: Back-extrapolation to x→0 suggests Y_min ≈ 1.65, so β ≈ 2.55 - 1.65 = 0.9
- γ: Transition occurs over ~10 units of x, so γ ≈ 0.1-0.3 is reasonable
- σ: Pure error from replicates ≈ 0.075-0.12

## Theoretical Justification

**Mechanism**: System approaches equilibrium asymptotically (common in enzyme kinetics, learning curves, resource depletion, saturation processes).

**Why this model?**
1. **Theoretically motivated**: Represents many natural saturation processes
2. **Interpretable**: Clear meaning for all parameters (asymptote, rate, amplitude)
3. **Smooth**: No discontinuities, gradual saturation
4. **EDA support**: Smooth saturation pattern observed, no sharp breakpoint

## Falsification Criteria

**Abandon this model if**:
1. R² < 0.80 (inadequate fit)
2. γ posterior includes zero or is extremely uncertain (no saturation rate detectable)
3. Convergence fails repeatedly (R-hat > 1.01 after tuning)
4. Systematic residual patterns indicating misspecification
5. α not in plausible range [2.0, 3.0]

## Expected Performance

- **R²**: ~0.88-0.89 (based on EDA asymptotic fit)
- **Convergence**: Good with proper initialization
- **Speed**: Medium (~30-60 seconds)
- **Interpretability**: Excellent (best of all models)

## Implementation

- **PPL**: Stan (CmdStanPy) or PyMC if Stan fails
- **Likelihood**: Gaussian
- **Initialization**: Use OLS-like fits to initialize parameters
- **LOO**: Save log_likelihood for model comparison

## Status

- Created: 2025-10-27
- Status: Ready for implementation
