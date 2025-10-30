# Mathematical Specifications: Three Parametric Models

## Model 1: Logarithmic Regression

### Functional Form
```
Y_i ~ Normal(μ_i, σ)
μ_i = α + β·log(x_i)
```

### Parameters
- `α`: Intercept (Y when x=1)
- `β`: Log-slope (ΔY per unit Δlog(x))
- `σ`: Residual SD

### Priors
```
α ~ Normal(1.75, 0.5)
β ~ Normal(0.27, 0.15)
σ ~ HalfNormal(0.2)
```

### Limiting Behavior
```
lim(x→∞) Y = ∞  (unbounded, slow growth)
lim(x→0⁺) Y = α
dY/dx = β/x     (decreasing rate)
```

### Interpretation
- At x=1: Y ≈ α
- At x=10: Y ≈ α + 2.3β
- At x=100: Y ≈ α + 4.6β
- Each 10-fold increase adds constant β·ln(10) ≈ 2.3β

---

## Model 2: Michaelis-Menten Saturation

### Functional Form
```
Y_i ~ Normal(μ_i, σ)
μ_i = Y_max - (Y_max - Y_min)·K/(K + x_i)
```

Alternative form:
```
μ_i = Y_min + (Y_max - Y_min)·x_i/(K + x_i)
```

### Parameters
- `Y_max`: Asymptotic maximum
- `Y_min`: Baseline minimum
- `K`: Half-saturation constant
- `σ`: Residual SD

### Priors
```
Y_max ~ Normal(2.7, 0.3)  [truncated at max(Y)]
Y_min ~ Normal(1.5, 0.3)
K ~ Normal(5, 3)  [K > 0]
σ ~ HalfNormal(0.2)
```

### Limiting Behavior
```
lim(x→∞) Y = Y_max  (finite asymptote)
lim(x→0⁺) Y = Y_min
At x=K: Y = (Y_max + Y_min)/2  (half-saturation)
dY/dx = (Y_max - Y_min)·K/(K + x)²  (decreasing)
```

### Interpretation
- At x=K: Y is halfway between baseline and maximum
- At x=10K: Y ≈ Y_max - 0.1·(Y_max - Y_min) (90% saturated)
- Saturation rate controlled by K:
  - Small K: Rapid saturation
  - Large K: Slow saturation

---

## Model 3: Quadratic Polynomial

### Functional Form
```
Y_i ~ Normal(μ_i, σ)
μ_i = α + β₁·x_i + β₂·x_i²
```

### Parameters
- `α`: Intercept (Y at x=0)
- `β₁`: Linear coefficient
- `β₂`: Quadratic coefficient
- `σ`: Residual SD

### Priors
```
α ~ Normal(1.7, 0.3)
β₁ ~ Normal(0.1, 0.05)
β₂ ~ Normal(-0.002, 0.001)
σ ~ HalfNormal(0.15)
```

### Limiting Behavior
```
If β₂ < 0 (expected):
  lim(x→∞) Y = -∞  (eventually decreases - IMPLAUSIBLE)
  Vertex at x = -β₁/(2β₂)
  Maximum Y at vertex

If β₂ > 0 (unlikely):
  lim(x→∞) Y = +∞  (accelerating growth)
  Vertex is minimum
```

### Interpretation
- `dY/dx = β₁ + 2β₂·x` (linear change in slope)
- At x=0: Slope = β₁
- At x=10: Slope = β₁ + 20β₂
- **Warning**: Only valid within observed range!

### Vertex Analysis (Critical)
For β₂ < 0 (concave down):
```
x_vertex = -β₁/(2β₂)
Y_vertex = α + β₁·x_vertex + β₂·x_vertex²

If x_vertex < max(x):
  → Model predicts Y decreases beyond vertex
  → SCIENTIFICALLY IMPLAUSIBLE
  → Use for interpolation only
```

---

## Comparison Table

| Feature | Logarithmic | Michaelis-Menten | Quadratic |
|---------|-------------|------------------|-----------|
| Parameters | 3 | 4 | 4 |
| Asymptote | None (∞) | Finite (Y_max) | None (or -∞) |
| Valid range | All x>0 | All x>0 | Local only |
| Extrapolation | Safe | Safe | UNSAFE |
| Interpretation | Clear | Clear | Empirical |
| Identifiability | High | Medium | High |
| Computational | Easy | Moderate | Easy |

---

## Expected Predictions at Key x Values

Using EDA-estimated parameters:

### At x = 1
- **Log**: α = 1.75
- **MM**: Y_min + (Y_max - Y_min)·1/(K + 1) ≈ 1.70
- **Quad**: α + β₁ + β₂ ≈ 1.84

### At x = 10
- **Log**: 1.75 + 0.27·ln(10) = 1.75 + 0.62 = 2.37
- **MM**: Y_min + (Y_max - Y_min)·10/(K + 10) ≈ 2.35
- **Quad**: 1.75 + 0.86 - 0.20 = 2.41

### At x = 31.5 (max observed)
- **Log**: 1.75 + 0.27·ln(31.5) = 1.75 + 0.93 = 2.68
- **MM**: Y_min + (Y_max - Y_min)·31.5/(K + 31.5) ≈ 2.58
- **Quad**: 1.75 + 2.71 - 1.98 = 2.48

### At x = 100 (extrapolation)
- **Log**: 1.75 + 0.27·ln(100) = 1.75 + 1.24 = 2.99
- **MM**: ≈ Y_max ≈ 2.70
- **Quad**: 1.75 + 8.60 - 20.0 = -9.65 (NEGATIVE! IMPLAUSIBLE!)

**Key insight**: Only log and MM are safe for extrapolation beyond observed range.

---

## Derivative Comparison (Rate of Change)

### dY/dx

**Logarithmic**:
```
dY/dx = β/x
```
- Decreases as 1/x
- Always positive (if β>0)
- At x=1: β
- At x=10: β/10
- At x=100: β/100

**Michaelis-Menten**:
```
dY/dx = (Y_max - Y_min)·K/(K + x)²
```
- Decreases as 1/(K+x)²
- Always positive (if Y_max > Y_min)
- At x=K: (Y_max - Y_min)/(4K)
- Approaches 0 as x→∞

**Quadratic**:
```
dY/dx = β₁ + 2β₂·x
```
- Linear in x
- Changes sign at x = -β₁/(2β₂)
- If β₂<0: Decreasing slope, eventually negative

---

## Second Derivative (Curvature)

### d²Y/dx²

**Logarithmic**:
```
d²Y/dx² = -β/x²
```
- Always negative (if β>0) → Concave down
- Curvature decreases with x

**Michaelis-Menten**:
```
d²Y/dx² = -2(Y_max - Y_min)·K/(K + x)³
```
- Always negative (if Y_max > Y_min) → Concave down
- Approaches 0 as x→∞

**Quadratic**:
```
d²Y/dx² = 2β₂
```
- Constant curvature
- Negative if β₂<0 (expected)

---

## Model Selection Criteria

### Prefer Logarithmic if:
- Data shows unbounded slow growth
- No evidence of true plateau
- Simplicity preferred (fewer parameters)
- Extrapolation needed

### Prefer Michaelis-Menten if:
- Strong evidence of asymptotic saturation
- Y_max well-identified from data
- Scientific mechanism suggests finite limit
- High-x data shows clear plateau

### Prefer Quadratic if:
- Neither log nor MM fits well
- Only interpolation needed (no extrapolation)
- Empirical approximation acceptable
- Short-term predictions only

---

## Prior Predictive Ranges (95% Interval)

### Logarithmic at x=10
```
μ ~ Normal(α + β·ln(10), combined uncertainty)
α ~ Normal(1.75, 0.5) → [0.75, 2.75]
β·ln(10) ~ Normal(0.62, 0.35) → [-0.08, 1.32]
Combined: μ ∈ [1.5, 3.0] approximately
```

### MM at x=10
```
Y_max ~ Normal(2.7, 0.3) → [2.1, 3.3]
Y_min ~ Normal(1.5, 0.3) → [0.9, 2.1]
K ~ Normal(5, 3) → [0, 11]
At x=10: μ ∈ [1.5, 3.0] approximately
```

### Quadratic at x=10
```
α ~ Normal(1.7, 0.3) → [1.1, 2.3]
β₁·10 ~ Normal(1.0, 0.5) → [0, 2.0]
β₂·100 ~ Normal(-0.2, 0.1) → [-0.4, 0]
Combined: μ ∈ [1.5, 3.0] approximately
```

All priors produce similar ranges at x=10, showing they are similarly informative.

---

## Falsification Test Statistics

### For Logarithmic
1. **max(Y_rep) - max(Y_obs)**: Should be ≈ 0
   - If consistently negative: Model underestimates maximum
2. **Residuals at high x**: Should be uncorrelated with x
3. **Proportion of Y_rep > max(Y_obs)**: Should be ≈ 0.5

### For MM
1. **Y_max - max(Y_obs)**: Should be small but positive
2. **Saturation percentage at max(x)**: Should be <100%
3. **K < max(x)**: Should be true with high probability

### For Quadratic
1. **P(vertex_x < max(x))**: Should be ≈ 0
2. **Residual curvature**: Should match data curvature
3. **Extrapolation at 1.5×max(x)**: Should still be positive

---

**Summary**: Three distinct models with different assumptions, limiting behaviors, and valid ranges. Logarithmic and MM are scientifically plausible for extrapolation; quadratic is empirical approximation only.
