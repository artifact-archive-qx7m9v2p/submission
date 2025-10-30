# Designer 2 - Model Proposals

## Overview

This directory contains Bayesian model proposals for the Eight Schools dataset, designed by Designer 2 working independently.

## Key Files

- **proposed_models.md**: Complete model specifications, priors, falsification criteria, and decision rules

## Three Model Classes Proposed

### Model 1: Adaptive Hierarchical Normal Model
- **Type**: Standard Bayesian hierarchical model
- **Key assumption**: Schools are exchangeable from common distribution
- **Falsify if**: tau posterior pushed far from 0, PPCs fail, extreme shrinkage
- **Implementation**: Stan with non-centered parameterization

### Model 2: Two-Component Mixture Model
- **Type**: Finite mixture model with 2 latent clusters
- **Key assumption**: Schools belong to one of two subgroups
- **Falsify if**: Clusters collapse (mu_1 ≈ mu_2), label switching, poor LOO
- **Implementation**: PyMC with mixture likelihood

### Model 3: Uncertainty in Sigma Model
- **Type**: Hierarchical model with measurement error on reported SEs
- **Key assumption**: Reported sigma_i have estimation error
- **Falsify if**: epsilon → 0, lambda_i all near 1, no PPC improvement
- **Implementation**: Stan with lognormal multiplicative error

## Design Philosophy

Each model represents a **distinct hypothesis** about why the data look the way they do:

1. **Model 1**: True homogeneity (low heterogeneity is real)
2. **Model 2**: Discrete subgroups (low heterogeneity is averaging artifact)
3. **Model 3**: Measurement error (variance paradox due to sigma misspecification)

All models have **explicit falsification criteria** - conditions under which I will abandon them.

## Expected Outcome

Given EDA findings (I² = 1.6%, variance paradox), **Model 1 is most likely adequate**. Models 2 and 3 are designed to test whether this conclusion could be wrong.

## Model Selection Strategy

1. Fit all three models
2. Compare via LOO-CV and WAIC
3. Conduct posterior predictive checks
4. Run stress tests (exclude outliers, prior sensitivity)
5. Select model with best predictive performance and theoretical plausibility
6. If models agree on key quantities (mu, shrinkage), report simplest (Model 1)

## Stress Tests Planned

1. Exclude School 5 (negative outlier)
2. Exclude School 8 (highest uncertainty)
3. Complete pooling baseline
4. Posterior predictive permutation test for I²
5. Prior sensitivity analysis

## Red Flags for Pivoting

- All models show posterior-prior conflict → consider t-likelihood
- Variance consistently wrong → question sigma_i scaling
- Computational failure everywhere → weak identification, use skeptical priors
- Extreme prior sensitivity → acknowledge n=8 too small for robust inference

## Contact

Designer 2, working independently in parallel with other designers.

Analysis date: 2025-10-29
