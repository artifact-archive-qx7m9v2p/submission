/*
Latent AR(1) Negative Binomial Model (Experiment 3)

Model addresses residual autocorrelation from Experiment 1 by adding AR(1) errors.

Observation model:
    C_t ~ NegativeBinomial(μ_t, φ)
    log(μ_t) = α_t

Latent AR(1) state process:
    α_t = β₀ + β₁·year_t + β₂·year_t² + ε_t
    ε_t = ρ·ε_{t-1} + η_t
    η_t ~ Normal(0, σ_η)
    ε_1 ~ Normal(0, σ_η/√(1-ρ²))

Non-centered parameterization for better sampling:
    ε_raw[t] ~ Normal(0, 1)
    ε[t] = ρ·ε[t-1] + σ_η·ε_raw[t]
*/

data {
  int<lower=1> N;           // Number of observations
  vector[N] year;           // Standardized year
  array[N] int<lower=0> C;  // Case counts
}

parameters {
  // Trend parameters
  real beta_0;
  real beta_1;
  real beta_2;

  // AR(1) parameters
  real<lower=0, upper=1> rho;    // AR(1) coefficient
  real<lower=0> sigma_eta;       // Innovation SD

  // Dispersion parameter
  real<lower=0> phi;

  // Non-centered parameterization for AR(1) errors
  real epsilon_raw_0;              // Initial error (raw)
  vector[N-1] epsilon_raw;         // Subsequent errors (raw)
}

transformed parameters {
  vector[N] epsilon;      // AR(1) errors
  vector[N] alpha;        // Latent state (log scale)
  vector[N] mu;           // Expected counts

  // Initial condition: ε[1] ~ Normal(0, σ_η/√(1-ρ²))
  real sigma_0 = sigma_eta / sqrt(1 - square(rho));
  epsilon[1] = sigma_0 * epsilon_raw_0;

  // AR(1) process: ε[t] = ρ·ε[t-1] + σ_η·ε_raw[t]
  for (t in 2:N) {
    epsilon[t] = rho * epsilon[t-1] + sigma_eta * epsilon_raw[t-1];
  }

  // Latent state: α[t] = trend + ε[t]
  for (t in 1:N) {
    alpha[t] = beta_0 + beta_1 * year[t] + beta_2 * square(year[t]) + epsilon[t];
  }

  // Expected counts
  mu = exp(alpha);
}

model {
  // Priors for trend parameters
  beta_0 ~ normal(4.7, 0.3);
  beta_1 ~ normal(0.8, 0.2);
  beta_2 ~ normal(0.3, 0.1);

  // Priors for AR(1) parameters
  // rho ~ Beta(12, 3) gives mean = 0.8
  rho ~ beta(12, 3);
  sigma_eta ~ normal(0, 0.5);  // HalfNormal through constraint

  // Prior for dispersion
  phi ~ gamma(2, 0.5);

  // Non-centered parameterization priors
  epsilon_raw_0 ~ std_normal();
  epsilon_raw ~ std_normal();

  // Likelihood
  C ~ neg_binomial_2(mu, phi);
}

generated quantities {
  // Pointwise log-likelihood for LOO-CV
  vector[N] log_lik;

  for (t in 1:N) {
    log_lik[t] = neg_binomial_2_lpmf(C[t] | mu[t], phi);
  }
}
