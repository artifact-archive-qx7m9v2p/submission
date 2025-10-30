// Negative Binomial Quadratic Model - Full Model for SBC
// Purpose: Fit model to data for simulation-based calibration

data {
  int<lower=1> N;              // Number of observations
  vector[N] year;              // Centered year values
  array[N] int<lower=0> y;     // Count data
}

parameters {
  real beta_0;                 // Intercept
  real beta_1;                 // Linear coefficient
  real beta_2;                 // Quadratic coefficient
  real<lower=0> phi;           // Overdispersion parameter
}

model {
  // Priors (ADJUSTED after prior predictive check)
  beta_0 ~ normal(4.7, 0.3);   // Tightened from 0.5
  beta_1 ~ normal(0.8, 0.2);   // Tightened from 0.3
  beta_2 ~ normal(0.3, 0.1);   // CRITICAL adjustment from 0.2
  phi ~ gamma(2, 0.5);

  // Likelihood
  vector[N] mu;
  for (i in 1:N) {
    mu[i] = exp(beta_0 + beta_1 * year[i] + beta_2 * year[i]^2);
  }
  y ~ neg_binomial_2(mu, phi);
}

generated quantities {
  // Posterior predictive samples
  array[N] int y_rep;
  vector[N] log_lik;

  {
    vector[N] mu;
    for (i in 1:N) {
      mu[i] = exp(beta_0 + beta_1 * year[i] + beta_2 * year[i]^2);
      y_rep[i] = neg_binomial_2_rng(mu[i], phi);
      log_lik[i] = neg_binomial_2_lpmf(y[i] | mu[i], phi);
    }
  }
}
