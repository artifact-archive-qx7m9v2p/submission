data {
  int<lower=0> N;                    // Number of observations
  vector[N] year;                    // Standardized year predictor
  array[N] int<lower=0> C;           // Count response
}

parameters {
  real beta_0;                       // Intercept (log scale)
  real beta_1;                       // Slope (growth rate)
  real<lower=0> phi;                 // Dispersion parameter
}

transformed parameters {
  vector[N] mu;                      // Expected counts

  for (i in 1:N) {
    mu[i] = exp(beta_0 + beta_1 * year[i]);
  }
}

model {
  // Priors
  beta_0 ~ normal(log(109.4), 1.0);
  beta_1 ~ normal(1.0, 0.5);
  phi ~ gamma(2, 0.1);

  // Likelihood
  C ~ neg_binomial_2(mu, phi);
}

generated quantities {
  // Posterior predictive samples
  array[N] int C_rep;

  // Log-likelihood for LOO
  vector[N] log_lik;

  // Pearson residuals
  vector[N] residuals;

  for (i in 1:N) {
    C_rep[i] = neg_binomial_2_rng(mu[i], phi);
    log_lik[i] = neg_binomial_2_lpmf(C[i] | mu[i], phi);
    residuals[i] = (C[i] - mu[i]) / sqrt(mu[i] + mu[i]^2 / phi);
  }
}
