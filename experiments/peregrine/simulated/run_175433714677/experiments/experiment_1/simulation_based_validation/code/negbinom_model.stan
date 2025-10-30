data {
  int<lower=0> n;              // Number of observations
  array[n] int<lower=0> C;     // Count data
  vector[n] year;              // Year predictor (standardized)
}

parameters {
  real beta_0;                 // Intercept
  real beta_1;                 // Slope
  real<lower=0> phi;          // Dispersion parameter
}

transformed parameters {
  vector[n] log_mu;

  // Linear predictor on log scale
  log_mu = beta_0 + beta_1 * year;
}

model {
  // Priors
  beta_0 ~ normal(4.3, 1.0);
  beta_1 ~ normal(0.85, 0.5);
  phi ~ exponential(0.667);

  // Likelihood using neg_binomial_2_log parameterization
  // neg_binomial_2_log(eta, phi) where eta = log(mu)
  C ~ neg_binomial_2_log(log_mu, phi);
}

generated quantities {
  // Log-likelihood for LOO-CV
  vector[n] log_lik;

  // Posterior predictive samples
  array[n] int C_rep;

  for (i in 1:n) {
    log_lik[i] = neg_binomial_2_log_lpmf(C[i] | log_mu[i], phi);
    C_rep[i] = neg_binomial_2_log_rng(log_mu[i], phi);
  }
}
