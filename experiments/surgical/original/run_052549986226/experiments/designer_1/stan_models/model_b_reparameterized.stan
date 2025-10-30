// Model B: Reparameterized Beta-Binomial
// Mean-concentration parameterization for better interpretability
// Designer 1 - Reparameterized model

data {
  int<lower=1> N;
  array[N] int<lower=0> n_trials;
  array[N] int<lower=0> r_success;
}

parameters {
  real<lower=0, upper=1> mu;       // Mean success rate
  real<lower=0> kappa;             // Concentration parameter
}

transformed parameters {
  real<lower=0> alpha = mu * kappa;
  real<lower=0> beta = (1 - mu) * kappa;
}

model {
  // Hyperpriors
  mu ~ beta(2, 18);     // Prior mean ~ 0.1, close to observed 0.076
  kappa ~ gamma(2, 0.1); // Prior mean = 20, but allow wide range

  // Likelihood
  r_success ~ beta_binomial(n_trials, alpha, beta);
}

generated quantities {
  // Variance and dispersion metrics
  real var_p = (mu * (1 - mu)) / (kappa + 1);
  real phi = 1 + 1 / kappa;  // Approximate overdispersion
  real icc = 1 / (1 + kappa); // Approximate intraclass correlation

  // Posterior predictive for group-level rates
  array[N] real<lower=0, upper=1> p_rep;
  for (i in 1:N) {
    p_rep[i] = beta_rng(alpha, beta);
  }

  // Posterior predictive for data
  array[N] int r_rep;
  for (i in 1:N) {
    real p_i = beta_rng(alpha, beta);
    r_rep[i] = binomial_rng(n_trials[i], p_i);
  }

  // Log-likelihood for LOO-CV
  array[N] real log_lik;
  for (i in 1:N) {
    log_lik[i] = beta_binomial_lpmf(r_success[i] | n_trials[i], alpha, beta);
  }

  // Posterior mean estimates for each group
  array[N] real p_posterior_mean;
  for (i in 1:N) {
    p_posterior_mean[i] = (r_success[i] + alpha) / (n_trials[i] + alpha + beta);
  }

  // Shrinkage factor: how much does each group shrink toward population mean?
  array[N] real shrinkage_factor;
  for (i in 1:N) {
    // Shrinkage = kappa / (kappa + n_i)
    shrinkage_factor[i] = kappa / (kappa + n_trials[i]);
  }
}
