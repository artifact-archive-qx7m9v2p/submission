
data {
  int<lower=0> N;
  array[N] int<lower=0> C;
  vector[N] year;
  vector[N] year_sq;
}

parameters {
  real beta0;
  real beta1;
  real beta2;  // Quadratic term
  real<lower=0> phi;
}

transformed parameters {
  vector[N] log_mu;
  vector[N] mu;

  log_mu = beta0 + beta1 * year + beta2 * year_sq;
  mu = exp(log_mu);
}

model {
  // Priors
  beta0 ~ normal(4.3, 1.0);
  beta1 ~ normal(0.85, 0.5);
  beta2 ~ normal(0, 0.5);
  phi ~ exponential(0.667);

  // Likelihood
  C ~ neg_binomial_2(mu, phi);
}

generated quantities {
  vector[N] log_lik;
  array[N] int C_pred;

  for (i in 1:N) {
    log_lik[i] = neg_binomial_2_lpmf(C[i] | mu[i], phi);
    C_pred[i] = neg_binomial_2_rng(mu[i], phi);
  }
}
