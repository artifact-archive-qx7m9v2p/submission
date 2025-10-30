data {
  int<lower=0> N;
  vector[N] year;
  array[N] int<lower=0> C;
}

parameters {
  real beta_0;
  real beta_1;
  real<lower=0> phi;
}

model {
  // Priors
  beta_0 ~ normal(4.69, 1.0);
  beta_1 ~ normal(1.0, 0.5);
  phi ~ gamma(2, 0.1);

  // Likelihood
  for (t in 1:N) {
    real mu = exp(beta_0 + beta_1 * year[t]);
    C[t] ~ neg_binomial_2(mu, phi);
  }
}

generated quantities {
  vector[N] log_lik;
  array[N] int C_rep;
  for (t in 1:N) {
    real mu = exp(beta_0 + beta_1 * year[t]);
    log_lik[t] = neg_binomial_2_lpmf(C[t] | mu, phi);
    C_rep[t] = neg_binomial_2_rng(mu, phi);
  }
}
