
data {
  int<lower=0> N;
  vector[N] year;
}

generated quantities {
  // Sample from priors
  real beta_0 = normal_rng(4.69, 1.0);
  real beta_1 = normal_rng(1.0, 0.5);
  real phi = gamma_rng(2, 0.1);

  // Generate synthetic data
  array[N] int<lower=0> C_sim;
  vector[N] mu;
  for (t in 1:N) {
    mu[t] = exp(beta_0 + beta_1 * year[t]);
    C_sim[t] = neg_binomial_2_rng(mu[t], phi);
  }
}
