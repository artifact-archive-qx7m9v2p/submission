data {
  int<lower=0> N;
  vector[N] year;
  array[N] int<lower=0> C;
}

parameters {
  real beta_0;
  real beta_1;
  real<lower=0> alpha;               // Gamma shape parameter
  vector<lower=0>[N] lambda;         // Random Poisson rates
}

transformed parameters {
  vector[N] mu;

  for (i in 1:N) {
    mu[i] = exp(beta_0 + beta_1 * year[i]);
  }
}

model {
  // Priors
  beta_0 ~ normal(log(109.4), 1.0);
  beta_1 ~ normal(1.0, 0.5);
  alpha ~ gamma(2, 0.1);

  // Hierarchical structure
  for (i in 1:N) {
    lambda[i] ~ gamma(alpha, alpha / mu[i]);  // E[lambda] = mu
    C[i] ~ poisson(lambda[i]);
  }
}

generated quantities {
  array[N] int C_rep;
  vector[N] log_lik;
  vector[N] residuals;

  // Convert to NegBin dispersion for comparison with Model 1
  real<lower=0> phi = alpha;

  for (i in 1:N) {
    // Generate from marginal distribution
    C_rep[i] = neg_binomial_2_rng(mu[i], phi);
    log_lik[i] = neg_binomial_2_lpmf(C[i] | mu[i], phi);
    residuals[i] = (C[i] - mu[i]) / sqrt(mu[i] + mu[i]^2 / phi);
  }
}
