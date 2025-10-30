data {
  int<lower=0> N;
  vector[N] x;
  vector[N] y;
}

parameters {
  real beta_0;
  real beta_1;
  real gamma_0;
  real gamma_1;
}

transformed parameters {
  vector[N] mu;
  vector[N] log_sigma;
  vector[N] sigma;

  for (i in 1:N) {
    mu[i] = beta_0 + beta_1 * log(x[i]);
    log_sigma[i] = gamma_0 + gamma_1 * x[i];
    sigma[i] = exp(log_sigma[i]);
  }
}

model {
  // Priors
  beta_0 ~ normal(1.8, 0.5);
  beta_1 ~ normal(0.3, 0.2);
  gamma_0 ~ normal(-2, 1);
  gamma_1 ~ normal(-0.05, 0.05);

  // Likelihood
  y ~ normal(mu, sigma);
}

generated quantities {
  vector[N] y_rep;
  vector[N] log_lik;

  for (i in 1:N) {
    y_rep[i] = normal_rng(mu[i], sigma[i]);
    log_lik[i] = normal_lpdf(y[i] | mu[i], sigma[i]);
  }
}
