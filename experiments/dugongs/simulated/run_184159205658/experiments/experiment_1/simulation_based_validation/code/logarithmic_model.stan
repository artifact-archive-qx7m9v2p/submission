data {
  int<lower=0> N;
  vector[N] x;
  vector[N] y;
}

transformed data {
  vector[N] log_x = log(x);
}

parameters {
  real beta0;
  real beta1;
  real<lower=0> sigma;
}

model {
  // Priors
  beta0 ~ normal(1.73, 0.5);
  beta1 ~ normal(0.28, 0.15);
  sigma ~ exponential(5);

  // Likelihood
  y ~ normal(beta0 + beta1 * log_x, sigma);
}

generated quantities {
  vector[N] y_rep;
  vector[N] log_lik;

  for (i in 1:N) {
    y_rep[i] = normal_rng(beta0 + beta1 * log_x[i], sigma);
    log_lik[i] = normal_lpdf(y[i] | beta0 + beta1 * log_x[i], sigma);
  }
}
