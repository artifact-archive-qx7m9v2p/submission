data {
  int<lower=0> N;
  vector[N] year;
  array[N] int<lower=0> C;
}

parameters {
  real beta_0;
  real beta_1;
  real beta_2;                       // Quadratic coefficient
  real<lower=0> phi;
}

transformed parameters {
  vector[N] mu;

  for (i in 1:N) {
    mu[i] = exp(beta_0 + beta_1 * year[i] + beta_2 * year[i]^2);
  }
}

model {
  // Priors
  beta_0 ~ normal(log(109.4), 1.0);
  beta_1 ~ normal(1.0, 0.5);
  beta_2 ~ normal(0, 0.5);           // Skeptical of quadratic
  phi ~ gamma(2, 0.1);

  // Likelihood
  C ~ neg_binomial_2(mu, phi);
}

generated quantities {
  array[N] int C_rep;
  vector[N] log_lik;
  vector[N] residuals;

  // Derived quantity: is quadratic needed?
  int<lower=0, upper=1> quadratic_important;
  quadratic_important = fabs(beta_2) > 0.1 ? 1 : 0;

  for (i in 1:N) {
    C_rep[i] = neg_binomial_2_rng(mu[i], phi);
    log_lik[i] = neg_binomial_2_lpmf(C[i] | mu[i], phi);
    residuals[i] = (C[i] - mu[i]) / sqrt(mu[i] + mu[i]^2 / phi);
  }
}
