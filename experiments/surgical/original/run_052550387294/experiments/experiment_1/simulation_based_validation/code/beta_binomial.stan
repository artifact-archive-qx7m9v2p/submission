data {
  int<lower=0> N;              // Number of trials
  array[N] int<lower=0> n;     // Sample sizes
  array[N] int<lower=0> r;     // Number of successes
}

parameters {
  real<lower=0, upper=1> mu;   // Mean success probability
  real<lower=0> phi;           // Concentration parameter
}

transformed parameters {
  real<lower=0> alpha;
  real<lower=0> beta;

  alpha = mu * phi;
  beta = (1 - mu) * phi;
}

model {
  // Priors
  mu ~ beta(2, 25);
  phi ~ gamma(2, 2);

  // Likelihood
  for (i in 1:N) {
    r[i] ~ beta_binomial(n[i], alpha, beta);
  }
}

generated quantities {
  array[N] int<lower=0> r_rep;      // Posterior predictive samples
  array[N] real log_lik;             // Pointwise log-likelihood

  for (i in 1:N) {
    r_rep[i] = beta_binomial_rng(n[i], alpha, beta);
    log_lik[i] = beta_binomial_lpmf(r[i] | n[i], alpha, beta);
  }
}
