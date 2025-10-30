data {
  int<lower=1> N;                    // Number of observations
  array[N] int<lower=0> C;           // Count data
  vector[N] year;                     // Standardized year covariate
  int<lower=1,upper=N> tau;          // Fixed changepoint location
}

parameters {
  real beta_0;                        // Intercept
  real beta_1;                        // Pre-break slope
  real beta_2;                        // Additional post-break slope
  real<lower=0> alpha;                // Dispersion parameter
  real<lower=0,upper=1> rho;          // AR(1) coefficient
  real<lower=0> sigma_eps;            // Innovation standard deviation
  vector[N] z;                        // Standard normal innovations (non-centered)
}

transformed parameters {
  vector[N] eps;                      // AR(1) errors
  vector[N] log_mu;                   // Log mean
  vector[N] mu;                       // Expected count
  real year_tau = year[tau];          // Year at changepoint

  // Non-centered AR(1) process with stationary initialization
  eps[1] = (sigma_eps / sqrt(1 - rho^2)) * z[1];
  for (t in 2:N) {
    eps[t] = rho * eps[t-1] + sigma_eps * z[t];
  }

  // Mean structure with fixed changepoint
  for (t in 1:N) {
    if (t <= tau) {
      log_mu[t] = beta_0 + beta_1 * year[t] + eps[t];
    } else {
      log_mu[t] = beta_0 + beta_1 * year[t] + beta_2 * (year[t] - year_tau) + eps[t];
    }
  }
  mu = exp(log_mu);
}

model {
  // Priors (REVISED after prior predictive check)
  beta_0 ~ normal(4.3, 0.5);
  beta_1 ~ normal(0.35, 0.3);
  beta_2 ~ normal(0.85, 0.5);
  alpha ~ gamma(2, 3);
  rho ~ beta(12, 1);
  sigma_eps ~ exponential(2);
  z ~ std_normal();

  // Likelihood: NegBinomial with phi = 1/alpha
  // variance = mu + mu^2 / phi = mu + alpha * mu^2
  C ~ neg_binomial_2(mu, 1.0 / alpha);
}

generated quantities {
  vector[N] log_lik;
  array[N] int C_rep;                 // Posterior predictive samples

  // Point-wise log-likelihood for LOO
  for (t in 1:N) {
    log_lik[t] = neg_binomial_2_lpmf(C[t] | mu[t], 1.0 / alpha);
    C_rep[t] = neg_binomial_2_rng(mu[t], 1.0 / alpha);
  }
}
