// Student-t Hierarchical Model for Binomial Data with Outliers
// Robust alternative to Normal hierarchical model
// Heavy-tailed random effects accommodate outliers without contamination

data {
  int<lower=1> N;                  // Number of groups
  array[N] int<lower=0> n_trials;  // Trials per group
  array[N] int<lower=0> r;         // Successes per group
}

parameters {
  real mu;                         // Population mean (logit scale)
  real<lower=0> sigma;             // Between-group SD
  real<lower=1> nu;                // Degrees of freedom (>1 for finite variance)
  vector[N] alpha_raw;             // Non-centered random effects
}

transformed parameters {
  vector[N] alpha;
  vector[N] logit_p;

  // Non-centered parameterization for efficiency
  alpha = sigma * alpha_raw;
  logit_p = mu + alpha;
}

model {
  // Priors
  mu ~ normal(-2.5, 1);            // Population mean, centered on 7.6%
  sigma ~ cauchy(0, 1);            // Weakly informative scale
  nu ~ gamma(2, 0.1);              // Heavy-tailed prior: mode at 10

  // Student-t random effects (non-centered)
  alpha_raw ~ student_t(nu, 0, 1);

  // Likelihood
  r ~ binomial_logit(n_trials, logit_p);
}

generated quantities {
  // Posterior predictive checks
  array[N] int r_rep;
  vector[N] p = inv_logit(logit_p);
  real mean_p = mean(p);

  // Posterior predictive samples
  for (i in 1:N) {
    r_rep[i] = binomial_rng(n_trials[i], p[i]);
  }

  // Compute overdispersion parameter from posterior
  real var_p = variance(p);
  real expected_var_p = mean_p * (1 - mean_p) / mean(to_vector(n_trials));
  real phi_posterior = var_p / expected_var_p;

  // Flag outlier groups (|alpha| > 2)
  array[N] int is_outlier;
  for (i in 1:N) {
    is_outlier[i] = (fabs(alpha[i]) > 2.0) ? 1 : 0;
  }

  // Log-likelihood for LOO-CV
  vector[N] log_lik;
  for (i in 1:N) {
    log_lik[i] = binomial_logit_lpmf(r[i] | n_trials[i], logit_p[i]);
  }
}
