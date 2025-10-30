// Model 1: Centered Parameterization
// Hierarchical binomial model with random effects on logit scale
// Standard parameterization: logit(p_i) = mu + alpha_i
//
// WARNING: May have computational issues (divergences, poor mixing)
// due to funnel geometry when sigma is uncertain and ICC is high.
// Recommended to use Model 2 (non-centered) instead.

data {
  int<lower=1> N;                    // Number of groups
  array[N] int<lower=0> n_trials;    // Trials per group
  array[N] int<lower=0> r_successes; // Successes per group
}

parameters {
  real mu;                           // Population mean (logit scale)
  real<lower=0> sigma;               // Between-group SD
  vector[N] alpha;                   // Group-specific effects
}

model {
  // Priors
  mu ~ normal(-2.5, 1.0);           // Centered on logit(0.076)
  sigma ~ cauchy(0, 1);             // Half-Cauchy for scale parameter
  alpha ~ normal(0, sigma);         // Group deviations (centered!)

  // Likelihood
  for (i in 1:N) {
    real p_i = inv_logit(mu + alpha[i]);
    r_successes[i] ~ binomial(n_trials[i], p_i);
  }
}

generated quantities {
  // Posterior predictions and diagnostics
  vector[N] p_posterior;             // Posterior success rates
  array[N] int r_pred;               // Posterior predictive counts
  real phi_posterior;                // Posterior overdispersion
  real mean_success_rate;            // Population mean on probability scale
  vector[N] log_lik;                 // Log-likelihood for LOO-CV

  // Calculate posteriors
  for (i in 1:N) {
    p_posterior[i] = inv_logit(mu + alpha[i]);
    r_pred[i] = binomial_rng(n_trials[i], p_posterior[i]);
    log_lik[i] = binomial_lpmf(r_successes[i] | n_trials[i], p_posterior[i]);
  }

  mean_success_rate = inv_logit(mu);

  // Calculate posterior overdispersion
  // phi = 1 + n_bar * Var(p_i) / (mean_p * (1 - mean_p))
  {
    real mean_p = inv_logit(mu);
    real var_p = variance(p_posterior);
    real mean_n = mean(to_vector(n_trials));
    phi_posterior = 1 + mean_n * var_p / (mean_p * (1 - mean_p));
  }
}
