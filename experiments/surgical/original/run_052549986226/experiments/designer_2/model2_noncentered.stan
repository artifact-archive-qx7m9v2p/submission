// Model 2: Non-Centered Parameterization (RECOMMENDED)
// Hierarchical binomial model with random effects on logit scale
// Non-centered: logit(p_i) = mu + sigma * z_i, z_i ~ N(0,1)
//
// Advantages:
// - Eliminates funnel geometry
// - Better sampling efficiency
// - Fewer divergences
// - Faster convergence
//
// Mathematically equivalent to Model 1 but computationally superior.

data {
  int<lower=1> N;                    // Number of groups
  array[N] int<lower=0> n_trials;    // Trials per group
  array[N] int<lower=0> r_successes; // Successes per group
}

parameters {
  real mu;                           // Population mean (logit scale)
  real<lower=0> sigma;               // Between-group SD
  vector[N] z;                       // Standardized group effects
}

transformed parameters {
  vector[N] alpha = sigma * z;       // Actual group effects (non-centered!)
}

model {
  // Priors
  mu ~ normal(-2.5, 1.0);           // Centered on logit(0.076)
  sigma ~ normal(0, 1);             // Half-normal via constraint
  z ~ std_normal();                 // Standard normal (independent of sigma!)

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
  vector[N] shrinkage;               // Shrinkage toward population mean

  // Calculate posteriors
  {
    real mean_p = inv_logit(mu);
    vector[N] p_obs = to_vector(r_successes) ./ to_vector(n_trials);

    for (i in 1:N) {
      p_posterior[i] = inv_logit(mu + alpha[i]);
      r_pred[i] = binomial_rng(n_trials[i], p_posterior[i]);
      log_lik[i] = binomial_lpmf(r_successes[i] | n_trials[i], p_posterior[i]);

      // Shrinkage: proportion moved from observed to population mean
      if (p_obs[i] != mean_p) {
        shrinkage[i] = (p_obs[i] - p_posterior[i]) / (p_obs[i] - mean_p);
      } else {
        shrinkage[i] = 0;
      }
    }

    mean_success_rate = mean_p;

    // Calculate posterior overdispersion
    real var_p = variance(p_posterior);
    real mean_n = mean(to_vector(n_trials));
    phi_posterior = 1 + mean_n * var_p / (mean_p * (1 - mean_p));
  }
}
