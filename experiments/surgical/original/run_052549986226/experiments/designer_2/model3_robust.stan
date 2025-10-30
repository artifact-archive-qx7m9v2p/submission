// Model 3: Robust Hierarchical Model with Heavy-Tailed Priors
// Non-centered parameterization with Student-t priors
// logit(p_i) = mu + sigma * z_i, z_i ~ Student-t(nu, 0, 1)
//
// Motivation:
// - Group 8 is extreme outlier (z = 3.94)
// - Heavy tails allow outliers without distorting population parameters
// - Posterior nu informs whether heavy tails were necessary
//
// If posterior nu > 30: normal adequate, use Model 2
// If posterior nu < 10: outliers present, Model 3 justified

data {
  int<lower=1> N;                    // Number of groups
  array[N] int<lower=0> n_trials;    // Trials per group
  array[N] int<lower=0> r_successes; // Successes per group
}

parameters {
  real mu;                           // Population mean (logit scale)
  real<lower=0> sigma;               // Between-group SD
  vector[N] z;                       // Standardized group effects (heavy-tailed!)
  real<lower=1> nu;                  // Degrees of freedom for Student-t
}

transformed parameters {
  vector[N] alpha = sigma * z;       // Actual group effects
}

model {
  // Priors
  mu ~ normal(-2.5, 1.0);           // Centered on logit(0.076)
  sigma ~ student_t(3, 0, 1);       // Half-Student-t via constraint
  nu ~ gamma(2, 0.1);               // Prior on degrees of freedom
  z ~ student_t(nu, 0, 1);          // Heavy-tailed group effects

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
  int<lower=0, upper=1> nu_high;     // Is nu > 30? (heavy tails unnecessary)

  // Calculate posteriors
  {
    real mean_p = inv_logit(mu);
    vector[N] p_obs = to_vector(r_successes) ./ to_vector(n_trials);

    for (i in 1:N) {
      p_posterior[i] = inv_logit(mu + alpha[i]);
      r_pred[i] = binomial_rng(n_trials[i], p_posterior[i]);
      log_lik[i] = binomial_lpmf(r_successes[i] | n_trials[i], p_posterior[i]);

      // Shrinkage calculation
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

    // Check if heavy tails were necessary
    nu_high = nu > 30 ? 1 : 0;
  }
}
