/**
 * Model 1: Heavy-Tailed Hierarchical Meta-Analysis
 * Student-t likelihood for robustness to outliers
 *
 * Designer: Model Designer 2
 * Date: 2025-10-28
 * Dataset: J=8 meta-analysis studies
 */

data {
  int<lower=1> J;                    // Number of studies
  vector[J] y;                       // Observed effects
  vector<lower=0>[J] sigma;          // Known standard errors (fixed)
}

parameters {
  real mu;                           // Population mean effect
  real<lower=0> tau;                 // Between-study standard deviation
  vector[J] theta_raw;               // Non-centered parameterization
  real<lower=1> nu;                  // Degrees of freedom (tail behavior)
}

transformed parameters {
  vector[J] theta;                   // Study-specific true effects

  // Non-centered parameterization (avoids funnel)
  theta = mu + tau * theta_raw;
}

model {
  // Priors
  mu ~ normal(0, 50);                // Weakly informative on mean
  tau ~ cauchy(0, 5);                // Heavy-tailed prior on heterogeneity
  theta_raw ~ normal(0, 1);          // Standard normal (for non-centered)
  nu ~ gamma(2, 0.1);                // Prior on df (mean=20, allows 3-100+)

  // Likelihood with t-distribution for robustness
  y ~ student_t(nu, theta, sigma);
}

generated quantities {
  // For model comparison
  vector[J] log_lik;

  // Posterior predictive checks
  vector[J] y_rep;

  // Predictions for new study
  real theta_new;
  real y_new;

  // Derived quantities
  real pooled_estimate = mu;
  real I_squared;
  real prediction_interval_width;

  // Compute log-likelihood for each observation (for LOO-CV)
  for (i in 1:J) {
    log_lik[i] = student_t_lpdf(y[i] | nu, theta[i], sigma[i]);
  }

  // Generate replicated data for posterior predictive checks
  for (i in 1:J) {
    y_rep[i] = student_t_rng(nu, theta[i], sigma[i]);
  }

  // Prediction for future study with median SE
  theta_new = normal_rng(mu, tau);
  y_new = student_t_rng(nu, theta_new, 11.0);  // median(sigma) = 11

  // I-squared statistic
  I_squared = 100 * tau^2 / (tau^2 + mean(sigma .* sigma));

  // 95% prediction interval width
  prediction_interval_width = 2 * 1.96 * sqrt(tau^2 + 11.0^2);
}
