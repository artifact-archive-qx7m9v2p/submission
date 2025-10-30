// Model 2: Hierarchical Logistic Regression with Quadratic Group Effect
// Tests whether sequential group_id has non-linear (quadratic) structure

data {
  int<lower=1> J;                    // number of groups
  array[J] int<lower=0> n;           // number of trials per group
  array[J] int<lower=0> r;           // number of successes per group
  vector[J] group_scaled;            // group_id scaled to [-1, 1]
  vector[J] group_scaled_sq;         // group_scaled^2
}

parameters {
  real beta_0;                       // intercept
  real beta_1;                       // linear term (trend)
  real beta_2;                       // quadratic term (curvature)
  real<lower=0> tau;                 // residual between-group SD
  vector[J] alpha_raw;               // non-centered parameterization
}

transformed parameters {
  vector[J] alpha;                   // group effects (logit scale)
  vector[J] mu;                      // regression predictor
  vector[J] p;                       // success probabilities

  // Quadratic regression structure
  mu = beta_0 + beta_1 * group_scaled + beta_2 * group_scaled_sq;

  // Non-centered parameterization
  alpha = mu + tau * alpha_raw;

  // Inverse logit transformation
  p = inv_logit(alpha);
}

model {
  // Priors
  beta_0 ~ normal(-2.6, 1.0);        // Intercept
  beta_1 ~ normal(0, 0.5);           // Linear term (symmetric prior)
  beta_2 ~ normal(0, 0.5);           // Quadratic term (symmetric prior)
  tau ~ normal(0, 0.5);              // Half-normal for scale
  alpha_raw ~ std_normal();          // Standard normal for NC parameterization

  // Likelihood
  r ~ binomial(n, p);
}

generated quantities {
  // Log-likelihood for LOO-CV
  vector[J] log_lik;

  // Posterior predictive samples
  array[J] int r_rep;

  // R-squared (variance explained by polynomial)
  real<lower=0, upper=1> R2;

  // Peak/trough location (if quadratic is significant)
  real peak_location;                // In scaled units [-1, 1]

  // Test for significant curvature
  int<lower=0, upper=1> is_curved;   // 1 if |beta_2| > 0.1

  // Compute log-likelihood and posterior predictive
  for (j in 1:J) {
    log_lik[j] = binomial_lpmf(r[j] | n[j], p[j]);
    r_rep[j] = binomial_rng(n[j], p[j]);
  }

  // R-squared
  {
    real var_fitted = variance(mu);
    real var_total = var_fitted + tau^2;
    R2 = var_fitted / var_total;
  }

  // Peak location of parabola (only meaningful if beta_2 != 0)
  peak_location = (fabs(beta_2) > 0.001) ? -beta_1 / (2.0 * beta_2) : 0.0;

  // Indicator for significant curvature
  is_curved = (fabs(beta_2) > 0.1) ? 1 : 0;
}
