// Beta-Binomial Hierarchical Model
// Designer #2 - Direct Probability Scale Parameterization
// Purpose: Avoid logit transformation, use natural conjugate structure

data {
  int<lower=1> J;              // Number of groups (12)
  int<lower=0> n[J];           // Trials per group
  int<lower=0> r[J];           // Successes per group
}

parameters {
  real<lower=0> alpha;         // Beta shape parameter 1
  real<lower=0> beta;          // Beta shape parameter 2
  vector<lower=0, upper=1>[J] theta;  // Group-level success probabilities
}

model {
  // Priors on Beta shape parameters
  alpha ~ gamma(2, 0.2);       // Mean = 10, SD = 7.07
  beta ~ gamma(2, 0.02);       // Mean = 100, SD = 70.7
                                // Induces E[theta] ~ 0.09 (9%), close to observed 7%

  // Hierarchical structure (natural conjugate)
  theta ~ beta(alpha, beta);

  // Likelihood
  r ~ binomial(n, theta);
}

generated quantities {
  // Population-level summaries
  real mu_pop = alpha / (alpha + beta);                    // Population mean
  real sigma_pop = sqrt(alpha * beta / ((alpha + beta)^2 * (alpha + beta + 1)));  // Population SD
  real kappa = alpha + beta;                               // Concentration parameter

  // Posterior predictive and diagnostics
  vector[J] log_lik;
  vector[J] r_rep;

  for (j in 1:J) {
    log_lik[j] = binomial_lpmf(r[j] | n[j], theta[j]);
    r_rep[j] = binomial_rng(n[j], theta[j]);
  }

  // Overdispersion metric
  real expected_var = mu_pop * (1 - mu_pop);
  real actual_var = variance(theta);
  real phi = actual_var / expected_var;                    // Should be > 1

  // Alternative parameterization (for interpretation)
  real mu_logit = logit(mu_pop);                          // Mean on logit scale
  real tau_logit = sd(logit(theta));                       // SD on logit scale

  // Shrinkage diagnostics
  vector[J] shrinkage;
  for (j in 1:J) {
    real mle_j = r[j] * 1.0 / n[j];
    shrinkage[j] = fabs(theta[j] - mle_j) / fabs(mu_pop - mle_j + 1e-6);
  }
}
