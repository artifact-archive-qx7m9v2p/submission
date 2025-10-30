// Finite Mixture Model (2-Component Hierarchical)
// Designer #2 - Subpopulation Structure
// Purpose: Model potential low-rate vs high-rate subpopulations

data {
  int<lower=1> J;              // Number of groups (12)
  int<lower=0> n[J];           // Trials per group
  int<lower=0> r[J];           // Successes per group
}

parameters {
  real<lower=0, upper=1> pi;   // Mixture weight (proportion in component 2)
  ordered[2] mu;               // Component means (logit scale), mu[1] < mu[2]
  vector<lower=0>[2] tau;      // Component scales (logit scale)
  vector[J] theta;             // Group-level logit rates
}

model {
  // Priors
  pi ~ beta(2, 2);             // Symmetric, allows skewed mixtures
  mu[1] ~ normal(-3, 0.5);     // Low-rate component: ~4.7%
  mu[2] ~ normal(-2, 0.5);     // High-rate component: ~11.9%
  tau ~ cauchy(0, 0.5);        // Within-component variance

  // Mixture likelihood for theta
  for (j in 1:J) {
    vector[2] lp;
    lp[1] = log(1 - pi) + normal_lpdf(theta[j] | mu[1], tau[1]);
    lp[2] = log(pi) + normal_lpdf(theta[j] | mu[2], tau[2]);
    target += log_sum_exp(lp);
  }

  // Binomial likelihood
  r ~ binomial_logit(n, theta);
}

generated quantities {
  vector[J] p = inv_logit(theta);
  vector[J] log_lik;
  vector[J] r_rep;

  // Posterior component assignments (MAP)
  int<lower=1, upper=2> z[J];
  vector[J] prob_comp2;         // Probability of being in component 2

  for (j in 1:J) {
    log_lik[j] = binomial_logit_lpmf(r[j] | n[j], theta[j]);
    r_rep[j] = binomial_rng(n[j], p[j]);

    // Posterior component probabilities
    real lp1 = log(1 - pi) + normal_lpdf(theta[j] | mu[1], tau[1]);
    real lp2 = log(pi) + normal_lpdf(theta[j] | mu[2], tau[2]);

    prob_comp2[j] = exp(lp2 - log_sum_exp(lp1, lp2));
    z[j] = (prob_comp2[j] > 0.5) ? 2 : 1;
  }

  // Component-level summaries
  real p_comp1 = inv_logit(mu[1]);       // Mean rate in low component
  real p_comp2 = inv_logit(mu[2]);       // Mean rate in high component
  real separation = mu[2] - mu[1];       // Component separation

  // Overall population mean (mixture)
  real mu_pop_logit = (1 - pi) * mu[1] + pi * mu[2];
  real mu_pop = inv_logit(mu_pop_logit);

  // Count of groups in each component
  int<lower=0, upper=J> n_comp1 = J - sum(z == 2);
  int<lower=0, upper=J> n_comp2 = sum(z == 2);

  // Assignment uncertainty (entropy measure)
  real mean_entropy = mean(-prob_comp2 .* log(prob_comp2 + 1e-10)
                            - (1 - prob_comp2) .* log(1 - prob_comp2 + 1e-10));
}
