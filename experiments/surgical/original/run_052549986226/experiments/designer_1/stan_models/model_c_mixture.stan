// Model C: Two-Component Mixture Beta-Binomial
// Tests hypothesis of discrete subpopulations
// Designer 1 - Falsification test model

data {
  int<lower=1> N;
  array[N] int<lower=0> n_trials;
  array[N] int<lower=0> r_success;
}

parameters {
  real<lower=0, upper=1> pi;              // Mixing proportion
  ordered[2] mu;                           // Ordered means (μ_1 < μ_2 for identifiability)
  vector<lower=0>[2] kappa;                // Concentrations for each component
}

transformed parameters {
  vector[2] alpha;
  vector[2] beta_param;  // Renamed to avoid conflict with beta distribution

  for (k in 1:2) {
    alpha[k] = mu[k] * kappa[k];
    beta_param[k] = (1 - mu[k]) * kappa[k];
  }
}

model {
  // Priors
  pi ~ beta(2, 2);       // Slightly favors balanced mixture
  mu[1] ~ beta(3, 50);   // Low-rate component, mean ~ 0.057
  mu[2] ~ beta(5, 20);   // High-rate component, mean ~ 0.20
  kappa ~ gamma(2, 0.1); // Same concentration prior for both

  // Mixture likelihood
  for (i in 1:N) {
    target += log_mix(pi,
                      beta_binomial_lpmf(r_success[i] | n_trials[i], alpha[1], beta_param[1]),
                      beta_binomial_lpmf(r_success[i] | n_trials[i], alpha[2], beta_param[2]));
  }
}

generated quantities {
  // Posterior probability of each group belonging to each component
  array[N] vector[2] component_prob;
  array[N] int component_assignment;  // Most likely component (1 or 2)

  for (i in 1:N) {
    real lp1 = beta_binomial_lpmf(r_success[i] | n_trials[i], alpha[1], beta_param[1]);
    real lp2 = beta_binomial_lpmf(r_success[i] | n_trials[i], alpha[2], beta_param[2]);
    real denom = log_sum_exp(log(pi) + lp1, log(1-pi) + lp2);

    component_prob[i][1] = exp(log(pi) + lp1 - denom);
    component_prob[i][2] = 1 - component_prob[i][1];

    // Assign to component with highest probability
    component_assignment[i] = (component_prob[i][1] > 0.5) ? 1 : 2;
  }

  // Posterior predictive
  array[N] int r_rep;
  for (i in 1:N) {
    int component = bernoulli_rng(pi) + 1;  // Sample component (1 or 2)
    real p_i = beta_rng(alpha[component], beta_param[component]);
    r_rep[i] = binomial_rng(n_trials[i], p_i);
  }

  // Log-likelihood for LOO-CV (mixture log-likelihood)
  array[N] real log_lik;
  for (i in 1:N) {
    log_lik[i] = log_mix(pi,
                         beta_binomial_lpmf(r_success[i] | n_trials[i], alpha[1], beta_param[1]),
                         beta_binomial_lpmf(r_success[i] | n_trials[i], alpha[2], beta_param[2]));
  }

  // Component-specific parameters
  real mean_p_component1 = mu[1];
  real mean_p_component2 = mu[2];
  real var_p_component1 = (mu[1] * (1 - mu[1])) / (kappa[1] + 1);
  real var_p_component2 = (mu[2] * (1 - mu[2])) / (kappa[2] + 1);

  // Overall mean (weighted by mixing proportion)
  real mean_p_overall = pi * mu[1] + (1 - pi) * mu[2];

  // Separation metric: how well-separated are the components?
  real component_separation = (mu[2] - mu[1]) / sqrt(var_p_component1 + var_p_component2);
}
