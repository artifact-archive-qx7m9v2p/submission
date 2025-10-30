// Horseshoe Prior Model for Sparse Hierarchical Effects
// Automatic selection: most groups near mean, few truly deviate
// Local-global shrinkage with adaptive per-group shrinkage factors

data {
  int<lower=1> N;                  // Number of groups
  array[N] int<lower=0> n_trials;
  array[N] int<lower=0> r;
  real<lower=0> tau0;              // Expected sparsity level (scale parameter)
}

parameters {
  real mu;                         // Population mean
  vector[N] alpha_raw;             // Raw random effects

  // Horseshoe prior components
  vector<lower=0>[N] lambda;       // Local shrinkage (per group)
  real<lower=0> tau;               // Global shrinkage
}

transformed parameters {
  vector[N] alpha;
  vector[N] logit_p;

  // Horseshoe shrinkage
  alpha = alpha_raw .* lambda * tau;
  logit_p = mu + alpha;
}

model {
  // Hyperpriors
  mu ~ normal(-2.5, 1);
  tau ~ cauchy(0, tau0);           // Global shrinkage scale

  // Horseshoe prior
  lambda ~ cauchy(0, 1);           // Local shrinkage scales
  alpha_raw ~ normal(0, 1);        // Standardized effects

  // Likelihood
  r ~ binomial_logit(n_trials, logit_p);
}

generated quantities {
  // Posterior predictive checks
  array[N] int r_rep;
  vector[N] p = inv_logit(logit_p);

  // Effective shrinkage for each group
  vector[N] kappa;  // kappa_i ≈ 1 means no shrinkage, ≈ 0 means full shrinkage
  for (i in 1:N) {
    kappa[i] = (lambda[i]^2) / (lambda[i]^2 + tau^2);
  }

  // Count number of "active" groups (large effects)
  int n_active = 0;
  for (i in 1:N) {
    if (lambda[i] > 0.5) {
      n_active += 1;
    }
  }

  // Posterior predictive samples
  for (i in 1:N) {
    r_rep[i] = binomial_rng(n_trials[i], p[i]);
  }

  // Identify sparse groups (λ < 0.2 indicates heavy shrinkage)
  array[N] int is_sparse;
  for (i in 1:N) {
    is_sparse[i] = (lambda[i] < 0.2) ? 1 : 0;
  }

  // Overdispersion metric
  real mean_p = mean(p);
  real var_p = variance(p);
  real expected_var_p = mean_p * (1 - mean_p) / mean(to_vector(n_trials));
  real phi_posterior = var_p / expected_var_p;

  // Log-likelihood for LOO-CV
  vector[N] log_lik;
  for (i in 1:N) {
    log_lik[i] = binomial_logit_lpmf(r[i] | n_trials[i], logit_p[i]);
  }
}
