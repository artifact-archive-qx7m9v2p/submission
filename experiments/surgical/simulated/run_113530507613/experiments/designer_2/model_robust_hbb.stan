// Robust Beta-Binomial with Student-t Hierarchy
// Uses heavy-tailed Student-t distribution for group effects
// Beta-binomial likelihood accounts for overdispersion
// Robust to extreme outliers (Groups 4, 8)

data {
  int<lower=1> J;              // number of groups
  array[J] int<lower=0> n;     // trials per group
  array[J] int<lower=0> r;     // successes per group
}

parameters {
  real<lower=0, upper=1> mu_p;        // population mean success rate
  real<lower=0> tau_p;                // population SD
  real<lower=1> nu;                   // degrees of freedom (heaviness of tails)
  real<lower=0> kappa;                // concentration parameter (overdispersion)
  vector<lower=0, upper=1>[J] p;      // group-level success rates
}

transformed parameters {
  vector<lower=0>[J] alpha;
  vector<lower=0>[J] beta_param;

  for (j in 1:J) {
    alpha[j] = p[j] * kappa;
    beta_param[j] = (1 - p[j]) * kappa;
  }
}

model {
  // Priors
  mu_p ~ beta(2, 28);                 // centered on ~0.067
  tau_p ~ cauchy(0, 0.05);            // heavy-tailed prior for scale
  nu ~ gamma(2, 0.1);                 // allows nu = 1 (Cauchy) to > 30 (normal)
  kappa ~ gamma(2, 0.1);              // overdispersion parameter

  // Hierarchical model with Student-t (robust to outliers)
  // Note: Stan automatically truncates to (0,1) given parameter bounds
  for (j in 1:J) {
    p[j] ~ student_t(nu, mu_p, tau_p);
  }

  // Beta-binomial likelihood (accounts for overdispersion)
  for (j in 1:J) {
    r[j] ~ beta_binomial(n[j], alpha[j], beta_param[j]);
  }
}

generated quantities {
  vector[J] log_lik;
  array[J] int r_rep;

  // Diagnostics
  real nu_effective;           // effective degrees of freedom
  real kappa_effective;        // effective concentration
  int is_heavy_tailed;         // nu < 10 indicates heavy tails
  int is_overdispersed;        // kappa < 500 indicates overdispersion

  // Shrinkage factors (how much each group shrinks toward population mean)
  vector[J] shrinkage;

  for (j in 1:J) {
    // Log-likelihood
    log_lik[j] = beta_binomial_lpmf(r[j] | n[j], alpha[j], beta_param[j]);

    // Posterior predictive
    r_rep[j] = beta_binomial_rng(n[j], alpha[j], beta_param[j]);

    // Shrinkage factor (distance from raw rate to population mean)
    real raw_rate = r[j] * 1.0 / n[j];
    shrinkage[j] = abs(p[j] - raw_rate) / (abs(mu_p - raw_rate) + 1e-10);
  }

  // Diagnostics
  nu_effective = nu;
  kappa_effective = kappa;
  is_heavy_tailed = (nu < 10) ? 1 : 0;
  is_overdispersed = (kappa < 500) ? 1 : 0;
}
