// Negative Binomial State-Space Model
// Non-centered parameterization for efficient sampling

data {
  int<lower=1> N;                    // Number of time points
  array[N] int<lower=0> C;           // Observed counts
}

parameters {
  real delta;                        // Drift (growth rate)
  real<lower=0> sigma_eta;           // Innovation SD
  real<lower=0> phi;                 // Negative binomial dispersion
  real eta_1;                        // Initial state
  vector[N-1] eta_raw;               // Non-centered innovations
}

transformed parameters {
  vector[N] eta;                     // Latent states (log-scale)

  // Non-centered parameterization for stability
  eta[1] = eta_1;
  for (t in 2:N) {
    eta[t] = eta[t-1] + delta + sigma_eta * eta_raw[t-1];
  }
}

model {
  // Priors (ADJUSTED from Round 2)
  delta ~ normal(0.05, 0.02);
  sigma_eta ~ exponential(20);      // Mean = 0.05
  phi ~ exponential(0.05);          // Mean = 20
  eta_1 ~ normal(log(50), 1);

  // Non-centered innovations
  eta_raw ~ std_normal();

  // Observation likelihood
  for (t in 1:N) {
    C[t] ~ neg_binomial_2_log(eta[t], phi);
  }
}

generated quantities {
  vector[N] log_lik;                 // For LOO-CV
  array[N] int<lower=0> C_pred;      // Posterior predictive samples

  for (t in 1:N) {
    log_lik[t] = neg_binomial_2_log_lpmf(C[t] | eta[t], phi);
    C_pred[t] = neg_binomial_2_log_rng(eta[t], phi);
  }
}
