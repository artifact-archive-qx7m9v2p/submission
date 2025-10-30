// Dirichlet Process Mixture of Beta-Binomials
// Non-parametric model that infers number of clusters from data
// Uses stick-breaking construction with truncation at K_max
// Does not pre-specify K (unlike FMM-3)

data {
  int<lower=1> J;              // number of groups (12)
  int<lower=1> K_max;          // truncation level for DP (e.g., 10)
  array[J] int<lower=0> n;     // trials per group
  array[J] int<lower=0> r;     // successes per group

  // Base distribution parameters
  real<lower=0> a0;            // Beta(a0, b0) base distribution
  real<lower=0> b0;
}

parameters {
  real<lower=0> alpha_dp;      // DP concentration parameter
  vector<lower=0, upper=1>[K_max] v;  // stick-breaking proportions
  vector<lower=0, upper=1>[K_max] p_star;  // cluster-specific success rates
}

transformed parameters {
  simplex[K_max] pi;           // cluster weights (from stick-breaking)

  // Stick-breaking construction
  {
    real stick_remaining = 1.0;
    for (k in 1:(K_max - 1)) {
      pi[k] = v[k] * stick_remaining;
      stick_remaining *= (1 - v[k]);
    }
    pi[K_max] = stick_remaining;
  }
}

model {
  // Priors
  alpha_dp ~ gamma(2, 2);      // concentration parameter
  v ~ beta(1, alpha_dp);       // stick-breaking weights
  p_star ~ beta(a0, b0);       // cluster-specific success rates

  // Mixture likelihood
  for (j in 1:J) {
    vector[K_max] log_contrib;
    for (k in 1:K_max) {
      log_contrib[k] = log(pi[k]) + binomial_lpmf(r[j] | n[j], p_star[k]);
    }
    target += log_sum_exp(log_contrib);
  }
}

generated quantities {
  vector[J] log_lik;
  array[J] int r_rep;
  matrix[J, K_max] cluster_prob;  // posterior cluster probabilities
  array[J] int z_hard;             // hard cluster assignments (MAP)

  // Diagnostics
  int K_effective;                 // number of active clusters
  vector[K_max] cluster_size;      // groups per cluster
  real alpha_dp_effective;         // effective concentration

  // Compute log-likelihood and cluster assignments
  for (j in 1:J) {
    vector[K_max] log_contrib;
    real log_lik_j;

    // Compute contributions from each cluster
    for (k in 1:K_max) {
      log_contrib[k] = log(pi[k]) + binomial_lpmf(r[j] | n[j], p_star[k]);
    }
    log_lik_j = log_sum_exp(log_contrib);

    // Log-likelihood
    log_lik[j] = log_lik_j;

    // Cluster probabilities
    cluster_prob[j] = softmax(log_contrib)';

    // Hard assignment (MAP)
    {
      int z_map = 1;
      real max_prob = cluster_prob[j, 1];
      for (k in 2:K_max) {
        if (cluster_prob[j, k] > max_prob) {
          max_prob = cluster_prob[j, k];
          z_map = k;
        }
      }
      z_hard[j] = z_map;
    }

    // Posterior predictive (from assigned cluster)
    {
      real u = uniform_rng(0, 1);
      real cum_prob = 0;
      int z_sample = K_max;
      for (k in 1:K_max) {
        cum_prob += cluster_prob[j, k];
        if (u <= cum_prob) {
          z_sample = k;
          break;
        }
      }
      r_rep[j] = binomial_rng(n[j], p_star[z_sample]);
    }
  }

  // Count effective clusters (weight > 0.05)
  K_effective = 0;
  for (k in 1:K_max) {
    if (pi[k] > 0.05) {
      K_effective += 1;
    }
  }

  // Compute cluster sizes
  cluster_size = rep_vector(0, K_max);
  for (j in 1:J) {
    cluster_size[z_hard[j]] += 1;
  }

  alpha_dp_effective = alpha_dp;
}
