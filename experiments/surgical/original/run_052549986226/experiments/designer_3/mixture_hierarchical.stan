// Finite Mixture Model for Binomial Data with Latent Subgroups
// Two components: normal groups vs outlier groups
// Marginalizes over latent cluster assignments

data {
  int<lower=1> N;                  // Number of groups
  int<lower=2> K;                  // Number of mixture components (K=2)
  array[N] int<lower=0> n_trials;
  array[N] int<lower=0> r;
}

parameters {
  // Cluster-specific parameters (ordered for identifiability)
  ordered[K] mu;                   // Cluster means (mu[1] < mu[2])
  vector<lower=0>[K] sigma;        // Cluster-specific SDs

  // Mixing proportion
  simplex[K] pi;                   // Sums to 1

  // Group-specific parameters
  vector[N] alpha_raw;             // Non-centered random effects
}

model {
  // Priors on cluster parameters
  mu[1] ~ normal(-3.0, 0.5);       // Low-rate cluster
  mu[2] ~ normal(-2.0, 0.5);       // High-rate cluster

  sigma[1] ~ normal(0, 0.3);       // Tight cluster
  sigma[2] ~ normal(0, 0.5);       // Looser cluster

  pi ~ dirichlet(rep_vector(1.0, K));  // Uniform mixing

  alpha_raw ~ normal(0, 1);        // Standardized effects

  // Marginalize over latent cluster assignments
  for (i in 1:N) {
    vector[K] log_pi_k;

    for (k in 1:K) {
      // Log probability of group i in cluster k
      log_pi_k[k] = log(pi[k]) +
                    normal_lpdf(alpha_raw[i] | 0, 1) +
                    binomial_logit_lpmf(r[i] | n_trials[i],
                                        mu[k] + sigma[k] * alpha_raw[i]);
    }

    // Marginalize over clusters (log-sum-exp trick)
    target += log_sum_exp(log_pi_k);
  }
}

generated quantities {
  // Posterior predictive checks
  array[N] int r_rep;
  vector[N] p;

  // Posterior cluster assignments
  array[N] simplex[K] prob_cluster;  // P(z_i = k | data)
  array[N] int<lower=1, upper=K> cluster_assignment;

  // Cluster size counts
  int n_cluster1 = 0;
  int n_cluster2 = 0;

  // Log-likelihood for LOO-CV
  vector[N] log_lik;

  for (i in 1:N) {
    vector[K] log_pi_k;

    // Compute posterior cluster probabilities
    for (k in 1:K) {
      log_pi_k[k] = log(pi[k]) +
                    normal_lpdf(alpha_raw[i] | 0, 1) +
                    binomial_logit_lpmf(r[i] | n_trials[i],
                                        mu[k] + sigma[k] * alpha_raw[i]);
    }

    prob_cluster[i] = softmax(log_pi_k);

    // Hard assignment (most probable cluster)
    cluster_assignment[i] = (prob_cluster[i][2] > 0.5) ? 2 : 1;

    // Count cluster sizes
    if (cluster_assignment[i] == 1) {
      n_cluster1 += 1;
    } else {
      n_cluster2 += 1;
    }

    // Posterior predictive: sample from mixture
    int z_sim = categorical_rng(pi);
    real logit_p_sim = mu[z_sim] + sigma[z_sim] * alpha_raw[i];
    p[i] = inv_logit(logit_p_sim);
    r_rep[i] = binomial_rng(n_trials[i], p[i]);

    // Log-likelihood (marginal over clusters)
    log_lik[i] = log_sum_exp(log_pi_k);
  }

  // Cluster separation metric
  real cluster_separation = mu[2] - mu[1];

  // Overdispersion
  real mean_p = mean(p);
  real var_p = variance(p);
  real expected_var_p = mean_p * (1 - mean_p) / mean(to_vector(n_trials));
  real phi_posterior = var_p / expected_var_p;
}
