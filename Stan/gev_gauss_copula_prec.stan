functions {
#include gev.stan
#include copula.stan
}

data {
  int<lower = 0> n_replicate;
  int<lower = 0> n_id;
  array[n_replicate, n_id] real y;
}

parameters {
  vector<lower = 0>[n_id] mu;
  vector<lower = 0>[n_id] sigma;
  vector<lower = 0, upper = 0.5>[n_id] xi;
  real<lower = -1, upper = 1> rho;
}

model {
  matrix[n_id, n_id] Omega = diag_matrix(rep_vector(1, n_id));
  Omega[2:(n_id - 1), 2:(n_id - 1)] = add_diag(Omega[2:(n_id - 1), 2:(n_id - 1)], rho^2);
  for (i in 1:n_id) {
    if (i > 1) {
      Omega[i, i - 1] = -rho;
    }
    if (i < n_id) {
      Omega[i, i + 1] = -rho;
    }
  }
  
  
  for (i in 1:n_replicate) {
    vector[n_id] U;
    for (j in 1:n_id) {
      U[j] = gev_cdf(y[i, j] | mu[j], sigma[j], xi[j]);
      target += gev_lpdf(y[i, j] | mu[j], sigma[j], xi[j]);
    }
    target += normal_copula_prec_lpdf(U | Omega);
  }
}
