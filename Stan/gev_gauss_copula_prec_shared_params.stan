functions {
  #include gev.stan
  #include copula.stan
}

data {
  int<lower = 0> n_replicate;
  int<lower = 0> n_id;
  array[n_replicate, n_id] real y;
  array[n_replicate, n_id] real y_test;
}

parameters {
  real<lower = 0> mu;
  real<lower = 0> sigma;
  real<lower = 0, upper = 0.5> xi;
  real<lower = -1, upper = 1> rho;
}

transformed parameters {
  
}

model {
  real rho_scaler = pow(1 - rho^2, -1);
  real diag_inside = rho^2 * rho_scaler;
  real off_diag = -rho * rho_scaler;
  matrix[n_id, n_id] Omega = diag_matrix(rep_vector(rho_scaler, n_id));
  Omega[2:(n_id - 1), 2:(n_id - 1)] = add_diag(Omega[2:(n_id - 1), 2:(n_id - 1)], diag_inside);
  for (i in 1:n_id) {
    if (i > 1) {
      Omega[i, i - 1] = off_diag;
    }
    if (i < n_id) {
      Omega[i, i + 1] = off_diag;
    }
  }
  
  
  for (i in 1:n_replicate) {
    vector[n_id] U;
    for (j in 1:n_id) {
      U[j] = gev_cdf(y[i, j] | mu, sigma, xi);
      target += gev_lpdf(y[i, j] | mu, sigma, xi);
    }
    target += normal_copula_prec_lpdf(U | Omega);
  }
}

generated quantities {
  real log_lik = 0;
  {
    real rho_scaler = pow(1 - rho^2, -1);
    real diag_inside = rho^2 * rho_scaler;
    real off_diag = -rho * rho_scaler;
    matrix[n_id, n_id] Omega = diag_matrix(rep_vector(rho_scaler, n_id));
    Omega[2:(n_id - 1), 2:(n_id - 1)] = add_diag(Omega[2:(n_id - 1), 2:(n_id - 1)], diag_inside);
    for (i in 1:n_id) {
      if (i > 1) {
        Omega[i, i - 1] = off_diag;
      }
      if (i < n_id) {
        Omega[i, i + 1] = off_diag;
      }
    }
    
    for (i in 1:n_replicate) {
      vector[n_id] U;
      for (j in 1:n_id) {
        U[j] = gev_cdf(y_test[i, j] | mu, sigma, xi);
        log_lik += gev_lpdf(y_test[i, j] | mu, sigma, xi);
      }
      log_lik += normal_copula_prec_lpdf(U | Omega);
    }
  }
  
}