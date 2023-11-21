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
}

transformed parameters {
  
}

model {
  for (i in 1:n_replicate) {
    for (j in 1:n_id) {
      target += gev_lpdf(y[i, j] | mu, sigma, xi);
    }
  }
}

generated quantities {
  real log_lik = 0;
  {
    
    for (i in 1:n_replicate) {
      for (j in 1:n_id) {
        log_lik += gev_lpdf(y_test[i, j] | mu, sigma, xi);
      }
    }
  }
}