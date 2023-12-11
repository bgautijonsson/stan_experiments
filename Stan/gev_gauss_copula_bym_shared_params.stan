functions {
  real gev_lpdf(real y, real mu, real sigma, real xi) {
    if (abs(xi) < 1e-10) {
      real z = (y - mu) / sigma;
      return -log(sigma) - z - exp(-z);
    } else {
      real z = 1 + xi * (y - mu) / sigma;
      if (z > 0) {
        return -log(sigma) - (1 + 1/xi) * log(z) - pow(z, -1/xi);
      } else {
        reject("Found incompatible GEV parameter values");
      }
    }
  }
  
  real normal_copula_bym_lpdf(vector U, vector re) {
    int N = rows(U);
    vector[N] Z = inv_Phi(U);
    vector[N] mu = rep_vector(0, N);
    matrix[N, N] I = diag_matrix(rep_vector(1, N));
    return normal_lpdf(Z | re, 1) - normal_lpdf(Z | 0, 1);
  }
  
  real icar_normal_lpdf(vector x, int N, array[] int node1, array[] int node2) {
    return  -0.5 * (dot_self((x[node1] - x[node2]))) +
    normal_lpdf(sum(x) | 0, 0.001 * N);
  }
  
  vector convolve_re_bym(vector spatial, vector random, real rho, real scaling_factor) {
    N = num_elements(spatial);
    vector[N] re = sqrt(rho / scaling_factor) * spatial + sqrt(1 - rho) * random;
    return re;
  }
}

data {
  int<lower = 0> n_replicate;
  int<lower = 0> n_id;
  array[n_replicate, n_id] real y;
  array[n_replicate, n_id] real y_test;
  
  int<lower = 0> N_neighbors;
  array[N_neighbors] int node1;
  array[N_neighbors] int node2;
  real<lower = 0> scaling_factor;
}

parameters {
  real<lower = 0> mu;
  real<lower = 0> sigma;
  real<lower = -0.5, upper = 1> xi;
  
  vector[n_id] copula_spatial;
  vector[n_id] copula_random;
  real<lower = 0, upper = 1> copula_rho;
}


model {
  vector[n_id] copula_re = convolve_re_bym(copula_spatial, copula_random, copula_rho, scaling_factor);
  
  copula_spatial ~ icar_normal_lpdf(N_neighbors, node1, node2);
  copula_random ~ std_normal();
  copula_rho ~ beta(2, 2);
  
  for (i in 1:n_replicate) {
    vector[n_id] U;
    for (j in 1:n_id) {
      U[j] = gev_cdf(y[i, j] | mu, sigma, xi);
      target += gev_lpdf(y[i, j] | mu, sigma, xi);
    }
    target += normal_copula_bym_lpdf(U | re);
  }
}
