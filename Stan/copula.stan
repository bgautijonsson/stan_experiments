real normal_copula_lpdf(vector U, matrix L) {
  int N = rows(U);
  vector[N] Z = inv_Phi(U);
  
  return multi_normal_cholesky_lpdf(Z | rep_vector(0, N), L) - std_normal_lpdf(Z);
}

real normal_copula_prec_lpdf(vector U, matrix L) {
  int N = rows(U);
  vector[N] Z = inv_Phi(U);
  vector[N] mu = rep_vector(0, N);
  matrix[N, N] I = diag_matrix(rep_vector(1, N));
  
  return multi_normal_prec_lpdf(Z | mu, L) - multi_normal_prec_lpdf(Z | mu, I);
}

vector lcdf_normal_copula_prec(vector U, matrix L) {
  int N = rows(U);
  matrix[N, N] L_chol = cholesky_decompose(L);
  vector[N] Z = L_chol *  inv_Phi(U);
  vector[N] out;
  for (i in 1:N) {
    out[i] = std_normal_lcdf(Z[i]);
  }
  
  return out;
}

real icar_copula_lpdf(vector U, int N_edges, array[] int node1, array[] int node2) {
  int N = rows(U);
  vector[N] Z = inv_Phi(U);
  return icar_normal_lpdf(Z | N_edges, node1, node2) - std_normal_lpdf(Z);
}

real icar_normal_lpdf(vector x, int N, array[] int node1, array[] int node2) {
  return  -0.5 * (dot_self((x[node1] - x[node2]))) +
    normal_lpdf(sum(x) | 0, 0.001 * N);
}

