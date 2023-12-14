#' Crate a dataset with marginal GEV distributions and connecteb by a Gaussian copula with an AR(1) structure
#'
#' @return A tibble containing the data
#' @export
make_data <- function(params, n_replicate, rho) {
  box::use(
    dplyr[mutate, inner_join, join_by],
    evd[qgev],
    purrr[pmap_dbl]
  )
  
  make_AR_cor_matrix_1d(n_id = nrow(params), rho = rho) |> 
    sample_gaussian_variables(n_replicates = n_replicate) |> 
    tidy_mvgauss() |> 
    mutate(
      U = pnorm(Z)
    ) |> 
    inner_join(
      params,
      by = join_by(id)
    ) |> 
    mutate(
      y = pmap_dbl(
        list(U, mu, sigma, xi), 
        \(U, mu, sigma, xi) qgev(p = U, loc = mu, scale = sigma, shape = xi)
      )
    )
}

#' Convert a matrix of multivariate normal variables to a tibble
#'
#' @param mvnorm_matrix A matrix of multivariate normal variables with n_replicates rows and n_id columns
#'
#' @return
#' @export
tidy_mvgauss <- function(mvnorm_matrix) {
  box::use(
    dplyr[as_tibble, mutate, row_number],
    tidyr[pivot_longer]
  )
  colnames(mvnorm_matrix) <- seq_len(ncol(mvnorm_matrix))
  
  mvnorm_matrix |> 
    as_tibble() |> 
    mutate(
      replicate = row_number(),
      .before = `1`
    ) |> 
    pivot_longer(
      c(-replicate),
      names_to = "id", names_transform = as.numeric,
      values_to = "Z"
    )
  
}

#' Sample from a given multivariate normal distribution
#'
#' @param cor_matrix 
#'
#' @return
#' @export
#'
#' @examples
sample_gaussian_variables <- function(cor_matrix, n_replicates) {
  mvtnorm::rmvnorm(
    n = n_replicates,
    sigma = cor_matrix
  )
}

#' Make an n_id x n_id correlation matrix for a 1-dimensional AR(1) process
#'
#' @param n_id number of IDs
#' @param rho AR parameter
#'
#' @return A correlation matrix
#' @export
make_AR_cor_matrix_1d <- function(n_id, rho = 0.5) {
  P <- make_AR_prec_matrix_1d(n_id = n_id, rho = rho)
  
  P_cor <- solve(P)
  P_cor
}

#' Make an n_id x n_id precision matrix for a 1-dimensional AR(1) process
#'
#' @param n_id number of IDs
#' @param rho AR parameter
#'
#' @return A precision matrix
#' @export
make_AR_prec_matrix_1d <- function(n_id, rho = 0.5) {
  P <- matrix(
    0, 
    nrow = n_id,
    ncol = n_id
  )
  diag(P) <- 1
  for (i in seq(1, n_id - 1)) {
    P[i, i + 1] <- -rho
  }
  
  for (i in seq(2, n_id)) {
    P[i, i - 1] <- -rho
  }
  
  for (i in seq(2, n_id - 1)) {
    P[i, i] <- 1 + rho^2
  }
  
  P <- P / (1 - rho^2)
}


