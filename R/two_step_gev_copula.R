# require(glue)
# require(dplyr)
# require(here)
# require(tidyr)
# require(cmdstanr)
# require(readr)
# require(posterior)
# require(arrow)
# library(stringr)

# replicate_vector <- c(5, 10, 20, 40, 80)
# replicate_weights <- 1 / log(replicate_vector)
# replicate_weights <- replicate_weights / sum(replicate_weights)

# id_vector <- c(5, 10, 20, 40, 80)
# id_weights <- 1 / log(id_vector)
# id_weights <- id_weights / sum(id_weights)

# n_replicate <- sample(replicate_vector, size = 1, prob = replicate_weights)
# n_id <- sample(id_vector, size = 1, prob = id_weights)
# rho <- runif(n = 1, min = -1, max = 1) * rbinom(1, 1, 0.9)

# params <- crossing(
#   mu = 10,
#   sigma = 3.8,
#   xi = 0.15,
#   id = seq_len(n_id)
# )


# here("R", "data.R") |> source()

# d <- params |>
#   make_data(n_replicate = n_replicate, rho = rho)


# y <- d |>
#   select(replicate, id, y) |>
#   pivot_wider(names_from = id, values_from = y) |>
#   select(-replicate) |>
#   as.matrix()

# y_test <- d |>
#   select(replicate, id, y) |>
#   pivot_wider(names_from = id, values_from = y) |>
#   select(-replicate) |>
#   as.matrix()

fit_twostep_model <- function(
    n_replicate,
    n_id,
    params,
    d,
    y,
    y_test) {
  Q <- make_AR_prec_matrix_1d(n_id = n_id, rho = 0.5)

  from <- c()
  to <- c()

  for (i in 1:n_id) {
    for (j in i:n_id) {
      if (j > i) {
        if (Q[i, j] != 0) {
          from <- c(from, i)
          to <- c(to, j)
        }
      }
    }
  }

  time_start <- Sys.time()

  stan_data <- list(
    n_replicate = n_replicate,
    n_id = nrow(params),
    y = y,
    y_test = y_test
  )

  inits <- y |>
    as.numeric() |>
    evd::fgev() |>
    purrr::pluck("estimate") |>
    as.list()




  names(inits) <- c("mu", "sigma", "xi")

  inits$xi <- pmin(inits$xi, 1)


  inits_iid <- list(
    inits
  )

  model_iid <- cmdstan_model(here("Stan", "gev_shared_params.stan"))

  results_iid <- model_iid$optimize(
    data = stan_data,
    init = inits_iid
  )

  estimates <- results_iid$mle()


  mu <- estimates[1]
  sigma <- estimates[2]
  xi <- estimates[3]


  Z <- d |>
    select(replicate, id, y) |>
    mutate(
      U = evd::pgev(y, loc = mu, scale = sigma, shape = xi),
      Z = qnorm(U)
    ) |>
    select(replicate, id, Z) |>
    pivot_wider(names_from = id, values_from = Z) |>
    select(-replicate) |>
    as.matrix()

  stan_data_copula <- list(
    N_repetitions = n_replicate,
    N_nodes = n_id,
    N_edges = length(from),
    y = Z,
    from = from,
    to = to
  )

  model_copula <- cmdstan_model(here("Stan", "LM_precision.stan"))

  results_copula <- model_copula$optimize(
    data = stan_data_copula
  )

  estimates_copula <- results_copula$mle()

  Q_offdiags <- estimates_copula[str_detect(names(estimates_copula), "Q_offdiags")]
  tau <- estimates_copula[str_detect(names(estimates_copula), "tau")]


  Q_hat <- matrix(
    0,
    nrow = n_id,
    ncol = n_id
  )

  for (i in 1:n_id) {
    Q_hat[i, i] <- tau[i]
  }

  for (i in 1:length(from)) {
    Q_hat[from[i], to[i]] <- Q_offdiags[i]
    Q_hat[to[i], from[i]] <- Q_offdiags[i]
  }

  diag_pre <- sum(diag(Q_hat))

  while ("try-error" %in% class(try(chol(Q_hat)))) {
    Q_hat <- Q_hat + 0.01 * diag(n_id)
  }

  diag_post <- sum(diag(Q_hat))

  sigma_hat <- solve(Q_hat)

  D <- matrix(
    0,
    nrow = n_id,
    ncol = n_id
  )

  diag(D) <- sqrt(diag(sigma_hat))

  Q_fin <- D %*% Q_hat %*% D


  stan_data_gev_copula <- list(
    n_replicate = n_replicate,
    n_id = n_id,
    y = y,
    y_test = y_test,
    Q = Q_fin
  )

  model_gev_copula <- cmdstan_model(here("Stan", "gev_gauss_copula_prec_shared_params.stan"))


  results_gev_copula <- model_gev_copula$sample(
    data = stan_data_gev_copula,
    chains = 4,
    parallel_chains = 4,
    iter_warmup = 1000,
    iter_sampling = 1000,
    init = list(inits, inits, inits, inits)
  )

  time_stop <- Sys.time()

  total_time <- lubridate::seconds(time_stop - time_start) |>
    as.numeric()

  results_gev_copula$draws(
    variables = c("mu", "sigma", "xi", "log_lik")
  ) |>
    summarise_draws() |>
    mutate(
      model = "two step",
      time = total_time,
      Q_diag_mult = diag_post / diag_pre
    )

  # if (diag_post > diag_pre) {
  #   print("Matrix was not positive definite")
  #   sprintf(
  #     "Diagonal of Q was increased by a factor of %.2f",
  #     diag_post / diag_pre
  #   ) |>
  #     print()
  # }
}
