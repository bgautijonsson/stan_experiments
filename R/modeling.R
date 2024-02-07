#' Fit a GEV model with a gaussian AR(1) copula to simulated data
#'
#' @export
fit_copula_model <- function(params, n_replicate, rho, output_dir) {
  here("R", "data.R") |> source()
  here("R", "two_step_gev_copula.R") |> source()

  d <- params |>
    make_data(n_replicate = n_replicate, rho = rho)


  n_id <- nrow(params)

  y <- d |>
    select(replicate, id, y) |>
    pivot_wider(names_from = id, values_from = y) |>
    select(-replicate) |>
    as.matrix()

  y_test <- d |>
    select(replicate, id, y) |>
    pivot_wider(names_from = id, values_from = y) |>
    select(-replicate) |>
    as.matrix()

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

  Z <- d |>
    select(replicate, id, Z) |>
    pivot_wider(names_from = id, values_from = Z) |>
    select(-replicate) |>
    as.matrix()

  get_rho <- function(rho) {
    scl <- sqrt(1 / (1 - rho^2))
    out <- 0
    for (i in seq_len(nrow(Z))) {
      x <- numeric(n_id)
      x[1:(n_id - 1)] <- scl * Z[i, 1:(n_id - 1)] - scl * rho * Z[i, 2:n_id]
      x[n_id] <- scl * sqrt(1 - rho^2) * Z[i, n_id]

      out <- out - n_id / 2 * log(2 * pi) - (n_id - 1) * (log(1 + rho) + log(1 - rho)) / 2 - sum(x^2) / 2
    }
    out <- out - sum(dnorm(Z, log = TRUE))
    out
  }

  inits$rho <- optimize(
    get_rho,
    interval = c(-1, 1),
    maximum = TRUE
  )$maximum

  inits$xi <- pmin(inits$xi, 1)



  inits_copula <- list(
    inits,
    inits,
    inits,
    inits
  )



  model_copula <- cmdstan_model(here("Stan", "gev_gauss_AR1_copula_prec_shared_params.stan"))

  results_copula <- model_copula$sample(
    data = stan_data,
    chains = 4,
    parallel_chains = 4,
    iter_warmup = 1000,
    iter_sampling = 1000,
    show_exceptions = F,
    init = inits_copula
  )

  # results_copula$draws() |>
  #   mcmc_hex(
  #     pars = c("mu", "rho")
  #   )
  #
  # results_copula$draws() |>
  #   mcmc_hex(
  #     pars = c("sigma", "rho")
  #   )

  out_copula <- results_copula$draws(variables = c("mu", "sigma", "xi", "rho", "log_lik")) |>
    as_draws_df() |>
    summarise_draws() |>
    mutate(
      model = "copula",
      time = results_copula$time()$total
    )




  inits_iid <- list(
    inits,
    inits,
    inits,
    inits
  )

  model_iid <- cmdstan_model(here("Stan", "gev_shared_params.stan"))

  results_iid <- model_iid$sample(
    data = stan_data,
    chains = 4,
    parallel_chains = 4,
    iter_warmup = 1000,
    iter_sampling = 1000,
    show_exceptions = FALSE,
    init = inits_iid
  )

  out_iid <- results_iid$draws(
    variables = c("mu", "sigma", "xi", "log_lik")
  ) |>
    as_draws_df() |>
    summarise_draws() |>
    mutate(
      model = "iid",
      time = results_iid$time()$total
    )



  out_twostep <- fit_twostep_model(
    n_replicate = n_replicate,
    n_id = n_id,
    params = params,
    d,
    y = y,
    y_test = y_test
  )


  out_copula |>
    bind_rows(
      out_iid
    ) |>
    bind_rows(
      out_twostep
    ) |>
    select(model, variable, posterior_mean = mean, rhat, time, Q_diag_mult) |>
    left_join(
      params |>
        distinct(mu, sigma, xi) |>
        pivot_longer(
          c(everything()),
          names_to = "variable",
          values_to = "true_value"
        ) |>
        add_row(
          variable = "rho", true_value = rho
        ),
      by = join_by(variable)
    ) |>
    mutate(
      n_replicate = n_replicate,
      n_id = nrow(params)
    ) |>
    write_parquet(
      here(output_dir, "part-0.parquet")
    )
}

require(glue)
require(dplyr)
require(here)
require(tidyr)
require(cmdstanr)
require(readr)
require(posterior)
require(arrow)
library(stringr)

replicate_vector <- c(5, 10, 20, 40, 80, 160)
replicate_weights <- rep(1, length(replicate_vector))
replicate_weights <- replicate_weights / sum(replicate_weights)

id_vector <- c(20, 40, 80)
id_weights <- rep(1, length(id_vector))
id_weights <- id_weights / sum(id_weights)

i <- here("results", "Stan", "copula_data", "shared_params", "posterior") |>
  list.files() |>
  parse_number() |>
  max()
i <- i + 1
# i <- 1
while (TRUE) {
  n_replicate <- sample(replicate_vector, size = 1, prob = replicate_weights)
  n_id <- sample(id_vector, size = 1, prob = id_weights)
  rho <- runif(n = 1, min = -1, max = 1) * rbinom(1, 1, 0.9)

  params <- crossing(
    mu = 10,
    sigma = 3.8,
    xi = 0.15,
    id = seq_len(n_id)
  )

  output_dir <- here("Results", "Stan", "copula_data", "shared_params", "posterior", glue("simulation={i}"))

  if (dir.exists(output_dir)) {
    here(output_dir, output_dir |> list.files()) |> file.remove()
  } else {
    dir.create(output_dir)
  }
  fit_copula_model(params = params, n_replicate = n_replicate, rho = rho, output_dir = output_dir)
  i <- i + 1
}
