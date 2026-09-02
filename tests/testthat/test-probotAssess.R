library(testthat)
library(ProBot)
library(torch)

# --- Build a tiny MDN for assessment tests ---
setup_assess <- function(n_test = 5, output_dim = 2, K = 3, n_samples = 100) {
  set.seed(42)
  input_dim <- 3

  mdl <- probotMakeMDN(input_dim, output_dim, K, hidden_dims = c(8, 8), device = "cpu")()
  mdl$eval()

  inp <- matrix(rnorm(n_test * input_dim), n_test, input_dim)

  col_means <- rep(0, output_dim)
  col_sds <- rep(1, output_dim)
  col_names <- paste0("p", 1:output_dim)

  # True params: just use random values as a reference
  params <- matrix(rnorm(n_test * output_dim), n_test, output_dim)

  list(inp = inp, mdl = mdl, K = K, params = params,
       col_means = col_means, col_sds = col_sds, col_names = col_names,
       n_samples = n_samples, n_test = n_test, output_dim = output_dim)
}

test_that("probotPIT returns values in [0, 1]", {
  s <- setup_assess(n_test = 3, n_samples = 200)
  pit <- probotPIT(s$inp, s$mdl, s$K, s$params, n_test = s$n_test,
                    n_samples = s$n_samples,
                    col_means = s$col_means, col_sds = s$col_sds,
                    col_names = s$col_names, verbose = FALSE)

  expect_true(all(pit >= 0 & pit <= 1))
  expect_equal(dim(pit), c(s$n_test, s$output_dim))
})

test_that("probotPIT sets column names", {
  s <- setup_assess(n_test = 2, n_samples = 100)
  pit <- probotPIT(s$inp, s$mdl, s$K, s$params, n_test = s$n_test,
                    n_samples = s$n_samples,
                    col_means = s$col_means, col_sds = s$col_sds,
                    col_names = s$col_names, verbose = FALSE)

  expect_equal(colnames(pit), s$col_names)
})

test_that("probotPIT auto-sets n_test from params", {
  s <- setup_assess(n_test = 3, n_samples = 100)
  pit <- probotPIT(s$inp, s$mdl, s$K, s$params, n_test = NULL,
                    n_samples = s$n_samples,
                    col_means = s$col_means, col_sds = s$col_sds,
                    verbose = FALSE)

  expect_equal(nrow(pit), s$n_test)
})

test_that("probotTARP returns values in [0, 1]", {
  s <- setup_assess(n_test = 3, n_samples = 200)
  tarp <- probotTARP(s$inp, s$mdl, s$K, s$params, n_test = s$n_test,
                      n_samples = s$n_samples,
                      col_means = s$col_means, col_sds = s$col_sds,
                      col_names = s$col_names, verbose = FALSE)

  expect_true(all(tarp >= 0 & tarp <= 1))
  expect_equal(length(tarp), s$n_test)
})

test_that("probotTARP caps n_test to nrow(params)", {
  s <- setup_assess(n_test = 3, n_samples = 100)
  tarp <- probotTARP(s$inp, s$mdl, s$K, s$params, n_test = 100,
                      n_samples = s$n_samples,
                      col_means = s$col_means, col_sds = s$col_sds,
                      verbose = FALSE)

  expect_equal(length(tarp), s$n_test)
})

test_that("probotTARP defaults to all rows (n_test = NULL)", {
  # Regression: the old default n_test = 1e4 silently truncated, so a 200-row
  # params matrix still returned 1e4 TARP values. The NULL default must use
  # every row.
  s <- setup_assess(n_test = 200, n_samples = 20)
  tarp <- probotTARP(s$inp, s$mdl, s$K, s$params, n_samples = s$n_samples,
                      col_means = s$col_means, col_sds = s$col_sds,
                      verbose = FALSE)

  expect_equal(length(tarp), 200)
})

test_that("probotPIT/probotTARP/probotCRPS handle n_test = 0 without crashing", {
  # 1:0 evaluates to c(1, 0) and the loop iterated i = 0 -> subscript error.
  s <- setup_assess(n_test = 3, n_samples = 100)

  pit <- probotPIT(s$inp, s$mdl, s$K, s$params, n_test = 0,
                    n_samples = s$n_samples,
                    col_means = s$col_means, col_sds = s$col_sds,
                    verbose = FALSE)
  expect_equal(dim(pit), c(0L, s$output_dim))

  tarp <- probotTARP(s$inp, s$mdl, s$K, s$params, n_test = 0,
                      n_samples = s$n_samples,
                      col_means = s$col_means, col_sds = s$col_sds,
                      verbose = FALSE)
  expect_equal(length(tarp), 0)

  crps <- probotCRPS(s$inp, s$mdl, s$K, s$params, n_test = 0,
                      n_samples = s$n_samples,
                      col_means = s$col_means, col_sds = s$col_sds,
                      verbose = FALSE)
  expect_equal(dim(crps), c(0L, s$output_dim))
})

test_that("assess functions accept a single observation as a vector", {
  # as.matrix() orients a bare vector as a column, which silently transposes
  # the conditioning input; a vector must be read as one row.
  s <- setup_assess(n_test = 1, n_samples = 200)

  pit_v <- probotPIT(s$inp[1, ], s$mdl, s$K, s$params[1, , drop = FALSE],
                      n_samples = s$n_samples, col_means = s$col_means,
                      col_sds = s$col_sds, verbose = FALSE)
  expect_equal(dim(pit_v), c(1L, s$output_dim))

  crps_v <- probotCRPS(s$inp[1, ], s$mdl, s$K, s$params[1, , drop = FALSE],
                        n_samples = s$n_samples, col_means = s$col_means,
                        col_sds = s$col_sds, verbose = FALSE)
  expect_equal(dim(crps_v), c(1L, s$output_dim))

  tarp_v <- probotTARP(s$inp[1, ], s$mdl, s$K, s$params[1, , drop = FALSE],
                        n_samples = s$n_samples, col_means = s$col_means,
                        col_sds = s$col_sds, verbose = FALSE)
  expect_length(tarp_v, 1L)
  expect_true(tarp_v >= 0 & tarp_v <= 1)
})

test_that("assess functions reject input with too few rows", {
  s <- setup_assess(n_test = 3, n_samples = 50)
  expect_error(
    probotPIT(s$inp[1:2, ], s$mdl, s$K, s$params, n_samples = s$n_samples,
              col_means = s$col_means, col_sds = s$col_sds, verbose = FALSE),
    "at least n_test rows"
  )
})

test_that("probotCRPS returns non-negative values", {
  s <- setup_assess(n_test = 3, n_samples = 200)
  crps <- probotCRPS(s$inp, s$mdl, s$K, s$params, n_test = s$n_test,
                      n_samples = s$n_samples,
                      col_means = s$col_means, col_sds = s$col_sds,
                      col_names = s$col_names, verbose = FALSE)

  expect_true(all(crps >= 0))
  expect_equal(dim(crps), c(s$n_test, s$output_dim))
})

test_that("probotCRPS sets column names", {
  s <- setup_assess(n_test = 2, n_samples = 100)
  crps <- probotCRPS(s$inp, s$mdl, s$K, s$params, n_test = s$n_test,
                      n_samples = s$n_samples,
                      col_means = s$col_means, col_sds = s$col_sds,
                      col_names = s$col_names, verbose = FALSE)

  expect_equal(colnames(crps), s$col_names)
})

# Brute-force CRPS from the sample estimator definition:
#   mean|z - y| - (1/(2 S^2)) * sum_i sum_j |z_i - z_j|
crps_brute_force <- function(samples, y) {
  S <- length(samples)
  mean(abs(samples - y)) -
    sum(outer(samples, samples, function(a, b) abs(a - b))) / (2 * S^2)
}

test_that("probotCRPS matches the brute-force double-sum estimator", {
  # Fixed samples cover the weight formula independently of the (random)
  # sampler: the sorted-sample penalty must equal the brute-force double sum.
  set.seed(17)
  S <- 300
  z <- rnorm(S)
  w <- ProBot:::.probotCRPSWeights(S)

  expect_equal(
    sum(w * sort(z)),
    sum(outer(z, z, function(a, b) abs(a - b))) / (2 * S^2),
    tolerance = 1e-12
  )

  # Full estimator: mean absolute error minus the spread penalty.
  y <- 0.3
  expect_equal(
    ProBot:::.probotCRPSSample(z, y, w),
    crps_brute_force(z, y),
    tolerance = 1e-12
  )

  # Spot-check at other sample sizes (weights scale as 1/S^2).
  for (S in c(5L, 100L, 1000L)) {
    z <- rnorm(S)
    w <- ProBot:::.probotCRPSWeights(S)
    expect_equal(
      ProBot:::.probotCRPSSample(z, 0, w),
      crps_brute_force(z, 0),
      tolerance = 1e-12
    )
  }

  # End-to-end with a deterministic reference: a single-component unit-Gaussian
  # MDN (mu = 10, log10_sigma = 0) with col_means = 0, col_sds = 1 samples
  # from N(10, 1) — no un-scaling. It must reproduce the analytic CRPS of
  # the normal distribution,
  #   CRPS = E|Z - t| - 1/sqrt(pi),
  #   E|Z - t| = 2*phi(t-mu) + (t-mu) * (2*Phi(t-mu) - 1)  (sigma = 1)
  # with only Monte Carlo noise (a few parts in 1e-3 at S = 1e4). This
  # exercises the weight formula through the real probotCRPS() +
  # probotSamplePostMDN() path; the previous halved weights would be off by
  # 1/(2*sqrt(pi)) ~ 0.28 (>> MC noise).
  # Stub module: constant (mu, log10_sigma, logit) head, whatever the input.
  o_det <- nn_module(
    initialize = function() {
      self$out <- torch_tensor(cbind(10, 0, 0))
    },
    forward = function(x) self$out$expand(c(x$size(1), -1))
  )()

  truth <- c(10, 10.3, 15)
  crps_det <- probotCRPS(matrix(rnorm(3), 3, 1), o_det, mdn_components = 1,
                         params = matrix(truth, 3, 1), n_samples = 1e4,
                         col_means = 0, col_sds = 1, verbose = FALSE)

  Eabs <- function(t, mu) 2 * dnorm(t - mu) + (t - mu) * (2 * pnorm(t - mu) - 1)
  for (k in 1:3) {
    expect_equal(crps_det[k, 1], Eabs(truth[k], 10) - 1 / sqrt(pi),
                 tolerance = 0.05)
  }
})
