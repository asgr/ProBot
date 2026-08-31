library(testthat)
library(ProBot)
library(torch)

# --- Build a tiny MDN output for assessment tests ---
setup_assess <- function(n_test = 5, output_dim = 2, K = 3, n_samples = 100) {
  set.seed(42)
  input_dim <- 3

  mdl <- probotMakeMDN(input_dim, output_dim, K, hidden_dims = c(8, 8), device = "cpu")()
  mdl$eval()

  inp <- matrix(rnorm(n_test * input_dim), n_test, input_dim)
  out <- probotPredictMDN(inp, mdl, K)

  col_means <- rep(0, output_dim)
  col_sds <- rep(1, output_dim)
  col_names <- paste0("p", 1:output_dim)

  # True params: just use random values as a reference
  params <- matrix(rnorm(n_test * output_dim), n_test, output_dim)

  list(out = out, params = params, col_means = col_means,
       col_sds = col_sds, col_names = col_names, n_samples = n_samples,
       n_test = n_test, output_dim = output_dim)
}

test_that("probotPIT returns values in [0, 1]", {
  s <- setup_assess(n_test = 3, n_samples = 200)
  pit <- probotPIT(s$out, s$params, n_test = s$n_test, n_samples = s$n_samples,
                    col_means = s$col_means, col_sds = s$col_sds,
                    col_names = s$col_names, verbose = FALSE)

  expect_true(all(pit >= 0 & pit <= 1))
  expect_equal(dim(pit), c(s$n_test, s$output_dim))
})

test_that("probotPIT sets column names", {
  s <- setup_assess(n_test = 2, n_samples = 100)
  pit <- probotPIT(s$out, s$params, n_test = s$n_test, n_samples = s$n_samples,
                    col_means = s$col_means, col_sds = s$col_sds,
                    col_names = s$col_names, verbose = FALSE)

  expect_equal(colnames(pit), s$col_names)
})

test_that("probotPIT auto-sets n_test from params", {
  s <- setup_assess(n_test = 3, n_samples = 100)
  pit <- probotPIT(s$out, s$params, n_test = NULL, n_samples = s$n_samples,
                    col_means = s$col_means, col_sds = s$col_sds,
                    verbose = FALSE)

  expect_equal(nrow(pit), s$n_test)
})

test_that("probotTARP returns values in [0, 1]", {
  s <- setup_assess(n_test = 3, n_samples = 200)
  tarp <- probotTARP(s$out, s$params, n_test = s$n_test, n_samples = s$n_samples,
                      col_means = s$col_means, col_sds = s$col_sds,
                      col_names = s$col_names, verbose = FALSE)

  expect_true(all(tarp >= 0 & tarp <= 1))
  expect_equal(length(tarp), s$n_test)
})

test_that("probotTARP caps n_test to nrow(params)", {
  s <- setup_assess(n_test = 3, n_samples = 100)
  tarp <- probotTARP(s$out, s$params, n_test = 100, n_samples = s$n_samples,
                      col_means = s$col_means, col_sds = s$col_sds,
                      verbose = FALSE)

  expect_equal(length(tarp), s$n_test)
})

test_that("probotCRPS returns non-negative values", {
  s <- setup_assess(n_test = 3, n_samples = 200)
  crps <- probotCRPS(s$out, s$params, n_test = s$n_test, n_samples = s$n_samples,
                      col_means = s$col_means, col_sds = s$col_sds,
                      col_names = s$col_names, verbose = FALSE)

  expect_true(all(crps >= 0))
  expect_equal(dim(crps), c(s$n_test, s$output_dim))
})

test_that("probotCRPS sets column names", {
  s <- setup_assess(n_test = 2, n_samples = 100)
  crps <- probotCRPS(s$out, s$params, n_test = s$n_test, n_samples = s$n_samples,
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
  o_det <- list(
    mu = torch_tensor(array(10, c(3, 1, 1))),
    log10_sigma = torch_tensor(array(0, c(3, 1, 1))),
    logits = torch_tensor(matrix(0, 3, 1))
  )
  truth <- c(10, 10.3, 15)
  crps_det <- probotCRPS(o_det, matrix(truth, 3, 1), n_samples = 1e4,
                         col_means = 0, col_sds = 1, verbose = FALSE)

  Eabs <- function(t, mu) 2 * dnorm(t - mu) + (t - mu) * (2 * pnorm(t - mu) - 1)
  for (k in 1:3) {
    expect_equal(crps_det[k, 1], Eabs(truth[k], 10) - 1 / sqrt(pi),
                 tolerance = 0.05)
  }
})
