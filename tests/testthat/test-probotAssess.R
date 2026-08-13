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
