library(testthat)
library(ProBot)
library(torch)

# --- Build a tiny trained-ish MDN for prediction tests ---
setup_mdn <- function() {
  set.seed(42)
  input_dim <- 3; output_dim <- 2; K <- 3
  hidden <- c(16, 16)

  mdl <- probotMakeMDN(input_dim, output_dim, K, hidden_dims = hidden, device = "cpu")()
  mdl$eval()

  inp <- matrix(rnorm(10 * input_dim), 10, input_dim)
  list(mdl = mdl, K = K, inp = inp, output_dim = output_dim)
}

test_that("probotPredictMDN returns a list with mu, log10_sigma, logits", {
  s <- setup_mdn()
  out <- probotPredictMDN(s$inp, s$mdl, s$K)

  expect_true("mu" %in% names(out))
  expect_true("log10_sigma" %in% names(out))
  expect_true("logits" %in% names(out))
})

test_that("probotPredictMDN output shapes are correct", {
  s <- setup_mdn()
  out <- probotPredictMDN(s$inp, s$mdl, s$K)

  n <- nrow(s$inp)
  expect_equal(dim(out$mu), c(n, s$K, s$output_dim))
  expect_equal(dim(out$log10_sigma), c(n, s$K, s$output_dim))
  expect_equal(dim(out$logits), c(n, s$K))
})

test_that("probotPredictMDN auto-detects device from model", {
  s <- setup_mdn()
  out <- probotPredictMDN(s$inp, s$mdl, s$K, device = NULL)
  expect_true("mu" %in% names(out))
})

test_that("probotSamplePostMDN returns correct dimensions", {
  s <- setup_mdn()
  out <- probotPredictMDN(s$inp, s$mdl, s$K)
  col_means <- colMeans(matrix(rnorm(100), 100, s$output_dim))
  col_sds <- apply(matrix(rnorm(100, 1, 0.5), 100, s$output_dim), 2, sd)

  n_samples <- 200
  samples <- probotSamplePostMDN(out, index = 1, n_samples = n_samples,
                                  col_means = col_means, col_sds = col_sds)

  expect_equal(dim(samples), c(n_samples, s$output_dim))
})

test_that("probotSamplePostMDN sets column names", {
  s <- setup_mdn()
  out <- probotPredictMDN(s$inp, s$mdl, s$K)
  col_means <- rep(0, s$output_dim)
  col_sds <- rep(1, s$output_dim)
  names <- c("a", "b")

  samples <- probotSamplePostMDN(out, index = 1, n_samples = 100,
                                  col_means = col_means, col_sds = col_sds,
                                  col_names = names)
  expect_equal(colnames(samples), names)
})

test_that("probotMarginalPostMDN returns list with post_mean and post_sd", {
  s <- setup_mdn()
  out <- probotPredictMDN(s$inp, s$mdl, s$K)
  col_means <- rep(0, s$output_dim)
  col_sds <- rep(1, s$output_dim)

  res <- probotMarginalPostMDN(out, col_means, col_sds)
  expect_true("post_mean" %in% names(res))
  expect_true("post_sd" %in% names(res))
})

test_that("probotMarginalPostMDN output dimensions are correct", {
  s <- setup_mdn()
  out <- probotPredictMDN(s$inp, s$mdl, s$K)
  col_means <- rep(0, s$output_dim)
  col_sds <- rep(1, s$output_dim)

  res <- probotMarginalPostMDN(out, col_means, col_sds)
  n <- nrow(s$inp)
  expect_equal(dim(res$post_mean), c(n, s$output_dim))
  expect_equal(dim(res$post_sd), c(n, s$output_dim))
})

test_that("probotMarginalPostMDN sets column names", {
  s <- setup_mdn()
  out <- probotPredictMDN(s$inp, s$mdl, s$K)
  col_means <- rep(0, s$output_dim)
  col_sds <- rep(1, s$output_dim)
  names <- c("x1", "x2")

  res <- probotMarginalPostMDN(out, col_means, col_sds, col_names = names)
  expect_equal(colnames(res$post_mean), names)
  expect_equal(colnames(res$post_sd), names)
})

# --- Point prediction tests ---
setup_point <- function() {
  set.seed(42)
  input_dim <- 3; output_dim <- 2
  mdl <- probotMakePoint(input_dim, output_dim, hidden_dims = c(16, 16), device = "cpu")()
  mdl$eval()
  inp <- matrix(rnorm(10 * input_dim), 10, input_dim)
  list(mdl = mdl, inp = inp, output_dim = output_dim)
}

test_that("probotPredictPoint returns a numeric matrix", {
  s <- setup_point()
  pred <- probotPredictPoint(s$inp, s$mdl)
  expect_true(is.matrix(pred))
  expect_true(is.numeric(pred))
  expect_equal(dim(pred), c(nrow(s$inp), s$output_dim))
})

test_that("probotPredictPointScaled returns un-scaled predictions", {
  s <- setup_point()
  col_means <- c(10, 20)
  col_sds <- c(2, 3)
  pred <- probotPredictPointScaled(s$inp, s$mdl, col_means, col_sds)
  expect_true(is.matrix(pred))
  expect_equal(dim(pred), c(nrow(s$inp), s$output_dim))
})
