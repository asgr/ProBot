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

test_that("probotSamplePostMDN soft-clamps log10_sigma to [-5, 5]", {
  # A single-component MDN with log10_sigma = 8 would sample from N(0, 1e8)
  # without clamping. Training/loss clamp log10_sigma to [-5, 5], so the
  # sampler must cap sigma at 1e5 and the sample SD should land near 1e5
  # (a hundred-fold separation from the unclamped 1e8).
  set.seed(7)
  o <- list(
    mu = torch_tensor(array(0, c(1, 1, 1))),
    log10_sigma = torch_tensor(array(8, c(1, 1, 1))),
    logits = torch_tensor(matrix(0, 1, 1))
  )
  n_samples <- 1e5
  samples <- probotSamplePostMDN(o, index = 1, n_samples = n_samples,
                                  col_means = 0, col_sds = 1)
  # The clamp is what this test is for: capped sigma = 1e5, not 1e8.
  expect_true(sd(samples) > 0.8e5 & sd(samples) < 1.2e5)
  # Sanity: sampling around 0, within a few Monte Carlo standard errors
  # (SE of the mean = sigma / sqrt(n) ~ 316 here).
  se <- 1e5 / sqrt(n_samples)
  expect_lt(abs(mean(samples)), 6 * se)
})

test_that("probotMarginalPostMDN soft-clamps log10_sigma to [-5, 5]", {
  # log10_sigma = 8 (sigma = 1e8) must be clamped to sigma = 1e5; the
  # analytic marginal SD is exactly that (weights = 1, mu = 0). The lower
  # bound (-5) is checked the same way.
  o_hi <- list(
    mu = torch_tensor(array(0, c(1, 1, 1))),
    log10_sigma = torch_tensor(array(8, c(1, 1, 1))),
    logits = torch_tensor(matrix(0, 1, 1))
  )
  res <- probotMarginalPostMDN(o_hi, col_means = 0, col_sds = 1)
  expect_equal(res$post_sd[1, 1], 1e5)

  o_lo <- list(
    mu = torch_tensor(array(0, c(1, 1, 1))),
    log10_sigma = torch_tensor(array(-8, c(1, 1, 1))),
    logits = torch_tensor(matrix(0, 1, 1))
  )
  res_lo <- probotMarginalPostMDN(o_lo, col_means = 0, col_sds = 1)
  expect_equal(res_lo$post_sd[1, 1], 1e-5)

  # An in-range value must pass through unchanged (sigma = 10^0.5).
  o_mid <- list(
    mu = torch_tensor(array(0, c(1, 1, 1))),
    log10_sigma = torch_tensor(array(0.5, c(1, 1, 1))),
    logits = torch_tensor(matrix(0, 1, 1))
  )
  res_mid <- probotMarginalPostMDN(o_mid, col_means = 0, col_sds = 1)
  expect_equal(res_mid$post_sd[1, 1], 10^0.5, tolerance = 1e-6)
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
