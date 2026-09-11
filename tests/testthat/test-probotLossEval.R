library(testthat)
library(ProBot)
library(torch)

# Shared fixtures: a small regression problem all three model types can score.
set.seed(717)
le_n <- 200L
le_D <- 3L
le_C <- 4L
le_K <- 3L
le_x <- matrix(rnorm(le_n * le_C), le_n, le_C)
le_y <- matrix(rnorm(le_n * le_D), le_n, le_D)

le_mdn <- probotMakeMDN(input_dim = le_C, output_dim = le_D,
                        mdn_components = le_K, hidden_dims = c(16, 16))()
le_pt <- probotMakePoint(input_dim = le_C, output_dim = le_D,
                         hidden_dims = c(16, 16))()
le_fl <- probotMakeFlow(input_dim = le_C, output_dim = le_D, n_layers = 2,
                        hidden_dim = 16, style = "realnvp")()

test_that("probotLossEval scores all three model types", {
  for (spec in list(mdn = le_mdn, point = le_pt, flow = le_fl)) {
    res <- probotLossEval(le_x, le_y, spec, mdn_components = le_K)
    expect_type(res, "list")
    expect_named(res, c("loss", "n", "batches", "model_type", "loss_name"))
    expect_length(res$loss, 1L)
    expect_true(is.finite(res$loss))
    expect_identical(res$n, le_n)
  }
})

test_that("probotLossEval dispatches the model type from the object", {
  expect_identical(probotLossEval(le_x, le_y, le_mdn, le_K)$model_type, "mdn")
  expect_identical(probotLossEval(le_x, le_y, le_pt)$model_type, "point")
  expect_identical(probotLossEval(le_x, le_y, le_fl)$model_type, "flow")
})

test_that("probotLossEval names the loss it used", {
  expect_identical(probotLossEval(le_x, le_y, le_mdn, le_K)$loss_name,
                   "probotLossMDN")
  expect_identical(probotLossEval(le_x, le_y, le_pt)$loss_name, "nnf_mse_loss")
  expect_identical(probotLossEval(le_x, le_y, le_fl)$loss_name, "probotLossNF")
  expect_identical(
    probotLossEval(le_x, le_y, le_mdn, le_K, loss_fn = probotLossMAE)$loss_name,
    "probotLossMAE")
  expect_identical(
    probotLossEval(le_x, le_y, le_pt, loss_fn = function(a, b) a$mean())$loss_name,
    "<custom>")
})

# The point of the row weighting: an unweighted mean of per-batch losses drifts
# whenever the final batch is short. batch = 7 does not divide 200.
test_that("probotLossEval is invariant to batch size", {
  one <- probotLossEval(le_x, le_y, le_mdn, le_K, batch = 1L)$loss
  seven <- probotLossEval(le_x, le_y, le_mdn, le_K, batch = 7L)$loss
  whole <- probotLossEval(le_x, le_y, le_mdn, le_K, batch = 1e4)$loss
  expect_equal(one, whole, tolerance = 1e-6)
  expect_equal(seven, whole, tolerance = 1e-6)

  f1 <- probotLossEval(le_x, le_y, le_fl, batch = 1L)$loss
  f13 <- probotLossEval(le_x, le_y, le_fl, batch = 13L)$loss
  expect_equal(f1, f13, tolerance = 1e-6)
})

test_that("probotLossEval counts batches correctly", {
  res <- probotLossEval(le_x, le_y, le_fl, batch = 137L)
  expect_identical(res$batches, 2L)
  res2 <- probotLossEval(le_x, le_y, le_fl, batch = le_n)
  expect_identical(res2$batches, 1L)
})

# Guards against the two easiest mistakes: swapping the argument order (this
# function takes context first, unlike probotLoss*()), and a stale idx.
test_that("probotLossEval rejects swapped input and output", {
  expect_error(probotLossEval(le_y, le_x, le_pt), "wrong order")
})

test_that("probotLossEval validates idx", {
  expect_error(probotLossEval(le_x, le_y, le_pt, idx = 1:99999), "beyond the data")
  expect_error(probotLossEval(le_x, le_y, le_pt, idx = c(0L, 5L)), "positive")
  expect_error(probotLossEval(le_x, le_y, le_pt, idx = c(5L, NA)), "positive")
})

test_that("probotLossEval rejects missing values and mismatched rows", {
  bad <- le_x
  bad[1, 1] <- NA
  expect_error(probotLossEval(bad, le_y, le_pt), "missing values")
  expect_error(probotLossEval(le_x, le_y[1:10, , drop = FALSE], le_pt),
               "same number of rows")
  expect_error(probotLossEval(le_x, le_y, le_pt, idx = integer(0)), "n = 0")
})

test_that("probotLossEval rejects a non-module and an unusable batch", {
  expect_error(probotLossEval(le_x, le_y, lm(y ~ x, data = data.frame(y = rnorm(5), x = 1:5))),
               "nn_module")
  expect_error(probotLossEval(le_x, le_y, le_pt, batch = 0L), "whole number")
})

test_that("probotLossEval needs mdn_components only for an MDN", {
  # Post-0.7.0 constructors store the count, so omitting it works.
  expect_true(is.finite(probotLossEval(le_x, le_y, le_mdn)$loss))
  # A foreign mdn_components is ignored quietly, so a model-comparison loop can
  # pass one set of arguments to every arm.
  expect_no_warning(probotLossEval(le_x, le_y, le_pt, mdn_components = le_K))
  expect_no_warning(probotLossEval(le_x, le_y, le_fl, mdn_components = le_K))
})

# The whole point of a scoring helper is that it must not perturb the model.
test_that("probotLossEval leaves parameters and training mode untouched", {
  before <- lapply(le_mdn$state_dict(), function(p) as.matrix(p$to(device = "cpu")))
  le_mdn$train()
  invisible(probotLossEval(le_x, le_y, le_mdn, le_K, batch = 64L))
  after <- lapply(le_mdn$state_dict(), function(p) as.matrix(p$to(device = "cpu")))
  expect_identical(before, after)
  expect_true(le_mdn$training)

  le_mdn$eval()
  invisible(probotLossEval(le_x, le_y, le_mdn, le_K))
  expect_false(le_mdn$training)
})

test_that("probotLossEval matches a direct single-batch loss call", {
  xt <- torch_tensor(le_x); yt <- torch_tensor(le_y)
  expect_equal(
    probotLossEval(le_x, le_y, le_mdn, le_K, batch = le_n)$loss,
    probotLossMDN(yt, le_mdn(xt), le_K)$item(), tolerance = 1e-7)
  expect_equal(
    probotLossEval(le_x, le_y, le_fl, batch = le_n)$loss,
    probotLossNF(yt, xt, le_fl)$item(), tolerance = 1e-7)
  expect_equal(
    probotLossEval(le_x, le_y, le_pt, batch = le_n)$loss,
    nnf_mse_loss(le_pt(xt), yt)$item(), tolerance = 1e-7)
})

test_that("probotLossEval's idx matches manual subsetting", {
  sub <- 11:40
  expect_equal(
    probotLossEval(le_x, le_y, le_fl, idx = sub)$loss,
    probotLossEval(le_x[sub, , drop = FALSE], le_y[sub, , drop = FALSE],
                   le_fl)$loss,
    tolerance = 1e-12)
})

test_that("probotLossEval accepts data frames, vectors and tensors", {
  expect_equal(probotLossEval(as.data.frame(le_x), le_y, le_pt)$loss,
               probotLossEval(le_x, le_y, le_pt)$loss, tolerance = 1e-12)
  expect_equal(probotLossEval(torch_tensor(le_x), torch_tensor(le_y), le_pt)$loss,
               probotLossEval(le_x, le_y, le_pt)$loss, tolerance = 1e-12)
  expect_true(is.finite(probotLossEval(le_x[1, , drop = FALSE],
                                       le_y[1, , drop = FALSE], le_pt)$loss))
})

test_that("probotLossEval works for every flow style, including a location head", {
  for (style in c("realnvp", "maf", "nsf")) {
    mdl <- probotMakeFlow(input_dim = le_C, output_dim = le_D, n_layers = 2,
                          hidden_dim = 16, style = style)()
    res <- probotLossEval(le_x, le_y, mdl)
    expect_identical(res$model_type, "flow")
    expect_true(is.finite(res$loss))
  }
  headed <- probotMakeFlow(input_dim = le_C, output_dim = le_D, n_layers = 2,
                           hidden_dim = 16, style = "nsf", loc_head = TRUE)()
  expect_true(is.finite(probotLossEval(le_x, le_y, headed)$loss))
})

test_that("probotLossEval verbose reports the score it returns", {
  expect_output(
    res <- probotLossEval(le_x, le_y, le_fl, batch = 64L, verbose = TRUE),
    "probotLossEval: flow probotLossNF")
  expect_true(is.finite(res$loss))
})
