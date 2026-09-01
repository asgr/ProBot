library(testthat)
library(ProBot)
library(torch)

# --- Helper: build a raw MDN output tensor for loss tests ---
make_mdn_pred <- function(batch_size, output_dim, mdn_components, device = "cpu") {
  n_features <- mdn_components * (2 * output_dim + 1)
  torch_zeros(batch_size, n_features, device = device)
}

make_mdn_true <- function(batch_size, output_dim, device = "cpu") {
  torch_zeros(batch_size, output_dim, device = device)
}

test_that("probotLossMDN returns a scalar tensor", {
  set.seed(42)
  batch <- 8; od <- 2; K <- 3
  y_true <- torch_randn(batch, od)
  y_pred <- torch_randn(batch, K * (2 * od + 1))
  loss <- probotLossMDN(y_true, y_pred, K)
  expect_s3_class(loss, "torch_tensor")
  expect_equal(length(loss$shape), 0L)  # scalar
})

test_that("probotLossMDN is finite", {
  set.seed(42)
  batch <- 8; od <- 2; K <- 3
  y_true <- torch_randn(batch, od)
  y_pred <- torch_randn(batch, K * (2 * od + 1))
  loss <- probotLossMDN(y_true, y_pred, K)
  expect_true(is.finite(as.numeric(loss)))
})

test_that("probotLossMDN handles 1D input", {
  set.seed(42)
  y_true <- torch_randn(2)  # 1D
  y_pred <- torch_randn(1, 3 * (2 * 1 + 1))
  loss <- probotLossMDN(y_true, y_pred, 3)
  expect_true(is.finite(as.numeric(loss)))
})

test_that("probotLossMSE returns a scalar tensor", {
  set.seed(42)
  batch <- 8; od <- 2; K <- 3
  y_true <- torch_randn(batch, od)
  y_pred <- torch_randn(batch, K * (2 * od + 1))
  loss <- probotLossMSE(y_true, y_pred, K)
  expect_s3_class(loss, "torch_tensor")
  expect_true(is.finite(as.numeric(loss)))
})

test_that("probotLossMAE returns a scalar tensor", {
  set.seed(42)
  batch <- 8; od <- 2; K <- 3
  y_true <- torch_randn(batch, od)
  y_pred <- torch_randn(batch, K * (2 * od + 1))
  loss <- probotLossMAE(y_true, y_pred, K)
  expect_s3_class(loss, "torch_tensor")
  expect_true(is.finite(as.numeric(loss)))
})

test_that("probotLossMAPE returns a scalar tensor", {
  set.seed(42)
  batch <- 8; od <- 2; K <- 3
  y_true <- torch_randn(batch, od) + 1  # avoid zeros
  y_pred <- torch_randn(batch, K * (2 * od + 1))
  loss <- probotLossMAPE(y_true, y_pred, K)
  expect_s3_class(loss, "torch_tensor")
  expect_true(is.finite(as.numeric(loss)))
})

test_that("probotLossHuber returns a scalar tensor", {
  set.seed(42)
  batch <- 8; od <- 2; K <- 3
  y_true <- torch_randn(batch, od)
  y_pred <- torch_randn(batch, K * (2 * od + 1))
  loss <- probotLossHuber(y_true, y_pred, K, delta = 0.5)
  expect_s3_class(loss, "torch_tensor")
  expect_true(is.finite(as.numeric(loss)))
})

test_that("probotLossHuber with default delta works", {
  set.seed(42)
  batch <- 8; od <- 2; K <- 3
  y_true <- torch_randn(batch, od)
  y_pred <- torch_randn(batch, K * (2 * od + 1))
  loss <- probotLossHuber(y_true, y_pred, K)
  expect_true(is.finite(as.numeric(loss)))
})

test_that("probotLossNF returns a scalar tensor", {
  set.seed(42)
  output_dim <- 4; input_dim <- 2; batch <- 4
  model <- probotMakeFlow(input_dim, output_dim, n_layers = 2, hidden_dim = 8)()
  theta <- torch_randn(batch, output_dim)
  x <- torch_randn(batch, input_dim)
  theta <- theta$to(device = model$parameters[[1]]$device)
  x <- x$to(device = model$parameters[[1]]$device)

  loss <- probotLossNF(theta, x, model)
  expect_s3_class(loss, "torch_tensor")
  expect_true(is.finite(as.numeric(loss)))
})

test_that("MSE loss is zero when predictions match truth", {
  batch <- 4; od <- 2; K <- 2
  y_true <- torch_tensor(matrix(c(0, 0, 0, 0, 0, 0, 0, 0), batch, od))

  # Build a prediction where all components have mu=0, equal logits
  pred <- torch_zeros(batch, K * (2 * od + 1))
  loss <- probotLossMSE(y_true, pred, K)
  expect_equal(as.numeric(loss), 0, tolerance = 1e-6)
})
