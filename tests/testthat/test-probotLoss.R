library(testthat)
library(ProBot)
library(torch)

# --- Helper: an MDN whose head emits (batch_size, K * (2 * output_dim + 1)) ---
# The MDN losses read the mixture count from the model, so a test can no longer
# hand them a bare tensor: .probotUnpackMDN() needs something that knows its own
# K. Zero weights/bias give a deterministic (mu, log10_sigma, logit) = (0, 0, 0)
# head; the same K/output_dim reproduce the widths the old tensor fixtures used.
make_mdn_model <- function(output_dim, mdn_components, zero = TRUE, seed = 42) {
  set.seed(seed)
  mdl <- probotMakeMDN(input_dim = 3L, output_dim = output_dim,
                       mdn_components = mdn_components,
                       hidden_dims = c(8L), device = "cpu")()
  if (zero) {
    head <- mdl$layers[[length(mdl$layers)]]
    with_no_grad({
      head$weight$zero_()
      head$bias$zero_()
    })
  }
  mdl$eval()
  mdl
}

make_mdn_pred <- function(model, batch_size) {
  model(torch_randn(batch_size, 3, device = "cpu"))
}

test_that("probotLossMDN returns a scalar tensor", {
  mdl <- make_mdn_model(output_dim = 2, mdn_components = 3)
  y_true <- torch_randn(8, 2)
  loss <- probotLossMDN(y_true, make_mdn_pred(mdl, 8), mdl)
  expect_s3_class(loss, "torch_tensor")
  expect_equal(length(loss$shape), 0L)  # scalar
})

test_that("probotLossMDN is finite", {
  mdl <- make_mdn_model(output_dim = 2, mdn_components = 3)
  y_true <- torch_randn(8, 2)
  loss <- probotLossMDN(y_true, make_mdn_pred(mdl, 8), mdl)
  expect_true(is.finite(as.numeric(loss)))
})

test_that("probotLossMDN handles 1D input", {
  mdl <- make_mdn_model(output_dim = 1, mdn_components = 3)
  y_true <- torch_randn(2)  # 1D, unscaled to (2, 1) by the loss
  loss <- probotLossMDN(y_true, make_mdn_pred(mdl, 2), mdl)
  expect_true(is.finite(as.numeric(loss)))
})

test_that("probotLossMSE returns a scalar tensor", {
  mdl <- make_mdn_model(output_dim = 2, mdn_components = 3)
  y_true <- torch_randn(8, 2)
  loss <- probotLossMSE(y_true, make_mdn_pred(mdl, 8), mdl)
  expect_s3_class(loss, "torch_tensor")
  expect_true(is.finite(as.numeric(loss)))
})

test_that("probotLossMAE returns a scalar tensor", {
  mdl <- make_mdn_model(output_dim = 2, mdn_components = 3)
  y_true <- torch_randn(8, 2)
  loss <- probotLossMAE(y_true, make_mdn_pred(mdl, 8), mdl)
  expect_s3_class(loss, "torch_tensor")
  expect_true(is.finite(as.numeric(loss)))
})

test_that("probotLossMAPE returns a scalar tensor", {
  mdl <- make_mdn_model(output_dim = 2, mdn_components = 3)
  y_true <- torch_randn(8, 2) + 1  # avoid zeros
  loss <- probotLossMAPE(y_true, make_mdn_pred(mdl, 8), mdl)
  expect_s3_class(loss, "torch_tensor")
  expect_true(is.finite(as.numeric(loss)))
})

test_that("probotLossHuber returns a scalar tensor", {
  mdl <- make_mdn_model(output_dim = 2, mdn_components = 3)
  y_true <- torch_randn(8, 2)
  loss <- probotLossHuber(y_true, make_mdn_pred(mdl, 8), mdl, delta = 0.5)
  expect_s3_class(loss, "torch_tensor")
  expect_true(is.finite(as.numeric(loss)))
})

test_that("probotLossHuber with default delta works", {
  mdl <- make_mdn_model(output_dim = 2, mdn_components = 3)
  y_true <- torch_randn(8, 2)
  loss <- probotLossHuber(y_true, make_mdn_pred(mdl, 8), mdl)
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

test_that("probotLossNF pairs each row's density term with its own Jacobian", {
  # The per-row log-likelihood must be an (N, 1) column. It used to be built as
  # a (1, N) base term plus the model's (N, 1) log|det J|, which broadcasts to
  # (N, N): element [i, j] pairs row i's Jacobian with row j's base density.
  # mean() over that matrix equals the correct row mean in real arithmetic, so
  # the *scalar loss* was identical and no value test can catch this -- only the
  # shape can. The shape is also the whole problem: (N, N) is 8.1e9 float32
  # elements at N = 90000, which is where a holdout scored in one batch started
  # reporting +223 for a model whose true verify NLL was -13.5.
  n <- 50L; d <- 4L
  # Deterministic z and log|det J|: distinct per row, no RNG to drift.
  zv <- seq_len(n * d) / 37 - 1.5
  zm <- matrix(zv, nrow = n)
  out <- list(
    z = torch_tensor(zm, dtype = torch_float()),
    log_det_jac = (torch_arange(1, n, dtype = torch_float()) * 3)$unsqueeze(2)
  )
  ll <- ProBot:::.probotNFLogLik(out)
  expect_identical(as.integer(ll$shape), c(n, 1L))
  expect_identical(as.integer(ll$numel()), n)

  z2row <- rowSums(zm^2)
  ldv <- as.numeric(seq_len(n) * 3)
  # Row i is row i's own log p: lpz_i + ld_i, both from the same row.
  expect_equal(as.numeric(as.array(ll)),
               -0.5 * (z2row + d * log(2 * pi)) + ldv, tolerance = 1e-5)

  # Negative control for the failure mode. The old (N, N) matrix had element
  # [i, j] = lpz_j + ld_i, so its mean was mean(lpz) + mean(ld) -- the same
  # number as the correct row mean, which is why no value test caught this. Only
  # the per-row values differ, and only the broadcast version is constant.
  cross <- mean(-0.5 * (z2row + d * log(2 * pi))) + mean(ldv)
  expect_equal(mean(as.numeric(as.array(ll))), cross, tolerance = 1e-5)
  expect_false(isTRUE(all.equal(as.numeric(as.array(ll)), rep(cross, n),
                                tolerance = 1e-2)))

  # And the scalar the trainer reports really is -mean(ll).
  stub <- torch::nn_module("nf_stub",
    initialize = function(o) { self$o <- o },
    forward = function(theta, x) self$o)
  mdl <- stub(out)
  expect_equal(as.numeric(probotLossNF(torch_randn(n, d), torch_randn(n, 2), mdl)),
               -cross, tolerance = 1e-5)
})

test_that("MSE loss is zero when predictions match truth", {
  # Zero-weighted head => every mu is 0, so the mixture mean is exactly 0.
  mdl <- make_mdn_model(output_dim = 2, mdn_components = 2)
  y_true <- torch_zeros(4, 2)
  loss <- probotLossMSE(y_true, make_mdn_pred(mdl, 4), mdl)
  expect_equal(as.numeric(loss), 0, tolerance = 1e-6)
})

# ---- the mixture count comes from the model, not from a caller ---------------

test_that("MDN losses resolve an ambiguous head from the stored fields", {
  # K = 5 with D = 2 and K = 1 with D = 12 both emit a 25-unit head, so the
  # width alone cannot say which; only the stored fields can settle it.
  mdl1 <- make_mdn_model(output_dim = 12, mdn_components = 1, seed = 7)
  rows1 <- mdl1$layers[[length(mdl1$layers)]]$weight$size(1)
  expect_identical(as.integer(rows1), 25L)
  expect_identical(ProBot:::.probotMDNK(mdl1, rows1), 1L)

  mdl5 <- make_mdn_model(output_dim = 2, mdn_components = 5, seed = 7)
  rows5 <- mdl5$layers[[length(mdl5$layers)]]$weight$size(1)
  expect_identical(as.integer(rows5), 25L)
  expect_identical(ProBot:::.probotMDNK(mdl5, rows5), 5L)

  # Unpacking must follow the same reading: (batch, K, D).
  p1 <- ProBot:::.probotUnpackMDN(make_mdn_pred(mdl1, 4), mdl1)
  p5 <- ProBot:::.probotUnpackMDN(make_mdn_pred(mdl5, 4), mdl5)
  expect_equal(as.integer(dim(p1$mu)), c(4L, 1L, 12L))
  expect_equal(as.integer(dim(p5$mu)), c(4L, 5L, 2L))
})

test_that(".probotMDNK resolves an ambiguous head from output_dim alone", {
  # A hand-built module that stores only output_dim: 25 units with D = 2 can
  # only be K = 5.
  mdl <- nn_module(initialize = function() {
    self$output_dim <- 2L
    self$head <- nn_linear(3, 25)
  }, forward = function(x) self$head(x))()
  expect_identical(ProBot:::.probotMDNK(mdl, 25L), 5L)
})

test_that(".probotMDNK resolves a head with a unique factorisation", {
  # 3 units can only be K = 1, D = 1, so an unannotated module is still usable.
  mdl <- nn_module(initialize = function() {
    self$head <- nn_linear(3, 3)
  }, forward = function(x) self$head(x))()
  expect_identical(ProBot:::.probotMDNK(mdl, 3L), 1L)
})

test_that(".probotMDNK refuses an ambiguous head with no stored fields", {
  # 15 units admits (K=1,D=7), (K=3,D=2) and (K=5,D=1); guessing would
  # mis-split the head, so the error names the splits it cannot choose between.
  mdl <- nn_module(initialize = function() {
    self$head <- nn_linear(3, 15)
  }, forward = function(x) self$head(x))()
  expect_error(ProBot:::.probotMDNK(mdl, 15L), "3 splits")
  expect_error(ProBot:::.probotMDNK(mdl, 15L), "K=3, D=2")
})

test_that(".probotMDNK reports a model inconsistent with its own head", {
  # Stored K/D imply a 15-unit head but the module emits 25 units: silently
  # trusting either number would mis-split it.
  mdl <- nn_module(initialize = function() {
    self$mdn_components <- 3L
    self$output_dim <- 2L
    self$head <- nn_linear(3, 25)
  }, forward = function(x) self$head(x))()
  expect_error(ProBot:::.probotMDNK(mdl, 25L), "internally inconsistent")
  # The same fields against the width they do imply resolve without complaint.
  expect_identical(ProBot:::.probotMDNK(mdl, 15L), 3L)
})

test_that(".probotMDNK rejects a stored count that cannot divide the head", {
  # 25 units with K = 4 has no integer D, so the count alone is not usable.
  mdl <- nn_module(initialize = function() {
    self$mdn_components <- 4L
    self$head <- nn_linear(3, 25)
  }, forward = function(x) self$head(x))()
  expect_error(ProBot:::.probotMDNK(mdl, 25L), "not consistent")
})

test_that(".probotMDNK names the removed argument when handed a bare number", {
  # The most likely way to hit this: a legacy positional mdn_components that
  # survived the signature change and arrived here as `model`.
  expect_error(ProBot:::.probotMDNK(3, 15L), "mdn_components")
  expect_error(ProBot:::.probotMDNK(NULL, 15L), "mdn_components")
})
