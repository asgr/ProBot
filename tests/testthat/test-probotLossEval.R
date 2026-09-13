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
    res <- probotLossEval(le_x, le_y, spec)
    expect_type(res, "list")
    expect_named(res, c("loss", "n", "batches", "model_type", "loss_name"))
    expect_length(res$loss, 1L)
    expect_true(is.finite(res$loss))
    expect_identical(res$n, le_n)
  }
})

test_that("probotLossEval dispatches the model type from the object", {
  expect_identical(probotLossEval(le_x, le_y, le_mdn)$model_type, "mdn")
  expect_identical(probotLossEval(le_x, le_y, le_pt)$model_type, "point")
  expect_identical(probotLossEval(le_x, le_y, le_fl)$model_type, "flow")
})

test_that("probotLossEval names the loss it used", {
  expect_identical(probotLossEval(le_x, le_y, le_mdn)$loss_name,
                   "probotLossMDN")
  expect_identical(probotLossEval(le_x, le_y, le_pt)$loss_name, "nnf_mse_loss")
  expect_identical(probotLossEval(le_x, le_y, le_fl)$loss_name, "probotLossNF")
  expect_identical(
    probotLossEval(le_x, le_y, le_mdn, loss_fn = probotLossMAE)$loss_name,
    "probotLossMAE")
  expect_identical(
    probotLossEval(le_x, le_y, le_pt, loss_fn = function(a, b) a$mean())$loss_name,
    "<custom>")
})

# The point of the row weighting: an unweighted mean of per-batch losses drifts
# whenever the final batch is short. batch = 7 does not divide 200.
test_that("probotLossEval is invariant to batch size", {
  one <- probotLossEval(le_x, le_y, le_mdn, batch = 1L)$loss
  seven <- probotLossEval(le_x, le_y, le_mdn, batch = 7L)$loss
  whole <- probotLossEval(le_x, le_y, le_mdn, batch = 1e4)$loss
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

test_that("probotLossEval no longer takes a mixture count", {
  # The count is read from the model, so all three types score with the same
  # three leading arguments -- which is what a model-comparison loop needs.
  expect_true(is.finite(probotLossEval(le_x, le_y, le_mdn)$loss))
  expect_true(is.finite(probotLossEval(le_x, le_y, le_pt)$loss))
  expect_true(is.finite(probotLossEval(le_x, le_y, le_fl)$loss))
  for (spec in list(le_mdn, le_pt, le_fl)) {
    expect_identical(names(probotLossEval(le_x, le_y, spec)),
                     c("loss", "n", "batches", "model_type", "loss_name"))
  }
  # mdn_components was the fourth formal and idx is now the fourth, so a legacy
  # positional count would silently score one row; it must be refused outright.
  expect_error(probotLossEval(le_x, le_y, le_mdn, le_K), "positionally")
  expect_error(probotLossEval(le_x, le_y, le_mdn, mdn_components = le_K),
               "unused argument")
})

# The whole point of a scoring helper is that it must not perturb the model.
test_that("probotLossEval leaves parameters and training mode untouched", {
  before <- lapply(le_mdn$state_dict(), function(p) as.matrix(p$to(device = "cpu")))
  le_mdn$train()
  invisible(probotLossEval(le_x, le_y, le_mdn, batch = 64L))
  after <- lapply(le_mdn$state_dict(), function(p) as.matrix(p$to(device = "cpu")))
  expect_identical(before, after)
  expect_true(le_mdn$training)

  le_mdn$eval()
  invisible(probotLossEval(le_x, le_y, le_mdn))
  expect_false(le_mdn$training)
})

test_that("probotLossEval matches a direct single-batch loss call", {
  # The direct calls below hand tensors to the model itself, so they must be on
  # the model's device: probotMake*() auto-places on MPS where it is available,
  # and a CPU tensor then errors. probotLossEval() needs no such care because it
  # copies its input to CPU and re-places it on the resolved device.
  on_dev <- function(m, model) torch_tensor(m, device = model$parameters[[1]]$device)
  expect_equal(
    probotLossEval(le_x, le_y, le_mdn, batch = le_n)$loss,
    probotLossMDN(on_dev(le_y, le_mdn), le_mdn(on_dev(le_x, le_mdn)), le_mdn)$item(),
    tolerance = 1e-7)
  expect_equal(
    probotLossEval(le_x, le_y, le_fl, batch = le_n)$loss,
    probotLossNF(on_dev(le_y, le_fl), on_dev(le_x, le_fl), le_fl)$item(),
    tolerance = 1e-7)
  expect_equal(
    probotLossEval(le_x, le_y, le_pt, batch = le_n)$loss,
    nnf_mse_loss(le_pt(on_dev(le_x, le_pt)), on_dev(le_y, le_pt))$item(),
    tolerance = 1e-7)
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

# ---- per_row -----------------------------------------------------------------

# The default must stay exactly the pre-per_row list, so existing callers that
# index or expect_named() the result keep working.
test_that("per_row defaults to FALSE and leaves the result shape alone", {
  res <- probotLossEval(le_x, le_y, le_fl)
  expect_named(res, c("loss", "n", "batches", "model_type", "loss_name"))
  expect_null(res$row_loss)
})

test_that("per_row returns one finite loss per scored row", {
  for (spec in list(mdn = le_mdn, point = le_pt, flow = le_fl)) {
    res <- probotLossEval(le_x, le_y, spec, per_row = TRUE)
    expect_length(res$row_loss, le_n)
    expect_true(all(is.finite(res$row_loss)))
    expect_true(res$loss_median <= res$loss_p90)
    expect_true(res$loss_max >= res$loss_p90)
    expect_gte(res$n_extreme, 0L)
  }
})

# The summary exists to catch a single tail row dominating a mean, so the two
# must be consistent: mean(row_loss) is the reported loss up to float32 error.
test_that("per-row losses average back to the reported scalar loss", {
  for (spec in list(mdn = le_mdn, point = le_pt, flow = le_fl)) {
    res <- probotLossEval(le_x, le_y, spec, per_row = TRUE)
    expect_equal(mean(res$row_loss), res$loss, tolerance = 1e-5)
  }
})

# probotLossMDN clamps log10_sigma before using it in *both* sigma and the
# log-normalisation. A row decomposition that used the raw value would agree in
# the easy cases and silently disagree once the clamp binds. Only the
# log10_sigma slice of the head is inflated: scaling mu as well drives the loss
# to ~1e8, where a real absolute error hides inside a loose relative tolerance.
test_that("per_row agrees with the scalar loss when the MDN sigma clamp binds", {
  big <- probotMakeMDN(input_dim = le_C, output_dim = le_D,
                       mdn_components = le_K, hidden_dims = c(16, 16))()
  head <- big$layers[[length(big$layers)]]
  # Head is flattened as K * (2D + 1), laid out per component as
  # mu 1..D, log10_sigma (D+1)..2D, logit 2D+1 -- hence the (2D + 1) stride.
  n_head <- le_K * (2L * le_D + 1L)
  sig_rows <- as.vector(outer((seq_len(le_K) - 1L) * (2L * le_D + 1L),
                              le_D + seq_len(le_D), `+`))
  sig_mask <- rep(0, n_head); sig_mask[sig_rows] <- 1
  keep <- 1 - sig_mask
  dev <- head$bias$device
  keep_t <- torch_tensor(keep, dtype = head$bias$dtype, device = dev)
  mask_t <- torch_tensor(sig_mask, dtype = head$bias$dtype, device = dev)
  with_no_grad({
    # Zero the sigma rows' weights and set their bias to +8, so every row has
    # log10_sigma == 8 and the +/-5 clamp binds uniformly. Inflating the weights
    # instead spreads log10_sigma both ways, and the negative tail pushes sigma
    # to 1e-5, making z^2 ~ 1e10 -- a loss of ~1e9 where any absolute
    # inconsistency hides inside a relative tolerance.
    head$weight$mul_(keep_t$unsqueeze(2))
    head$bias$mul_(keep_t)
    head$bias$add_(mask_t$mul_(8))
  })
  raw <- .probotUnpackMDN(
    big(torch_tensor(le_x, device = dev)), big)$log10_sigma
  expect_gt(as.numeric(torch_min(torch_abs(raw))), 5)

  res <- probotLossEval(le_x, le_y, big, per_row = TRUE)
  expect_true(all(is.finite(res$row_loss)))
  expect_equal(mean(res$row_loss), res$loss, tolerance = 1e-5)
})

test_that("per_row is invariant to batch size", {
  one <- probotLossEval(le_x, le_y, le_fl, per_row = TRUE, batch = 1L)$row_loss
  sev <- probotLossEval(le_x, le_y, le_fl, per_row = TRUE, batch = 7L)$row_loss
  all_ <- probotLossEval(le_x, le_y, le_fl, per_row = TRUE, batch = 1e4)$row_loss
  expect_equal(one, all_, tolerance = 1e-5)
  expect_equal(sev, all_, tolerance = 1e-5)
})

test_that("per_row respects idx", {
  sub <- 11:40
  res <- probotLossEval(le_x, le_y, le_fl, idx = sub, per_row = TRUE)
  ref <- probotLossEval(le_x[sub, , drop = FALSE], le_y[sub, , drop = FALSE],
                        le_fl, per_row = TRUE)
  expect_length(res$row_loss, 30L)
  expect_equal(res$row_loss, ref$row_loss, tolerance = 1e-6)
})

# A hand-built flow with an enormous latent is the failure mode this feature
# exists to surface: the mean is wrecked while the median is untouched.
test_that("per_row flags a single tail row that dominates the mean", {
  res <- probotLossEval(le_x, le_y, le_fl, per_row = TRUE)
  spiked <- res$row_loss
  spiked[1] <- spiked[1] + 1e5
  s <- ProBot:::.probotLossEvalSummary(spiked)
  # One huge row should wreck the mean while barely moving the median. The
  # 1e5 spike moves the mean by ~500 and the median by at most a few MADs.
  expect_gt(mean(spiked) - mean(res$row_loss), 50)
  expect_lt(abs(stats::median(spiked) - stats::median(res$row_loss)),
            2 * s$loss_mad)
  expect_identical(s$n_extreme, 1L)
  expect_gt(s$loss_max, 1e5)
  expect_equal(s$extreme_threshold, s$loss_median + 10 * s$loss_mad, tolerance = 1e-8)
})

test_that("a non-decomposable loss_fn warns and omits the summary", {
  expect_warning(
    res <- probotLossEval(le_x, le_y, le_mdn,
                          loss_fn = probotLossMAE, per_row = TRUE),
    "default loss")
  expect_named(res, c("loss", "n", "batches", "model_type", "loss_name"))
})

test_that("per_row works for every flow style and a location head", {
  for (style in c("realnvp", "maf", "nsf")) {
    mdl <- probotMakeFlow(input_dim = le_C, output_dim = le_D, n_layers = 2,
                          hidden_dim = 16, style = style)()
    res <- probotLossEval(le_x, le_y, mdl, per_row = TRUE)
    expect_length(res$row_loss, le_n)
    expect_equal(mean(res$row_loss), res$loss, tolerance = 1e-5)
  }
  headed <- probotMakeFlow(input_dim = le_C, output_dim = le_D, n_layers = 2,
                           hidden_dim = 16, style = "nsf", loc_head = TRUE)()
  hres <- probotLossEval(le_x, le_y, headed, per_row = TRUE)
  expect_equal(mean(hres$row_loss), hres$loss, tolerance = 1e-5)
})
