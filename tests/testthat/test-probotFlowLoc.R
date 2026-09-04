library(testthat)
library(ProBot)
library(torch)

# Tests for the residual location head, theta = mu(x) + f_inverse(z, x).
#
# The load-bearing property is that wrapping a flow in the head must not change
# the change-of-variables maths: z = f_forward(theta - mu(x), x) has an identity
# Jacobian in theta, so the base flow's log_det_jac is already
# log|det dz/dtheta| exactly. Every test below that could otherwise pass by
# accident is anchored to that claim with an independent Jacobian computation.

# Full (D x D) Jacobian built by differentiating each z component in its own
# graph. Summing z before differentiating would give a row-summed gradient, not
# a determinant, and a fresh forward pass per column is required because the
# graph is consumed by each backward() -- both are traps that have previously
# manufactured "disagreements" of order 1 nats that were not model bugs.
.probotExactLogDet <- function(model, theta_row, x_row) {
  D <- theta_row$size(2)
  J <- matrix(NA_real_, D, D)
  for (j in seq_len(D)) {
    ti <- theta_row$clone()$detach()$requires_grad_(TRUE)
    o <- model$forward(ti, x_row)
    g <- autograd_grad(o$z$narrow(2, j, 1)$sum(), ti)[[1]]
    J[j, ] <- as.numeric(g$squeeze())
  }
  as.numeric(determinant(J, log = TRUE)$modulus)
}

.make_loc <- function(style = "nsf", n_layers = 3, hidden_dim = 16, ...) {
  probotMakeFlow(10, 4, style = style, n_layers = n_layers, n_blocks = 2,
                 hidden_dim = hidden_dim, loc_head = TRUE,
                 loc_hidden_dims = c(8, 8), device = "cpu", ...)()
}

test_that("probotMakeFlow wraps every style in a location head", {
  for (style in c("realnvp", "maf", "nsf")) {
    mdl <- .make_loc(style)
    expect_s3_class(mdl, "probotFlowLoc")
    expect_true(inherits(mdl, .probotFlowClasses))
    # The head contributes 3 nn_linears (weights + biases); everything else is
    # namespaced under base_flow.
    expect_equal(sum(grepl("^loc_layers[.]", names(mdl$state_dict()))), 6)
    # Reported log_det_jac disagrees in orientation between styles ((1,B) for
    # realnvp, (B,1) for maf/nsf); only its length is contractual.
    o <- mdl$forward(torch_tensor(matrix(rnorm(6 * 4), 6, 4)),
                     torch_tensor(matrix(rnorm(6 * 10), 6, 10)))
    expect_equal(o$z$size(2), 4)
    expect_equal(o$log_det_jac$numel(), 6)
  }
})

test_that("the head leaves the change of variables exact", {
  # Double precision: a float32 Jacobian check could only bound the error near
  # 1e-3, too loose to separate "correct" from "nearly".
  x <- torch_tensor(matrix(rnorm(8 * 10), 8, 10), dtype = torch_double())
  th <- torch_tensor(matrix(rnorm(8 * 4), 8, 4), dtype = torch_double())

  for (style in c("realnvp", "maf", "nsf")) {
    headed <- .make_loc(style)
    headed$to(dtype = torch_double())
    plain <- probotMakeFlow(10, 4, style = style, n_layers = 3, n_blocks = 2,
                            hidden_dim = 16, device = "cpu")()
    plain$to(dtype = torch_double())

    for (mdl in list(headed, plain)) {
      exact <- vapply(seq_len(8),
                      function(i) .probotExactLogDet(mdl, th[i, , drop = FALSE],
                                                     x[i, , drop = FALSE]), 0)
      reported <- as.numeric(mdl$forward(th, x)$log_det_jac$cpu())
      expect_lt(max(abs(reported - exact)), 1e-8)
    }
  }
})

test_that("the head is an exact additive shift with no z dependence", {
  mdl <- .make_loc("nsf")
  x <- torch_tensor(matrix(rnorm(3 * 10), 3, 10))
  mu <- as.matrix(mdl$location(x)$cpu())
  z1 <- torch_randn(c(3, 4))

  inv1 <- as.matrix(mdl$inverse(z1, x)$cpu())
  base1 <- as.matrix(mdl$base_flow$inverse(z1, x)$cpu())
  expect_lt(max(abs((inv1 - base1) - mu)), 1e-5)

  # forward() and inverse() must compose through the head.
  th <- torch_tensor(matrix(rnorm(3 * 4), 3, 4))
  back <- as.matrix(mdl$inverse(mdl$forward(th, x)$z, x)$cpu())
  expect_lt(max(abs(back - as.matrix(th$cpu()))), 1e-3)
})

test_that("probotLossNF consumes a headed model unchanged", {
  mdl <- .make_loc("nsf")
  loss <- probotLossNF(torch_tensor(matrix(rnorm(16 * 4), 16, 4)),
                       torch_tensor(matrix(rnorm(16 * 10), 16, 10)), mdl)
  expect_true(is.finite(loss$item()))
  expect_no_error(loss$backward())
})

test_that("point = 'loc' reads the head and refuses models without one", {
  mdl <- .make_loc("nsf")
  x <- torch_tensor(matrix(rnorm(5 * 10), 5, 10))
  est <- ProBot:::.probotFlowPointEstimate(mdl, x, output_dim = 4, point = "loc")
  expect_lt(max(abs(as.matrix((est - mdl$location(x))$cpu()))), 1e-12)

  plain <- probotMakeFlow(10, 4, style = "nsf", n_layers = 2, device = "cpu")()
  expect_error(
    ProBot:::.probotFlowPointEstimate(plain, x, output_dim = 4, point = "loc"),
    "loc_head"
  )
})

test_that("probotFlowLoc round trips through probotSave/probotLoad", {
  path <- tempfile(fileext = ".pt")
  on.exit(unlink(path), add = TRUE)

  mdl <- .make_loc("nsf", n_layers = 3, hidden_dim = 16)
  opt <- optim_adam(mdl$parameters, lr = 1e-3)
  z <- torch_randn(c(8, 4))
  ctx <- torch_tensor(matrix(rnorm(8 * 10), 8, 10))
  with_no_grad({ pre <- as.matrix(mdl$inverse(z, ctx)$cpu()) })

  probotSave(model = mdl, optimizer = opt, filename = path, model_type = "flow",
             input_dim = 10, output_dim = 4, n_layers = 3, hidden_dim = 16,
             n_bins = 8, tail_bound = 3, soft_clamp = 3)

  back <- probotLoad(path, device = "cpu")
  expect_s3_class(back$model, "probotFlowLoc")
  # flow_style is inferred by unwrapping the head, not from class() directly.
  expect_equal(back$metadata$flow_style, "nsf")
  expect_true(isTRUE(back$metadata$loc_head))
  expect_equal(as.integer(back$metadata$loc_hidden_dims), c(8L, 8L))
  with_no_grad({ post <- as.matrix(back$model$inverse(z, ctx)$cpu()) })
  expect_lt(max(abs(pre - post)), 1e-5)

  # A headed checkpoint cannot be coerced into a plain skeleton by a style
  # override: the "base_flow." key namespace must fail the strict load.
  expect_error(probotLoad(path, device = "cpu", flow_style = "realnvp"))
})

test_that("checkpoints saved without loc_head metadata load as plain flows", {
  path <- tempfile(fileext = ".pt")
  on.exit(unlink(path), add = TRUE)
  mdl <- probotMakeFlow(10, 4, style = "maf", n_blocks = 2, n_layers = 2,
                        hidden_dim = 16, device = "cpu")()
  probotSave(model = mdl, filename = path, model_type = "flow", input_dim = 10,
             output_dim = 4, n_blocks = 2, hidden_dim = 16)
  back <- probotLoad(path, device = "cpu")
  expect_false(inherits(back$model, "probotFlowLoc"))
  expect_equal(back$metadata$flow_style, "maf")
})

test_that("point_estimate returns the head without drawing samples", {
  mdl <- .make_loc("nsf")
  x <- matrix(rnorm(5 * 10), 5, 10)

  est <- probotSamplePostNF(x, mdl, output_dim = 4, point_estimate = TRUE)
  expect_equal(dim(est), c(5, 4))
  expect_equal(est, probotSamplePostNF(x, mdl, output_dim = 4,
                                       point_estimate = TRUE))

  one <- probotSamplePostNF(x[1, ], mdl, output_dim = 4, point_estimate = TRUE)
  expect_null(dim(one))
  # Not bit-identical to row 1 of `est`: a length-1 batch hits a different
  # float32 BLAS path. Tight but nonzero.
  expect_equal(one, est[1, ], tolerance = 1e-5)

  cm <- 1:4; cs <- rep(2, 4)
  uns <- probotSamplePostNF(x, mdl, col_means = cm, col_sds = cs,
                            col_names = paste0("p", 4:1), point_estimate = TRUE)
  expect_equal(colnames(uns), c("p4", "p3", "p2", "p1"))
  expect_lt(max(abs(uns - (est * 2 + matrix(cm, 5, 4, byrow = TRUE)))), 1e-4)

  # On a plain flow the same argument falls back to the z = 0 centre.
  plain <- probotMakeFlow(10, 4, style = "nsf", n_layers = 2, device = "cpu")()
  p <- probotSamplePostNF(x, plain, output_dim = 4, point_estimate = TRUE)
  c0 <- plain$inverse(torch_zeros(c(5, 4)), torch_tensor(x))
  expect_lt(max(abs(p - as.matrix(c0$cpu()))), 1e-5)

  expect_error(probotSamplePostNF(x, mdl, point_estimate = TRUE), "output_dim")
})

test_that("assessment functions dispatch to a headed flow", {
  mdl <- .make_loc("nsf")
  inp <- matrix(rnorm(12 * 10), 12, 10)
  truth <- matrix(rnorm(12 * 4), 12, 4)

  torch_manual_seed(5); set.seed(5)
  pit <- probotPIT(inp, mdl, params = truth, n_test = 12, n_samples = 60,
                   batch_size = 12, verbose = FALSE)
  expect_equal(dim(pit), c(12, 4))
  expect_true(all(is.finite(pit)))

  torch_manual_seed(5); set.seed(5)
  crps <- probotCRPS(inp, mdl, params = truth, n_test = 12, n_samples = 60,
                     batch_size = 12, verbose = FALSE)
  expect_equal(dim(crps), c(12, 4))

  torch_manual_seed(5); set.seed(5)
  tarp <- probotTARP(inp, mdl, params = truth, n_test = 12, n_samples = 60,
                     batch_size = 12, verbose = FALSE)
  # TARP is one coverage probability per observation, so it has no dim.
  expect_length(tarp, 12)
})

test_that("training a headed flow keeps both lambda arms working", {
  x <- matrix(rnorm(48 * 10), 48, 10)
  th <- matrix(rnorm(48 * 4), 48, 4)

  for (lam in c(0, 0.3)) {
    mdl <- .make_loc("nsf")
    dl <- probotDataLoader(x, th, batch = 16, shuffle = FALSE, device = "cpu")
    res <- probotTrainFlow(mdl, dl, optim_adam(mdl$parameters, lr = 3e-4),
                           epochs = 3, verbose = FALSE, early_stop = FALSE,
                           lambda = lam,
                           point = if (lam > 0) "loc" else "centre")
    expect_equal(nrow(res$history), 3)
    expect_true(all(is.finite(res$history$loss)))
    expect_equal("rmse" %in% names(res$history), lam > 0)
  }

  # lambda = 0 keeps the historical (epoch, loss) schema.
  mdl <- .make_loc("nsf")
  res <- probotTrainFlow(mdl, probotDataLoader(x, th, batch = 48, shuffle = FALSE,
                                               device = "cpu"),
                         optim_adam(mdl$parameters, lr = 1e-3), epochs = 2,
                         verbose = FALSE, early_stop = FALSE)
  expect_equal(names(res$history), c("epoch", "loss"))
})

test_that("the location head costs parameters on top of the base flow", {
  plain <- probotMakeFlow(10, 4, style = "nsf", n_layers = 3, hidden_dim = 16,
                          device = "cpu")()
  headed <- probotMakeFlow(10, 4, style = "nsf", n_layers = 3, hidden_dim = 16,
                           loc_head = TRUE, loc_hidden_dims = c(8, 8),
                           device = "cpu")()
  n_plain <- sum(vapply(plain$parameters, function(p) prod(dim(p)), 0))
  n_head <- sum(vapply(headed$parameters, function(p) prod(dim(p)), 0))
  # 10->8 (80+8), 8->8 (64+8), 8->4 (32+4).
  expect_equal(n_head - n_plain, 196)
})
