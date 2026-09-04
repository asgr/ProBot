library(testthat)
library(ProBot)
library(torch)

# --- MDN training (minimal) ---
test_that("probotSingleEpochMDN runs without error", {
  set.seed(42)
  input_dim <- 3; output_dim <- 2; K <- 3
  n <- 40

  inp <- matrix(rnorm(n * input_dim), n, input_dim)
  tgt <- matrix(rnorm(n * output_dim), n, output_dim)

  mdl <- probotMakeMDN(input_dim, output_dim, K, hidden_dims = c(8, 8), device = "cpu")()
  opt <- optim_adam(mdl$parameters, lr = 1e-3)
  dl <- probotDataLoader(inp, tgt, batch = 16, device = "cpu")

  metrics <- probotSingleEpochMDN(mdl, dl, opt, K)
  expect_true("loss" %in% names(metrics))
  expect_true("mae" %in% names(metrics))
  expect_true("rmse" %in% names(metrics))
  expect_true("sigma" %in% names(metrics))
  expect_true("mix" %in% names(metrics))
  expect_true(is.finite(metrics$loss))
})

test_that("probotSingleEpochMDN with lambda > 0 works", {
  set.seed(42)
  input_dim <- 3; output_dim <- 2; K <- 3
  n <- 40

  inp <- matrix(rnorm(n * input_dim), n, input_dim)
  tgt <- matrix(rnorm(n * output_dim), n, output_dim)

  mdl <- probotMakeMDN(input_dim, output_dim, K, hidden_dims = c(8, 8), device = "cpu")()
  opt <- optim_adam(mdl$parameters, lr = 1e-3)
  dl <- probotDataLoader(inp, tgt, batch = 16, device = "cpu")

  metrics <- probotSingleEpochMDN(mdl, dl, opt, K, lambda = 0.5)
  expect_true(is.finite(metrics$loss))
})

test_that("probotSingleEpochMDN MAE/RMSE are element-wise, not row-wise", {
  set.seed(249)
  input_dim <- 3; output_dim <- 2; K <- 3; n <- 40

  inp <- matrix(rnorm(n * input_dim), n, input_dim)
  tgt <- matrix(rnorm(n * output_dim), n, output_dim)

  mdl <- probotMakeMDN(input_dim, output_dim, K, hidden_dims = c(8, 8), device = "cpu")()
  # lr = 0 freezes the model: reported metrics must equal the true
  # element-wise error of the mixture-mean predictions.
  opt <- optim_adam(mdl$parameters, lr = 0)
  dl <- probotDataLoader(inp, tgt, batch = 16, device = "cpu")

  metrics <- probotSingleEpochMDN(mdl, dl, opt, K)

  raw <- mdl(torch_tensor(inp, dtype = torch_float()))
  p <- ProBot:::.probotUnpackMDN(raw, K)
  weights <- as.array(nnf_softmax(p$logits, dim = 2))
  mu <- as.array(p$mu)
  mu_mix <- matrix(0, n, output_dim)
  for (k in 1:K) mu_mix <- mu_mix + weights[, k] * mu[, k, ]

  err <- tgt - mu_mix
  expect_equal(metrics$mae, mean(abs(err)), tolerance = 1e-5)
  expect_equal(metrics$rmse, sqrt(mean(err^2)), tolerance = 1e-5)
})

test_that("probotTrainMDN completes and returns model and history", {
  set.seed(42)
  input_dim <- 3; output_dim <- 2; K <- 3
  n <- 40

  inp <- matrix(rnorm(n * input_dim), n, input_dim)
  tgt <- matrix(rnorm(n * output_dim), n, output_dim)

  mdl <- probotMakeMDN(input_dim, output_dim, K, hidden_dims = c(8, 8), device = "cpu")()
  opt <- optim_adam(mdl$parameters, lr = 1e-3)
  dl <- probotDataLoader(inp, tgt, batch = 16, device = "cpu")

  res <- probotTrainMDN(mdl, dl, opt, epochs = 3, mdn_components = K,
                         verbose = FALSE, early_stop = FALSE)
  expect_true(!is.null(res$model))
  expect_true(!is.null(res$history))
  expect_true(is.data.frame(res$history))
  expect_true("epoch" %in% names(res$history))
  expect_true("loss" %in% names(res$history))
})

# --- Point training ---
test_that("probotSingleEpochPoint runs without error", {
  set.seed(42)
  input_dim <- 3; output_dim <- 2
  n <- 40

  inp <- matrix(rnorm(n * input_dim), n, input_dim)
  tgt <- matrix(rnorm(n * output_dim), n, output_dim)

  mdl <- probotMakePoint(input_dim, output_dim, hidden_dims = c(8, 8), device = "cpu")()
  opt <- optim_adam(mdl$parameters, lr = 1e-3)
  dl <- probotDataLoader(inp, tgt, batch = 16, device = "cpu")

  metrics <- probotSingleEpochPoint(mdl, dl, opt)
  expect_true("loss" %in% names(metrics))
  expect_true("mae" %in% names(metrics))
  expect_true("rmse" %in% names(metrics))
  expect_true(is.finite(metrics$loss))
})

test_that("probotSingleEpochPoint MAE/RMSE are element-wise, not row-wise", {
  set.seed(17)
  input_dim <- 3; output_dim <- 2; n <- 40

  inp <- matrix(rnorm(n * input_dim), n, input_dim)
  tgt <- matrix(rnorm(n * output_dim), n, output_dim)

  mdl <- probotMakePoint(input_dim, output_dim, hidden_dims = c(8, 8), device = "cpu")()
  # lr = 0 freezes the model: reported metrics must equal the true
  # element-wise error of the (unchanged) predictions. With the old row-based
  # denominator these were inflated by ncol(tgt).
  opt <- optim_adam(mdl$parameters, lr = 0)
  dl <- probotDataLoader(inp, tgt, batch = 16, device = "cpu")

  metrics <- probotSingleEpochPoint(mdl, dl, opt)

  pred <- probotPredictPoint(inp, mdl, device = "cpu")
  expect_equal(metrics$mae, mean(abs(tgt - pred)), tolerance = 1e-5)
  expect_equal(metrics$rmse, sqrt(mean((tgt - pred)^2)), tolerance = 1e-5)
})

test_that("probotTrainPoint completes and returns model and history", {
  set.seed(42)
  input_dim <- 3; output_dim <- 2
  n <- 40

  inp <- matrix(rnorm(n * input_dim), n, input_dim)
  tgt <- matrix(rnorm(n * output_dim), n, output_dim)

  mdl <- probotMakePoint(input_dim, output_dim, hidden_dims = c(8, 8), device = "cpu")()
  opt <- optim_adam(mdl$parameters, lr = 1e-3)
  dl <- probotDataLoader(inp, tgt, batch = 16, device = "cpu")

  res <- probotTrainPoint(mdl, dl, opt, epochs = 3, verbose = FALSE, early_stop = FALSE)
  expect_true(!is.null(res$model))
  expect_true(!is.null(res$history))
  expect_true(is.data.frame(res$history))
})

# --- Flow training ---
test_that("probotSingleEpochFlow runs without error", {
  set.seed(42)
  output_dim <- 4; input_dim <- 2
  n <- 32

  # For flow: input=batch[[1]]=x (context), output=batch[[2]]=theta
  x <- matrix(rnorm(n * input_dim), n, input_dim)
  theta <- matrix(rnorm(n * output_dim), n, output_dim)

  mdl <- probotMakeFlow(input_dim, output_dim, n_layers = 2, hidden_dim = 8, device = "cpu")()
  opt <- optim_adam(mdl$parameters, lr = 1e-3)
  dl <- probotDataLoader(x, theta, batch = 16, device = "cpu")

  metrics <- probotSingleEpochFlow(mdl, dl, opt)
  expect_true("loss" %in% names(metrics))
  expect_true(is.finite(metrics$loss))
})

test_that("probotTrainFlow completes and returns model and history", {
  set.seed(42)
  output_dim <- 4; input_dim <- 2
  n <- 32

  x <- matrix(rnorm(n * input_dim), n, input_dim)
  theta <- matrix(rnorm(n * output_dim), n, output_dim)

  mdl <- probotMakeFlow(input_dim, output_dim, n_layers = 2, hidden_dim = 8, device = "cpu")()
  opt <- optim_adam(mdl$parameters, lr = 1e-3)
  dl <- probotDataLoader(x, theta, batch = 16, device = "cpu")

  res <- probotTrainFlow(mdl, dl, opt, epochs = 3, verbose = FALSE, early_stop = FALSE)
  expect_true(!is.null(res$model))
  expect_true(!is.null(res$history))
  expect_true(is.data.frame(res$history))
})

test_that("probotSingleEpochFlow with lambda = 0 matches the pre-blend behaviour", {
  # shuffle = FALSE so both loaders see the same batch order; lambda = 0 must
  # not alter the numerics of a run at all.
  mk_f <- function() {
    set.seed(42); torch_manual_seed(42)
    mdl <- probotMakeFlow(2, 4, n_layers = 2, hidden_dim = 8, device = "cpu")()
    list(
      m = mdl,
      d = probotDataLoader(matrix(rnorm(32 * 2), 32, 2),
                           matrix(rnorm(32 * 4), 32, 4),
                           batch = 16, shuffle = FALSE, device = "cpu"),
      o = optim_adam(mdl$parameters, lr = 1e-3)
    )
  }

  plain <- mk_f()
  blended <- mk_f()
  res_plain <- probotSingleEpochFlow(plain$m, plain$d, plain$o)
  res_blend <- probotSingleEpochFlow(blended$m, blended$d, blended$o, lambda = 0)

  expect_identical(res_blend$loss, res_plain$loss)
  expect_equal(
    lapply(blended$m$parameters, function(p) as.numeric(as.matrix(p))),
    lapply(plain$m$parameters, function(p) as.numeric(as.matrix(p)))
  )
  # mae/rmse are only reported when the MSE term is active
  expect_named(res_blend, "loss")
})

test_that("probotSingleEpochFlow lambda > 0 adds differentiable MSE metrics", {
  for (style in c("realnvp", "maf", "nsf")) {
    set.seed(42)
    x <- matrix(rnorm(32 * 2), 32, 2)
    theta <- matrix(rnorm(32 * 4), 32, 4)
    mdl <- probotMakeFlow(2, 4, n_layers = 2, hidden_dim = 8,
                          device = "cpu", style = style)()
    dl <- probotDataLoader(x, theta, batch = 16, device = "cpu")
    opt <- optim_adam(mdl$parameters, lr = 1e-3)

    m <- probotSingleEpochFlow(mdl, dl, opt, lambda = 0.5)
    expect_named(m, c("loss", "mae", "rmse"), info = style)
    expect_true(all(is.finite(unlist(m))), info = style)

    # The MSE term must actually move parameters: gradients flow through the
    # inverse map for every style. lambda = 1 means the loss is *purely* the
    # centre-MSE term, so a style whose inverse broke the autograd graph would
    # leave every parameter untouched.
    newm <- function() {
      torch_manual_seed(99)
      probotMakeFlow(2, 4, n_layers = 2, hidden_dim = 8,
                     device = "cpu", style = style)()
    }
    untouched <- newm()
    moved <- newm()
    expect_true(identical(as.numeric(as.matrix(untouched$parameters[[1]])),
                          as.numeric(as.matrix(moved$parameters[[1]]))))
    dl2 <- probotDataLoader(x, theta, batch = 32, shuffle = FALSE, device = "cpu")
    probotSingleEpochFlow(moved, dl2, optim_adam(moved$parameters, lr = 1e-2),
                          lambda = 1)
    changed <- vapply(seq_along(moved$parameters), function(i) {
      a <- as.matrix(untouched$parameters[[i]])
      b <- as.matrix(moved$parameters[[i]])
      any(a != b)
    }, logical(1))
    expect_true(all(changed), info = style)
  }
})

test_that("probotTrainFlow passes lambda through to the epoch function", {
  set.seed(42)
  x <- matrix(rnorm(32 * 2), 32, 2)
  theta <- matrix(rnorm(32 * 4), 32, 4)
  mdl <- probotMakeFlow(2, 4, n_layers = 2, hidden_dim = 8, device = "cpu")()
  dl <- probotDataLoader(x, theta, batch = 16, device = "cpu")
  opt <- optim_adam(mdl$parameters, lr = 1e-3)

  res <- probotTrainFlow(mdl, dl, opt, epochs = 3, lambda = 0.3,
                         verbose = FALSE, early_stop = FALSE)
  expect_true(all(c("mae", "rmse") %in% names(res$history)))
  expect_equal(nrow(res$history), 3)
})

test_that("probotSingleEpochFlow rejects invalid lambda", {
  set.seed(42)
  mdl <- probotMakeFlow(2, 4, n_layers = 2, hidden_dim = 8, device = "cpu")()
  dl <- probotDataLoader(matrix(rnorm(32 * 2), 32, 2),
                         matrix(rnorm(32 * 4), 32, 4),
                         batch = 16, device = "cpu")
  opt <- optim_adam(mdl$parameters, lr = 1e-3)

  expect_error(probotSingleEpochFlow(mdl, dl, opt, lambda = -0.1), "'lambda'")
  expect_error(probotSingleEpochFlow(mdl, dl, opt, lambda = 1.5), "'lambda'")
  expect_error(probotSingleEpochFlow(mdl, dl, opt, lambda = c(0.2, 0.3)), "'lambda'")
  expect_error(probotSingleEpochFlow(mdl, dl, opt, lambda = NA_real_), "'lambda'")
})

# --- point estimator -------------------------------------------------------
test_that(".probotFlowPointEstimate groups draws by observation, not by draw", {
  # Echo stub: returns the context it was fed. A correct tiling means each
  # group of n_point_samples rows shares one context row, so the per-group
  # mean must reproduce the input exactly. A wrong grouping cannot.
  n_obs <- 4; m <- 5; od <- 3
  ctx <- torch_tensor(matrix(rnorm(n_obs * od), n_obs, od))
  seen <- NULL
  stub <- list(inverse = function(z, x) { seen <<- x; x })

  est <- .probotFlowPointEstimate(stub, ctx, output_dim = od,
                                  point = "mean", n_point_samples = m)
  expect_equal(as.matrix(est), as.matrix(ctx), tolerance = 1e-6)
  expect_equal(est$shape, c(n_obs, od))

  # Negative control: grouping along the other axis must NOT reproduce ctx.
  wrong <- seen$reshape(c(m, n_obs, od))$mean(dim = 1)
  expect_false(isTRUE(all.equal(as.matrix(wrong), as.matrix(ctx), tolerance = 1e-6)))
})

test_that("point = 'centre' matches inlining model$inverse(zeros) directly", {
  torch_manual_seed(21)
  mdl <- probotMakeFlow(2, 4, n_layers = 2, hidden_dim = 8, device = "cpu")()
  ctx <- torch_randn(c(6, 2))
  direct <- mdl$inverse(torch_zeros(c(6, 4)), ctx)
  helper <- .probotFlowPointEstimate(mdl, ctx, output_dim = 4, point = "centre")
  expect_equal(as.matrix(helper), as.matrix(direct), tolerance = 0)
})

test_that("point = 'mean' concentrates on the posterior mean as m grows", {
  # Statistical property of the estimator, tested with common random numbers
  # so it is a paired comparison rather than a race against measurement noise.
  # One pool of draws per observation is averaged over nested subsets; the
  # deviation of a length-m partial mean from the full-pool mean scales as
  # sqrt(1/m - 1/M), so m = 32 should sit roughly 4x closer than m = 2.
  torch_manual_seed(7)
  mdl <- probotMakeFlow(2, 3, n_layers = 2, hidden_dim = 16, device = "cpu")()
  xt <- torch_tensor(matrix(rnorm(16 * 2), 16, 2))
  od <- 3; M <- 2048L; n_obs <- 16L

  # Single pool, tiled exactly as the estimator tiles it, so subset means line
  # up with observations.
  z <- torch_randn(c(n_obs * M, od))
  ctx <- xt$unsqueeze(2)$expand(c(n_obs, M, 2))$reshape(c(n_obs * M, 2))
  pool <- with_no_grad(mdl$inverse(z, ctx))$reshape(c(n_obs, M, od))

  dev <- function(m) {
    sub <- pool$narrow(2, 1, m)$mean(dim = 2)
    mean(abs(as.matrix(sub - pool$mean(dim = 2))))
  }
  d_small <- dev(2L)
  d_large <- dev(32L)

  expect_lt(d_large, d_small)
  # Theory ratio is sqrt((1/2 - 1/M) / (1/32 - 1/M)) ~= 4. Allow a generous
  # band: this fails only if averaging is not actually reducing variance.
  ratio <- d_small / d_large
  expect_gt(ratio, 1.5)
  expect_lt(ratio, 12)
})

test_that("point = 'centre' is deterministic; 'mean' is not, and they differ", {
  # 'centre' inverts a fixed z = 0, so repeated calls must agree exactly.
  # 'mean' draws, so it moves run to run. The two must also disagree with each
  # other -- otherwise the mean branch is silently aliasing the centre branch.
  torch_manual_seed(7)
  mdl <- probotMakeFlow(2, 3, n_layers = 2, hidden_dim = 16, device = "cpu")()
  xt <- torch_tensor(matrix(rnorm(16 * 2), 16, 2))

  c1 <- as.matrix(with_no_grad(.probotFlowPointEstimate(mdl, xt, 3, point = "centre")))
  c2 <- as.matrix(with_no_grad(.probotFlowPointEstimate(mdl, xt, 3, point = "centre")))
  expect_identical(c1, c2)

  m1 <- as.matrix(with_no_grad(
    .probotFlowPointEstimate(mdl, xt, 3, point = "mean", n_point_samples = 64)))
  m2 <- as.matrix(with_no_grad(
    .probotFlowPointEstimate(mdl, xt, 3, point = "mean", n_point_samples = 64)))
  expect_false(identical(m1, m2))
  expect_equal(dim(m1), dim(c1))
  expect_false(isTRUE(all.equal(c1, m1, tolerance = 1e-3)))
})

test_that("point = 'mean' trains without error for all styles", {
  for (style in c("realnvp", "maf", "nsf")) {
    set.seed(42)
    x <- matrix(rnorm(32 * 2), 32, 2)
    theta <- matrix(rnorm(32 * 4), 32, 4)
    mdl <- probotMakeFlow(2, 4, n_layers = 2, hidden_dim = 8,
                          device = "cpu", style = style)()
    dl <- probotDataLoader(x, theta, batch = 16, shuffle = FALSE, device = "cpu")
    opt <- optim_adam(mdl$parameters, lr = 1e-3)

    before <- lapply(mdl$parameters, function(p) as.matrix(p))
    m <- probotSingleEpochFlow(mdl, dl, opt, lambda = 0.5,
                               point = "mean", n_point_samples = 8)
    expect_named(m, c("loss", "mae", "rmse"), info = style)
    expect_true(all(is.finite(unlist(m))), info = style)
    # Gradients must flow through the mean-of-draws map, so params move.
    expect_false(isTRUE(all.equal(before,
                                  lapply(mdl$parameters, function(p) as.matrix(p)))),
                 info = style)
  }
})

test_that("probotTrainFlow passes point and n_point_samples through", {
  set.seed(42)
  x <- matrix(rnorm(32 * 2), 32, 2)
  theta <- matrix(rnorm(32 * 4), 32, 4)
  mdl <- probotMakeFlow(2, 4, n_layers = 2, hidden_dim = 8, device = "cpu")()
  dl <- probotDataLoader(x, theta, batch = 16, device = "cpu")
  opt <- optim_adam(mdl$parameters, lr = 1e-3)

  res <- probotTrainFlow(mdl, dl, opt, epochs = 2, lambda = 0.5,
                         point = "mean", n_point_samples = 4,
                         verbose = FALSE, early_stop = FALSE)
  expect_true(all(c("mae", "rmse") %in% names(res$history)))
  expect_error(probotTrainFlow(mdl, dl, opt, epochs = 1, lambda = 0.5,
                               point = "median", verbose = FALSE))
  expect_error(probotSingleEpochFlow(mdl, dl, opt, lambda = 0.5, point = "nope"))
})

test_that("n_point_samples is validated only when it is used", {
  set.seed(42)
  x <- matrix(rnorm(32 * 2), 32, 2)
  theta <- matrix(rnorm(32 * 4), 32, 4)
  mdl <- probotMakeFlow(2, 4, n_layers = 2, hidden_dim = 8, device = "cpu")()
  dl <- probotDataLoader(x, theta, batch = 16, device = "cpu")
  opt <- optim_adam(mdl$parameters, lr = 1e-3)

  # Irrelevant at lambda = 0, so must not error
  expect_no_error(probotSingleEpochFlow(mdl, dl, opt, lambda = 0,
                                        n_point_samples = -5))
  expect_error(probotSingleEpochFlow(mdl, dl, opt, lambda = 0.5,
                                     n_point_samples = 0), "'n_point_samples'")
  expect_error(probotSingleEpochFlow(mdl, dl, opt, lambda = 0.5,
                                     n_point_samples = 2.5), "'n_point_samples'")
})

test_that("point defaults to centre, keeping prior behaviour", {
  expect_equal(formals(probotSingleEpochFlow)$point, "centre")
  expect_equal(formals(probotTrainFlow)$point, "centre")
  expect_equal(formals(probotTrainFlow)$lambda, 0)
})
