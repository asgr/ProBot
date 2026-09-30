library(testthat)
library(ProBot)
library(torch)

# Helpers ---------------------------------------------------------------

# A point net trained on pure noise cannot fit the holdout, and one trained on
# a *clean* learnable signal drives both losses down together, so neither gives
# an overfitting signal. The recipe below is learnable signal plus per-row
# irreducible noise: the net memorises the noise on the training rows, the
# training loss keeps falling, and the holdout loss turns upward. That
# asymmetry is the thing the stop rule claims to detect.
.make_overfit_data <- function(n = 80, input_dim = 4) {
  set.seed(99)
  x <- matrix(rnorm(n * input_dim), n, input_dim)
  theta <- cbind(0.8 * x[, 1], 0.8 * x[, 2]) +
    matrix(rnorm(n * 2, sd = 0.5), n, 2)
  list(x = x, theta = theta, input_dim = input_dim, output_dim = 2)
}

# Score the model by hand on named rows, as an independent check on the
# loop's holdout column.
.reference_val_loss <- function(model, x, theta, idx, device = "cpu") {
  probotLossEval(x[idx, , drop = FALSE], theta[idx, , drop = FALSE], model,
                 device = device)$loss
}

# .probotSplitLoaders ---------------------------------------------------

test_that("the split is disjoint, complete and keeps row contents", {
  set.seed(1)
  n <- 50
  x <- matrix(rnorm(n * 3), n, 3)
  theta <- matrix(rnorm(n * 2), n, 2)
  dl <- probotDataLoader(x, theta, batch = 16, shuffle = TRUE, device = "cpu")

  sp <- ProBot:::.probotSplitLoaders(dl, 0.2, seed = 7)

  expect_equal(sp$n_train + sp$n_val, n)
  expect_equal(sp$n_val, 10L)
  expect_false(any(intersect(sp$train_idx, sp$val_idx)))
  expect_equal(sort(c(sp$train_idx, sp$val_idx)), seq_len(n))
  # Row order inside each half is preserved, so the split is auditable.
  expect_equal(sp$val_idx, sort(sp$val_idx))
  expect_equal(sp$train_idx, sort(sp$train_idx))

  # Tensors on both sides really are the rows they claim (float32 round-trip).
  xt <- as.matrix(sp$val$dataset$tensors[[1]]$to(device = "cpu"))
  tht <- as.matrix(sp$val$dataset$tensors[[2]]$to(device = "cpu"))
  expect_equal(xt, x[sp$val_idx, , drop = FALSE], tolerance = 1e-6)
  expect_equal(tht, theta[sp$val_idx, , drop = FALSE], tolerance = 1e-6)

  xtr <- as.matrix(sp$train$dataset$tensors[[1]]$to(device = "cpu"))
  expect_equal(xtr, x[sp$train_idx, , drop = FALSE], tolerance = 1e-6)
})

test_that("the holdout loader is deterministic and the training one is not", {
  set.seed(1)
  n <- 60
  dl <- probotDataLoader(matrix(rnorm(n * 3), n, 3), matrix(rnorm(n * 2), n, 2),
                         batch = 16, shuffle = TRUE, device = "cpu")
  sp <- ProBot:::.probotSplitLoaders(dl, 0.1, seed = 3)

  # The verification set must replay the same rows in the same order every
  # epoch, otherwise the cross-epoch trend the stop rule reads is noise.
  first <- NULL
  for (rep in 1:3) {
    seen <- NULL
    coro::loop(for (b in sp$val) seen <- c(seen, as.numeric(b[[1]]$select(1, 1))))
    if (is.null(first)) first <- seen else expect_identical(first, seen)
  }
  expect_true(inherits(sp$train$sampler, "utils_sampler_random"))
  expect_false(inherits(sp$val$sampler, "utils_sampler_random"))
})

test_that("the training loader inherits batch size and shuffle setting", {
  n <- 40
  x <- matrix(rnorm(n * 3), n, 3)
  theta <- matrix(rnorm(n * 2), n, 2)

  shuffled <- probotDataLoader(x, theta, batch = 7, shuffle = TRUE, device = "cpu")
  sp <- ProBot:::.probotSplitLoaders(shuffled, 0.25)
  expect_equal(sp$train$batch_size, 7)
  expect_true(inherits(sp$train$sampler, "utils_sampler_random"))

  plain <- probotDataLoader(x, theta, batch = 7, shuffle = FALSE, device = "cpu")
  sp2 <- ProBot:::.probotSplitLoaders(plain, 0.25)
  expect_false(inherits(sp2$train$sampler, "utils_sampler_random"))
})

test_that("an extreme holdout fraction still leaves a row on each side", {
  n <- 50
  dl <- probotDataLoader(matrix(rnorm(n * 3), n, 3), matrix(rnorm(n * 2), n, 2),
                         batch = 8, device = "cpu")
  lo <- ProBot:::.probotSplitLoaders(dl, 0.0001)
  hi <- ProBot:::.probotSplitLoaders(dl, 0.9999)
  expect_gte(lo$n_val, 1L)
  expect_gte(lo$n_train, 1L)
  expect_gte(hi$n_val, 1L)
  expect_gte(hi$n_train, 1L)
  expect_equal(lo$n_train + lo$n_val, n)
  expect_equal(hi$n_train + hi$n_val, n)
})

test_that("split_seed makes the split reproducible without stealing the RNG", {
  n <- 50
  dl <- probotDataLoader(matrix(rnorm(n * 3), n, 3), matrix(rnorm(n * 2), n, 2),
                         batch = 8, device = "cpu")

  a <- ProBot:::.probotSplitLoaders(dl, 0.2, seed = 42)$val_idx
  b <- ProBot:::.probotSplitLoaders(dl, 0.2, seed = 42)$val_idx
  expect_identical(a, b)
  # An unseeded split is a fresh draw, and must not depend on the seeded one.
  c1 <- ProBot:::.probotSplitLoaders(dl, 0.2)$val_idx
  expect_false(identical(a, c1))

  # An explicit seed must leave the caller's stream alone, so the split is not
  # silently a source of irreproducibility in the rest of a script.
  set.seed(5)
  seed_before <- get(".Random.seed", envir = globalenv())
  invisible(ProBot:::.probotSplitLoaders(dl, 0.2, seed = 99))
  expect_identical(get(".Random.seed", envir = globalenv()), seed_before)
  expect_identical(runif(1), suppressWarnings({set.seed(5); runif(1)}))

  # No seed at all: the split consumes the stream, as any sampling should.
  set.seed(5)
  invisible(ProBot:::.probotSplitLoaders(dl, 0.2))
  expect_false(identical(get(".Random.seed", envir = globalenv()), seed_before))
})

# .probotTrend / .probotHoldoutStop -------------------------------------

test_that(".probotTrend returns the per-epoch slope of a straight line", {
  v <- 10 - 0.25 * (1:30)
  expect_equal(ProBot:::.probotTrend(v, 20), -0.25, tolerance = 1e-12)
  # Window length must not rescale the answer, only re-anchor it.
  expect_equal(ProBot:::.probotTrend(v, 5), -0.25, tolerance = 1e-12)
  expect_equal(ProBot:::.probotTrend(rep(3, 10), 4), 0)
  expect_true(is.na(ProBot:::.probotTrend(1, 4)))
  # tail() silently shortens, so a request longer than the series is not an
  # error; and a non-finite value must yield NA rather than a NaN slope.
  expect_equal(ProBot:::.probotTrend(c(1, 2, 3), 20), 1)
  expect_true(is.na(ProBot:::.probotTrend(c(1, 2, NaN, 4), 3)))
})

test_that("the stop rule fires only for the asymmetric overfitting case", {
  # train falling, verify rising -> stop after `patience` epochs
  tr <- 5 - 0.05 * (1:20)
  va <- 3 + 0.02 * (1:20)
  st <- 0L
  fired <- NA_integer_
  for (e in 6:20) {
    hs <- ProBot:::.probotHoldoutStop(tr[seq_len(e)], va[seq_len(e)],
                                      window = 6, min_delta = 1e-3,
                                      train_trend_epsilon = 0.005,
                                      patience = 3L, state = st)
    st <- hs$state
    if (hs$stop) { fired <- e; break }
  }
  expect_equal(fired, 8L)
  expect_true(hs$train_trend < 0)
  expect_true(hs$val_trend > 0)

  # both falling -> never flag
  hs2 <- ProBot:::.probotHoldoutStop(tr, 3 - 0.05 * (1:20), window = 6,
                                     min_delta = 1e-3,
                                     train_trend_epsilon = 0.005,
                                     patience = 3L, state = 0L)
  expect_false(hs2$stop)
  expect_equal(hs2$state, 0L)

  # train flat, verify rising -> that is stagnation, not over-fitting, so the
  # plateau rule owns it and this one must stay quiet
  hs3 <- ProBot:::.probotHoldoutStop(rep(4, 20), va, window = 6,
                                     min_delta = 1e-3,
                                     train_trend_epsilon = 0.005,
                                     patience = 3L, state = 0L)
  expect_false(hs3$stop)
  expect_equal(hs3$state, 0L)

  # a single bad epoch must not end the run: the counter resets
  hs4 <- ProBot:::.probotHoldoutStop(tr[1:20], c(va[1:19], va[19] - 0.6),
                                     window = 6, min_delta = 1e-3,
                                     train_trend_epsilon = 0.005,
                                     patience = 3L, state = 2L)
  expect_equal(hs4$state, 0L)
  expect_false(hs4$stop)
})

test_that("a NaN trend neither stops nor throws", {
  # The documented failure mode of the old plateau rule was
  # "missing value where TRUE/FALSE needed" on a NaN epoch. The holdout rule
  # must degrade to "keep going" instead.
  hs <- ProBot:::.probotHoldoutStop(c(1, 2, NaN, 4), 1:4, window = 4,
                                    min_delta = 1e-3, train_trend_epsilon = 0.01,
                                    patience = 1L, state = 0L)
  expect_false(hs$stop)
  expect_equal(hs$state, 0L)
})

# Trainer integration ---------------------------------------------------

test_that("a holdout split is on by default and adds val_loss to history", {
  d <- .make_overfit_data()
  dl <- probotDataLoader(d$x, d$theta, batch = 16, device = "cpu")
  mdl <- probotMakePoint(4, 2, hidden_dims = c(16, 16), device = "cpu")()
  res <- probotTrainPoint(mdl, dl, optim_adam(mdl$parameters, lr = 1e-2),
                          epochs = 4, verbose = FALSE, early_stop = FALSE,
                          split_seed = 11)

  expect_true("val_loss" %in% names(res$history))
  expect_equal(nrow(res$history), 4L)
  expect_true(all(res$history$val_loss > 0))
  # 10% of the rows held out
  n <- nrow(d$x)
  expect_equal(res$validation$n_verify, as.integer(round(n * 0.1)))
  expect_equal(res$validation$n_train, n - res$validation$n_verify)
  expect_equal(res$validation$source, "holdout_fraction")
  expect_equal(res$validation$holdout_fraction, 0.1)
  expect_length(res$validation$verify_idx, res$validation$n_verify)
})

test_that("the holdout column is exactly the loss on the reported rows", {
  d <- .make_overfit_data()
  dl <- probotDataLoader(d$x, d$theta, batch = 16, device = "cpu")
  mdl <- probotMakePoint(4, 2, hidden_dims = c(16, 16), device = "cpu")()
  res <- probotTrainPoint(mdl, dl, optim_adam(mdl$parameters, lr = 1e-3),
                          epochs = 2, verbose = FALSE, early_stop = FALSE,
                          split_seed = 11)
  idx <- res$validation$verify_idx
  # probotLossEval() is an independent scoring path (R matrices, its own
  # batching), so agreement is a real cross-check rather than a tautology.
  expect_equal(tail(res$history$val_loss, 1L),
               .reference_val_loss(mdl, d$x, d$theta, idx),
               tolerance = 1e-5)
})

test_that("the model trains on the training rows only", {
  # lr = 0 freezes the net, so the epoch loss must equal an independent scoring
  # of exactly the training rows -- and a *different* number from scoring all
  # rows. A silently ignored split makes those two equal, which is what the
  # second expectation catches.
  set.seed(9)
  n <- 100
  x <- matrix(rnorm(n * 3), n, 3)
  theta <- matrix(rnorm(n * 2), n, 2)
  dl <- probotDataLoader(x, theta, batch = 16, shuffle = FALSE, device = "cpu")

  mdl <- probotMakePoint(3, 2, hidden_dims = c(16, 16), device = "cpu")()
  res <- probotTrainPoint(mdl, dl, optim_adam(mdl$parameters, lr = 0),
                          epochs = 1, verbose = FALSE, early_stop = FALSE,
                          split_seed = 11)
  train_rows <- setdiff(seq_len(n), res$validation$verify_idx)
  expect_equal(length(train_rows), res$validation$n_train)

  on_train <- .reference_val_loss(mdl, x, theta, train_rows)
  on_all <- probotLossEval(x, theta, mdl, device = "cpu")$loss
  expect_equal(res$history$loss[1], on_train, tolerance = 1e-5)
  expect_false(isTRUE(all.equal(on_train, on_all, tolerance = 1e-3)))
  # And n_train + n_verify is the original dataset, i.e. nothing was dropped.
  expect_equal(res$validation$n_train + res$validation$n_verify, n)
})

test_that("holdout_fraction = 0 reproduces the pre-split behaviour", {
  d <- .make_overfit_data()
  dl <- probotDataLoader(d$x, d$theta, batch = 16, device = "cpu")
  mdl <- probotMakePoint(4, 2, hidden_dims = c(16, 16), device = "cpu")()
  res <- probotTrainPoint(mdl, dl, optim_adam(mdl$parameters, lr = 1e-3),
                          epochs = 3, verbose = FALSE, early_stop = FALSE,
                          holdout_fraction = 0)
  expect_equal(names(res$history), c("epoch", "loss", "mae", "rmse"))
  expect_null(res$validation)
})

test_that("an explicit val_dataloader overrides the split and is used as given", {
  d <- .make_overfit_data()
  dl <- probotDataLoader(d$x, d$theta, batch = 16, device = "cpu")
  # Deliberately overlapping rows: an explicit loader means "score these".
  dv <- probotDataLoader(d$x[1:25, , drop = FALSE], d$theta[1:25, , drop = FALSE],
                         batch = 50, shuffle = FALSE, device = "cpu")
  mdl <- probotMakePoint(4, 2, hidden_dims = c(16, 16), device = "cpu")()
  res <- probotTrainPoint(mdl, dl, optim_adam(mdl$parameters, lr = 1e-3),
                          epochs = 2, verbose = FALSE, early_stop = FALSE,
                          val_dataloader = dv)
  expect_equal(res$validation$n_verify, 25L)
  expect_equal(res$validation$n_train, nrow(d$x))
  expect_equal(res$validation$source, "val_dataloader")
  expect_null(res$validation$verify_idx)
  # No split happened, so training still saw all rows.
  expect_equal(tail(res$history$val_loss, 1L),
               .reference_val_loss(mdl, d$x, d$theta, 1:25), tolerance = 1e-5)
})

test_that("validation arguments are rejected before any epoch runs", {
  d <- .make_overfit_data()
  dl <- probotDataLoader(d$x, d$theta, batch = 16, device = "cpu")
  mdl <- probotMakePoint(4, 2, hidden_dims = c(16, 16), device = "cpu")()
  opt <- optim_adam(mdl$parameters, lr = 1e-3)

  expect_error(probotTrainPoint(mdl, dl, opt, epochs = 1,
                                holdout_fraction = 1), "'holdout_fraction'")
  expect_error(probotTrainPoint(mdl, dl, opt, epochs = 1,
                                holdout_fraction = -0.1), "'holdout_fraction'")
  expect_error(probotTrainPoint(mdl, dl, opt, epochs = 1,
                                holdout_fraction = NA_real_), "'holdout_fraction'")
  expect_error(probotTrainPoint(mdl, dl, opt, epochs = 1,
                                holdout_fraction = c(0.1, 0.2)), "'holdout_fraction'")
  expect_error(probotTrainPoint(mdl, dl, opt, epochs = 1,
                                val_dataloader = "nope"), "'val_dataloader'")
  # The dataloader must actually be a loader, not a dataset
  expect_error(probotTrainPoint(mdl, d$x, opt, epochs = 1), "dataloader",
               fixed = TRUE)
})

test_that("a dataset too small to trend warns and trains on everything", {
  set.seed(4)
  x <- matrix(rnorm(5 * 3), 5, 3)
  theta <- matrix(rnorm(5 * 2), 5, 2)
  dl <- probotDataLoader(x, theta, batch = 2, device = "cpu")
  mdl <- probotMakePoint(3, 2, hidden_dims = c(8, 8), device = "cpu")()

  expect_warning(res <- probotTrainPoint(mdl, dl,
                                         optim_adam(mdl$parameters, lr = 1e-3),
                                         epochs = 2, verbose = FALSE,
                                         early_stop = FALSE),
                 "too small")
  expect_equal(names(res$history), c("epoch", "loss", "mae", "rmse"))
  expect_null(res$validation)
})

test_that("holdout_stop = FALSE reports the verification loss but never stops", {
  d <- .make_overfit_data()
  dl <- probotDataLoader(d$x, d$theta, batch = 16, device = "cpu")

  run <- function(stop_flag, pat = 5L, es = TRUE) {
    mdl <- probotMakePoint(4, 2, hidden_dims = c(64, 64), device = "cpu")()
    probotTrainPoint(mdl, dl, optim_adam(mdl$parameters, lr = 3e-3),
                     epochs = 60, verbose = FALSE,
                     # stop_delta tiny so the plateau rule cannot be what ends
                     # this run; only the holdout rule can.
                     early_stop = es, stop_window = 20, stop_delta = 1e-12,
                     holdout_stop = stop_flag, holdout_patience = pat,
                     split_seed = 5)
  }
  off <- run(FALSE)
  expect_equal(nrow(off$history), 60L)
  expect_null(off$validation$stop_reason)
  # The scenario must actually overfit, or the positive control below is
  # vacuous: the best epoch has to sit well before the end, the final holdout
  # loss be worse than that best, and the training loss still be falling.
  expect_lt(off$validation$best_epoch, 30L)
  expect_gt(tail(off$history$val_loss, 1L), off$validation$best_val_loss)
  expect_lt(tail(off$history$loss, 1L), head(off$history$loss, 1L))

  on <- run(TRUE)
  # Same seed, so the only difference is the rule: it must cut the run short.
  expect_lt(nrow(on$history), nrow(off$history))
  expect_equal(on$validation$stop_reason, "holdout_overfit")
  expect_true(on$validation$best_epoch < nrow(on$history))

  # patience trades run length against tolerance
  expect_lt(nrow(run(TRUE, pat = 1L)$history), nrow(run(TRUE, pat = 20L)$history))
  # ...and early_stop = FALSE overrides both rules
  expect_equal(nrow(run(TRUE, pat = 1L, es = FALSE)$history), 60L)
})

test_that("the plateau rule still reports its own reason", {
  d <- .make_overfit_data()
  dl <- probotDataLoader(d$x, d$theta, batch = 16, device = "cpu")
  mdl <- probotMakePoint(4, 2, hidden_dims = c(16, 16), device = "cpu")()
  res <- probotTrainPoint(mdl, dl, optim_adam(mdl$parameters, lr = 1e-6),
                          epochs = 40, verbose = FALSE,
                          early_stop = TRUE, stop_window = 5, stop_delta = 1e-3,
                          holdout_stop = FALSE)
  expect_equal(res$validation$stop_reason, "train_loss_plateau")
  expect_lt(nrow(res$history), 40L)
})

test_that("every model type scores its holdout, blended or not", {
  d <- .make_overfit_data()
  dl <- probotDataLoader(d$x, d$theta, batch = 16, device = "cpu")

  # Point
  mp <- probotMakePoint(4, 2, hidden_dims = c(16, 16), device = "cpu")()
  rp <- probotTrainPoint(mp, dl, optim_adam(mp$parameters, lr = 1e-3),
                         epochs = 2, verbose = FALSE, early_stop = FALSE)
  expect_true("val_loss" %in% names(rp$history))
  expect_true(all(is.finite(rp$history$val_loss)))

  # MDN
  mm <- probotMakeMDN(4, 2, 3, hidden_dims = c(16, 16), device = "cpu")()
  rm_ <- probotTrainMDN(mm, dl, optim_adam(mm$parameters, lr = 1e-3),
                        epochs = 2, verbose = FALSE, early_stop = FALSE)
  expect_true("val_loss" %in% names(rm_$history))
  expect_true(all(is.finite(rm_$history$val_loss)))

  # Every flow style, plain and blended
  for (style in c("realnvp", "maf", "nsf")) {
    mf <- probotMakeFlow(4, 2, n_layers = 2, hidden_dim = 16, style = style,
                         device = "cpu")()
    rf <- probotTrainFlow(mf, dl, optim_adam(mf$parameters, lr = 1e-3),
                          epochs = 2, verbose = FALSE, early_stop = FALSE)
    expect_true("val_loss" %in% names(rf$history), info = style)
    expect_true(all(is.finite(rf$history$val_loss)), info = style)
    # unblended: loss only, no verify errors
    expect_false(any(c("val_mae", "val_rmse") %in% names(rf$history)),
                 info = style)

    mg <- probotMakeFlow(4, 2, n_layers = 2, hidden_dim = 16, style = style,
                         device = "cpu")()
    rg <- probotTrainFlow(mg, dl, optim_adam(mg$parameters, lr = 1e-3),
                          epochs = 2, lambda = 0.5, verbose = FALSE,
                          early_stop = FALSE)
    expect_true(all(c("val_loss", "val_mae", "val_rmse") %in%
                      names(rg$history)), info = style)
    expect_true(all(is.finite(unlist(rg$history[c("val_loss", "val_mae",
                                                  "val_rmse")]))), info = style)
  }
})

test_that("the holdout mirrors the lambda blend, not the pure loss", {
  # With lambda > 0 the trainer optimises the blend. If the holdout were scored
  # with the unblended loss, the two history columns would measure different
  # functionals and the stop rule would compare trends that are not comparable.
  d <- .make_overfit_data()
  dl <- probotDataLoader(d$x, d$theta, batch = 16, device = "cpu")
  mdl <- probotMakeMDN(4, 2, 3, hidden_dims = c(16, 16), device = "cpu")()
  res <- probotTrainMDN(mdl, dl, optim_adam(mdl$parameters, lr = 1e-3),
                        epochs = 1, lambda = 0.7, verbose = FALSE,
                        early_stop = FALSE, split_seed = 21)
  idx <- res$validation$verify_idx
  pure <- probotLossEval(d$x[idx, , drop = FALSE], d$theta[idx, , drop = FALSE],
                         mdl, device = "cpu")$loss
  blended <- res$history$val_loss[1]

  # Recompute the blend by hand from the model's own mixture mean.
  raw <- mdl(torch_tensor(d$x[idx, ], dtype = torch_float(), device = "cpu"))
  p <- ProBot:::.probotUnpackMDN(raw, mdl)
  w <- as.matrix(nnf_softmax(p$logits, dim = 2)$to(device = "cpu"))
  mu <- as.array(p$mu$to(device = "cpu"))
  mu_mix <- matrix(0, length(idx), 2)
  for (k in 1:3) mu_mix <- mu_mix + w[, k] * mu[, k, ]
  mse <- mean((d$theta[idx, ] - mu_mix)^2)

  expect_equal(blended, 0.3 * pure + 0.7 * mse, tolerance = 1e-4)
  expect_false(isTRUE(all.equal(blended, pure)))
})

test_that("verbose output names both the training and verification loss", {
  d <- .make_overfit_data()
  dl <- probotDataLoader(d$x, d$theta, batch = 16, device = "cpu")
  mdl <- probotMakePoint(4, 2, hidden_dims = c(16, 16), device = "cpu")()

  out <- capture.output(
    res <- probotTrainPoint(mdl, dl, optim_adam(mdl$parameters, lr = 1e-3),
                            epochs = 3, verbose = TRUE, early_stop = FALSE,
                            checkpoint_every = 1, split_seed = 11))
  txt <- paste(out, collapse = "\n")
  expect_match(txt, sprintf("Split %d train / %d verify rows",
                            nrow(d$x) - 8L, 8L))
  expect_match(txt, "Train Loss")
  expect_match(txt, "Verify Loss")
  # The printed verify number is the one in the history, not a re-computation.
  # out[1] is the "Split ..." banner, so epoch 1 is out[2].
  expect_match(out[2], sprintf("Verify Loss %.6f", res$history$val_loss[1]),
               fixed = TRUE)

  # Without a split, the line keeps its old single-loss shape.
  out0 <- capture.output(probotTrainPoint(mdl, dl,
                                          optim_adam(mdl$parameters, lr = 1e-3),
                                          epochs = 1, verbose = TRUE,
                                          early_stop = FALSE,
                                          holdout_fraction = 0))
  expect_match(paste(out0, collapse = "\n"), "Train Loss")
  expect_failure(expect_match(paste(out0, collapse = "\n"), "Verify",
                              fixed = TRUE))
})

test_that("MDN verbose precision and metrics survive the split", {
  d <- .make_overfit_data()
  dl <- probotDataLoader(d$x, d$theta, batch = 16, device = "cpu")
  mdl <- probotMakeMDN(4, 2, 3, hidden_dims = c(16, 16), device = "cpu")()
  out <- capture.output(probotTrainMDN(mdl, dl, optim_adam(mdl$parameters,
                                                           lr = 1e-3),
                                       epochs = 1, verbose = TRUE,
                                       early_stop = FALSE, checkpoint_every = 1))
  expect_match(out[2], "^Epoch 1 Train Loss [0-9]+\\.[0-9]{3} ")
  expect_match(out[2], "MAE .* RMSE .* Sigma .* MixSD .* Mix \\[")
  expect_match(out[2], "Verify Loss")
  # lambda = 0 has no point estimate, so there is no verification MAE to show.
  expect_failure(expect_match(out[2], "Verify MAE"))

  mdl2 <- probotMakeMDN(4, 2, 3, hidden_dims = c(16, 16), device = "cpu")()
  out2 <- capture.output(probotTrainMDN(mdl2, dl, optim_adam(mdl2$parameters,
                                                            lr = 1e-3),
                                        epochs = 1, lambda = 0.5, verbose = TRUE,
                                        early_stop = FALSE,
                                        checkpoint_every = 1))
  expect_match(out2[2], "Verify MAE")
})

test_that("training mode is restored after the holdout pass", {
  d <- .make_overfit_data()
  dl <- probotDataLoader(d$x, d$theta, batch = 16, device = "cpu")
  # Dropout makes eval()/train() observable: the same forward differs between
  # modes, so a model left in eval() would be detectable.
  mdl <- probotMakeMDN(4, 2, 3, hidden_dims = c(16, 16), dropout = 0.5,
                       device = "cpu")()
  res <- probotTrainMDN(mdl, dl, optim_adam(mdl$parameters, lr = 1e-3),
                        epochs = 2, verbose = FALSE, early_stop = FALSE)
  expect_true(mdl$training)
  expect_true(all(is.finite(res$history$val_loss)))
})

test_that("checkpointing still writes to disk with a holdout active", {
  d <- .make_overfit_data()
  dl <- probotDataLoader(d$x, d$theta, batch = 16, device = "cpu")
  mdl <- probotMakePoint(4, 2, hidden_dims = c(8, 8), device = "cpu")()
  dir <- file.path(tempdir(), paste0("probot_ckpt_", as.integer(Sys.time())))
  dir.create(dir, showWarnings = FALSE)
  on.exit(unlink(dir, recursive = TRUE), add = TRUE)

  probotTrainPoint(mdl, dl, optim_adam(mdl$parameters, lr = 1e-3), epochs = 4,
                   verbose = FALSE, early_stop = FALSE, checkpoint_dir = dir,
                   checkpoint_every = 2)
  expect_setequal(list.files(dir),
                  c("point_epoch_002.pt", "point_epoch_004.pt"))
})

test_that("the holdout costs the reported rows and nothing else", {
  # A 10% holdout must not be quietly scoring the training rows too: the verify
  # loss is computed over exactly n_verify rows, which the loop records.
  d <- .make_overfit_data(n = 101)
  dl <- probotDataLoader(d$x, d$theta, batch = 32, device = "cpu")
  mdl <- probotMakePoint(4, 2, hidden_dims = c(8, 8), device = "cpu")()
  res <- probotTrainPoint(mdl, dl, optim_adam(mdl$parameters, lr = 1e-3),
                          epochs = 1, verbose = FALSE, early_stop = FALSE,
                          holdout_fraction = 0.3, val_batch = 7, split_seed = 8)
  expect_equal(res$validation$n_verify, 30L)   # round(101 * 0.3)
  expect_equal(res$validation$n_train, 71L)
  # val_batch only changes how many batches the pass is cut into, not the mean.
  expect_equal(tail(res$history$val_loss, 1L),
               .reference_val_loss(mdl, d$x, d$theta,
                                   res$validation$verify_idx),
               tolerance = 1e-5)
})

test_that("a pre-split history can be resumed into a run with a split", {
  # rbind() needs identical columns, so a history recorded with
  # holdout_fraction = 0 used to fail with "numbers of columns of arguments do
  # not match" when continued by a default (splitting) run.
  d <- .make_overfit_data(n = 60)
  dl <- probotDataLoader(d$x, d$theta, batch = 16, device = "cpu")
  mdl <- probotMakePoint(4, 2, hidden_dims = c(16, 16), device = "cpu")()
  opt <- optim_adam(mdl$parameters, lr = 1e-3)

  plain <- probotTrainPoint(mdl, dl, opt, epochs = 3, verbose = FALSE,
                            early_stop = FALSE, holdout_fraction = 0)
  expect_false("val_loss" %in% names(plain$history))

  res <- probotTrainPoint(mdl, dl, opt, epochs = 2, history = plain$history,
                          verbose = FALSE, early_stop = FALSE)
  expect_equal(nrow(res$history), 5L)
  expect_true("val_loss" %in% names(res$history))
  # The three pre-split epochs genuinely have no verification number.
  expect_equal(sum(is.na(res$history$val_loss)), 3L)
  expect_true(all(res$history$val_loss > 0, na.rm = TRUE))
})

test_that("continuation without a split in either run is unaffected", {
  d <- .make_overfit_data(n = 60)
  dl <- probotDataLoader(d$x, d$theta, batch = 16, device = "cpu")
  mdl <- probotMakePoint(4, 2, hidden_dims = c(16, 16), device = "cpu")()
  opt <- optim_adam(mdl$parameters, lr = 1e-3)
  a <- probotTrainPoint(mdl, dl, opt, epochs = 2, verbose = FALSE,
                        early_stop = FALSE, holdout_fraction = 0)
  b <- probotTrainPoint(mdl, dl, opt, epochs = 2, history = a$history,
                        verbose = FALSE, early_stop = FALSE,
                        holdout_fraction = 0)
  expect_equal(names(b$history), c("epoch", "loss", "mae", "rmse"))
  expect_equal(nrow(b$history), 4L)
  expect_false(anyNA(b$history))
})

test_that("the split works on the model's own device, not only CPU", {
  # Every other test in this file pins device = "cpu", which hid a real bug:
  # index_select()'s index tensor must live on the same device as the data, and
  # probotDataLoader() defaults to MPS. Across devices torch's MPS backend
  # aborts the process rather than raising an R error, so this cannot be a
  # tolerance check -- it has to actually run.
  skip_if_not(torch::backends_mps_is_available(), "no MPS backend")
  set.seed(9)
  n <- 60
  x <- matrix(rnorm(n * 3), n, 3)
  theta <- matrix(rnorm(n * 2), n, 2)
  dl <- probotDataLoader(x, theta, batch = 16)
  expect_equal(as.character(dl$dataset$tensors[[1]]$device), "mps:0")

  mdl <- probotMakePoint(3, 2, hidden_dims = c(16, 16))()
  res <- probotTrainPoint(mdl, dl, optim_adam(mdl$parameters, lr = 1e-3),
                          epochs = 3, verbose = FALSE, early_stop = FALSE,
                          split_seed = 5)
  expect_equal(res$validation$n_verify, 6L)
  expect_true(all(is.finite(res$history$val_loss)))
  # The split tensors stay on the accelerator; nothing round-trips to CPU.
  sp <- ProBot:::.probotSplitLoaders(dl, 0.1, seed = 5)
  expect_equal(as.character(sp$val$dataset$tensors[[1]]$device), "mps:0")
  expect_equal(as.character(sp$train$dataset$tensors[[2]]$device), "mps:0")
  expect_equal(sp$n_train + sp$n_val, n)
})

test_that("the verify batch is capped rather than one giant pass", {
  # Defaulting the verify batch to the whole holdout is what let a single
  # 90000-row pass dominate the reported number. It is capped at 4096 to match
  # probotLossEval()'s default batch, so the two independent scoring paths stay
  # comparable and no loss function gets handed a batch that makes it build an
  # intermediate quadratic in N.
  dl <- ProBot::probotDataLoader(matrix(rnorm(400), 40), matrix(rnorm(400), 40),
                                 batch = 4, shuffle = FALSE, device = "cpu")
  sp <- ProBot:::.probotSplitLoaders(dl, 0.5, seed = 3)
  expect_equal(sp$val$batch_size, 20L)          # n_val below the cap: unchanged
  expect_equal(sp$train$batch_size, 4L)         # training batch untouched

  big <- ProBot::probotDataLoader(matrix(rnorm(20000), 10000),
                                  matrix(rnorm(20000), 10000), batch = 4,
                                  shuffle = FALSE, device = "cpu")
  sp2 <- ProBot:::.probotSplitLoaders(big, 0.5, seed = 3)
  expect_equal(sp2$n_val, 5000L)
  expect_equal(sp2$val$batch_size, 4096L)

  # An explicit request still wins over the cap.
  sp3 <- ProBot:::.probotSplitLoaders(big, 0.5, seed = 3, val_batch = 100000L)
  expect_equal(sp3$val$batch_size, 100000L)
})

test_that("an explicit val_batch is honoured", {
  d <- .make_overfit_data()
  dl <- probotDataLoader(d$x, d$theta, batch = 16, device = "cpu")
  mdl <- probotMakePoint(4, 2, hidden_dims = c(8, 8), device = "cpu")()
  res <- probotTrainPoint(mdl, dl, optim_adam(mdl$parameters, lr = 1e-3),
                          epochs = 1, verbose = FALSE, early_stop = FALSE,
                          holdout_fraction = 0.5, val_batch = 13,
                          split_seed = 8)
  sp <- ProBot:::.probotSplitLoaders(dl, 0.5, seed = 8, val_batch = 13)
  expect_equal(sp$val$batch_size, 13)
  expect_equal(res$validation$n_verify, sp$n_val)
})
