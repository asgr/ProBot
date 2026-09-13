.probotChooseDevice <- function(device) {
  if (is.null(device)) {
    device <- if (backends_mps_is_available()) "mps" else "cpu"
  }
  torch_device(device)
}

# Validate a loss_fn argument. mdn_components used to sit immediately before
# loss_fn in probotSingleEpochMDN()/probotTrainMDN(), so a legacy positional
# call binds the mixture count to loss_fn.
.probotCheckLossFn <- function(loss_fn) {
  if (is.function(loss_fn)) return(invisible(TRUE))
  stop("'loss_fn' must be a loss function, but is a ", class(loss_fn)[1L],
       ". If this call passed mdn_components by position, note that the ",
       "argument has been removed -- the mixture count is read from the model.",
       call. = FALSE)
}

# How many arguments of a call were written by position.
#
# names(sys.call()) has one entry per element of the call *including* the
# function name in slot 1, and that slot is always "", so the empty-name count
# must drop it: otherwise probotSave(m, o, f, "point", col_means = cm) reports
# five positional arguments when only four are, and refuses a legal call.
#
# names() is NULL exactly when no argument is named, which must not be counted
# as zero (`%in% NULL` is always FALSE); that case is length(call) - 1.
#
# match.call() cannot be used here: it resolves positional arguments to their
# formal names, erasing precisely the information being counted.
.probotNPositional <- function(call.) {
  if (length(call.) <= 1L) return(0L)
  snm <- names(call.)
  if (is.null(snm)) return(length(call.) - 1L)
  as.integer(sum((snm == "" | is.na(snm))[-1L]))
}

# Reject a call that passes more arguments by position than `n_safe`.
#
# When an argument is deleted from the middle of a signature, everything after
# it shifts up a slot, and a legacy positional call then binds the old value to
# the new neighbour. Usually that fails noisily downstream, but not always:
# dropping mdn_components from probotSamplePostMDN() sent a legacy 3rd
# positional argument to n_samples, so probotSamplePostMDN(x, model, 5) quietly
# returned five draws instead of erroring. Guard the calls where the shift is
# silent and let the others fail on their own.
#
# sys.call(-1L) is the call to the guarded function, since this helper runs
# inside it.
.probotPositionalLimit <- function(n_safe, what, call. = sys.call(-1L)) {
  n_positional <- .probotNPositional(call.)
  if (n_positional > n_safe) {
    stop(deparse(call.[[1L]]), "(): only ", n_safe, " arguments (", what,
         ") may be given positionally; everything after them must be named. ",
         "This call passed ", n_positional, " by position, which would ",
         "silently mis-bind a value removed from the signature.", call. = FALSE)
  }
  invisible(TRUE)
}

# ---------------------------------------------------------------------------
# Train / holdout splitting
# ---------------------------------------------------------------------------

# sample.int() but reproducible without claiming the caller's RNG stream
# permanently. withr is not a dependency of this package, so the seed is saved
# and restored by hand.
.probotWithSeed <- function(seed, code) {
  if (is.null(seed)) return(code)
  had_seed <- exists(".Random.seed", envir = globalenv(), inherits = FALSE)
  if (had_seed) {
    old_seed <- get(".Random.seed", envir = globalenv(), inherits = FALSE)
  }
  set.seed(as.integer(seed))
  on.exit({
    if (had_seed) {
      assign(".Random.seed", old_seed, envir = globalenv())
    } else {
      rm(".Random.seed", envir = globalenv())
    }
  })
  code
}

# Split one dataloader's dataset into two loaders over disjoint rows.
#
# The holdout loader is always shuffle = FALSE and drop_last = FALSE: the
# overfitting rules compare the holdout loss across epochs, so a reshuffled or
# truncated holdout would add sampling noise to exactly the quantity being
# trended. The training loader keeps the original batch size, drop_last and
# shuffle settings.
#
# Because the training rows are re-indexed into a compacted tensor, a run with a
# holdout is NOT numerically identical to the same run with
# holdout_fraction = 0, even at the same seed.
.probotSplitLoaders <- function(dataloader, holdout_fraction = 0.1, seed = NULL,
                                val_batch = NULL) {
  tensors <- dataloader$dataset$tensors
  if (is.null(tensors) || length(tensors) < 2L) {
    stop("cannot split this dataloader: its dataset does not expose input and ",
         "output tensors (needed to slice a holdout).", call. = FALSE)
  }
  n <- tensors[[1]]$size(1)
  if (n < 2L) {
    stop("cannot split a dataset with only ", n, " row(s).", call. = FALSE)
  }

  # At least one row on each side, however extreme the requested fraction.
  n_val <- max(1L, min(n - 1L, as.integer(round(n * holdout_fraction))))

  idx <- .probotWithSeed(seed, sample.int(n))
  val_idx <- sort(idx[seq_len(n_val)])
  train_idx <- sort(idx[-seq_len(n_val)])

  # torch indexes from 1. Both halves keep their original row order, so the
  # split is stable across epochs and the training loader's order is only
  # changed by compaction, not by the holdout draw.
  #
  # The index tensor has to be built on the *data's* device: probotDataLoader()
  # places tensors on MPS by default, and index_select across devices aborts in
  # torch's MPS backend rather than raising an R error.
  take <- function(t, i) {
    t$index_select(1L, torch_tensor(as.integer(i), dtype = torch_long(),
                                    device = t$device))
  }
  train_tensors <- lapply(tensors, take, i = train_idx)
  val_tensors <- lapply(tensors, take, i = val_idx)

  batch <- dataloader$batch_size
  if (is.null(batch)) batch <- 1L
  shuffled <- inherits(dataloader$sampler, "utils_sampler_random")
  if (is.null(val_batch)) val_batch <- n_val

  list(
    train = dataloader(do.call(tensor_dataset, train_tensors),
                       batch_size = batch, shuffle = shuffled,
                       drop_last = isTRUE(dataloader$drop_last)),
    val = dataloader(do.call(tensor_dataset, val_tensors),
                     batch_size = val_batch, shuffle = FALSE,
                     drop_last = FALSE),
    train_idx = train_idx,
    val_idx = val_idx,
    n_train = length(train_idx),
    n_val = n_val
  )
}

# Build the validation spec a trainer hands to .probotTrainLoop(), or NULL.
#
# Precedence is an explicit val_dataloader, then holdout_fraction. Validation
# happens here, before epoch 1, so a bad argument costs no training time.
#
# `score_fn` does the per-batch loss for one model type; it captures the
# caller's own loss_fn and its lambda blend so the holdout number is measured on
# exactly the same quantity as the training number. Comparing a pure-NLL holdout
# against a blended training loss would make the overfitting rule meaningless.
.probotValSetup <- function(dataloader,
                            holdout_fraction = NULL,
                            val_dataloader = NULL,
                            val_batch = NULL,
                            split_seed = NULL) {
  # Every trainer calls this before epoch 1, so it is also where the dataloader
  # itself gets checked. Without it, passing a matrix or data frame by mistake
  # surfaces as "$ operator is invalid for atomic vectors" from the split.
  if (!inherits(dataloader, "dataloader")) {
    stop("'dataloader' must be a torch dataloader (see probotDataLoader).",
         call. = FALSE)
  }

  if (!is.null(val_dataloader)) {
    if (!inherits(val_dataloader, "dataloader")) {
      stop("'val_dataloader' must be a torch dataloader (see probotDataLoader).",
           call. = FALSE)
    }
    n_val <- length(val_dataloader$dataset)
    if (n_val < 1L) stop("'val_dataloader' has no rows to score.", call. = FALSE)
    return(list(
      dataloader = val_dataloader, n_val = n_val,
      n_train = length(dataloader$dataset),
      val_idx = NULL, source = "val_dataloader"
    ))
  }

  if (is.null(holdout_fraction)) holdout_fraction <- 0
  if (!is.numeric(holdout_fraction) || length(holdout_fraction) != 1L ||
      is.na(holdout_fraction) || holdout_fraction < 0 || holdout_fraction >= 1) {
    stop("'holdout_fraction' must be a single number in [0, 1) ",
         "(0 disables the split).", call. = FALSE)
  }
  if (holdout_fraction == 0) return(NULL)

  n <- length(dataloader$dataset)
  if (n < 10L) {
    warning("holdout split skipped: the dataloader has only ", n, " row(s), so a ",
            "verification set would be too small to trend. Training on all rows ",
            "instead; pass holdout_fraction = 0 to silence this.", call. = FALSE)
    return(NULL)
  }

  sp <- .probotSplitLoaders(dataloader, holdout_fraction = holdout_fraction,
                            seed = split_seed, val_batch = val_batch)
  list(
    dataloader = sp$val, n_val = sp$n_val, n_train = sp$n_train,
    val_idx = sp$val_idx, train_dataloader = sp$train,
    source = "holdout_fraction", fraction = holdout_fraction
  )
}

# Loss function for one holdout batch, mirroring the trainer's own maths.
#
# The blend matters: with lambda > 0 the trainer optimises
# (1 - lambda) * loss_fn + lambda * mse, and a holdout measured on the pure
# loss_fn would not be the same quantity as the training number -- the
# overfitting rule compares trends of the two columns, so they must be the same
# functional. The blended term is also what makes val_mae/val_rmse meaningful,
# since unblended there is no point estimate to take an error of.
#
# Returns list(loss, mae, rmse, n_elems) where the errors are *sums over target
# elements* and mae/rmse are NULL when the blend is off, matching what
# probotSingleEpoch*() accumulate.
.probotValScorer <- function(model_type, loss_fn, lambda = 0,
                             point = "centre", n_point_samples = 32) {
  force(lambda)
  function(xb, yb, model) {
    if (model_type == "flow") {
      current <- loss_fn(output_true = yb, output_pred = xb, model = model)
      if (lambda <= 0) return(list(loss = current$item()))
      theta_hat <- .probotFlowPointEstimate(model, context = xb,
                                            output_dim = yb$size(2),
                                            point = point,
                                            n_point_samples = n_point_samples)
      mse <- ((yb - theta_hat)^2)$mean()
      return(list(
        loss = (current * (1 - lambda) + mse * lambda)$item(),
        mae = torch_abs(yb - theta_hat)$sum()$item(),
        rmse = ((yb - theta_hat)^2)$sum()$item(),
        n_elems = yb$numel()
      ))
    }

    pred <- model(xb)
    if (model_type == "mdn") {
      current <- loss_fn(output_true = yb, output_pred = pred, model = model)
      if (lambda <= 0) return(list(loss = current$item()))
      p <- .probotUnpackMDN(pred, model)
      weights <- nnf_softmax(p$logits, dim = 2)
      theta_hat <- (weights$unsqueeze(3) * p$mu)$sum(dim = 2)
    } else {
      current <- loss_fn(pred, yb)
      if (lambda <= 0) return(list(loss = current$item()))
      # Point has no lambda in its own trainer; a caller who asks for the blend
      # gets the identical (i.e. pure MSE) value, because its loss_fn *is* MSE.
      theta_hat <- pred
    }

    mse <- ((yb - theta_hat)^2)$mean()
    list(
      loss = (current * (1 - lambda) + mse * lambda)$item(),
      mae = torch_abs(yb - theta_hat)$sum()$item(),
      rmse = ((yb - theta_hat)^2)$sum()$item(),
      n_elems = yb$numel()
    )
  }
}

# Mean loss over the holdout, in eval mode, with no gradient step.
#
# Iterating the loader (rather than slicing tensors by hand) keeps this correct
# for any batch size, and a shuffle = FALSE loader replays the same rows every
# epoch, which is what makes the epoch-to-epoch trend comparable.
#
# Deliberately NOT probotLossEval(): that function exists to accept plain R
# matrices, so it routes the data through CPU matrices and back. The holdout
# tensors are already on the model's device, and this runs every epoch.
.probotValScore <- function(model, val, score_fn) {
  was_training <- model$training
  model$eval()
  on.exit(if (was_training) model$train())

  total <- 0
  mae_total <- 0
  rmse_total <- 0
  n_elems <- 0L
  n <- 0L

  with_no_grad({
    coro::loop(for (batch in val$dataloader) {
      res <- score_fn(batch[[1]], batch[[2]], model)
      b <- batch[[1]]$size(1)
      total <- total + res$loss * b
      n <- n + b
      if (!is.null(res$mae)) {
        # mae/rmse arrive as per-batch sums over target *elements*, matching the
        # denominators the trainers use, so a short final batch is not
        # over-weighted.
        mae_total <- mae_total + res$mae
        rmse_total <- rmse_total + res$rmse
        n_elems <- n_elems + res$n_elems
      }
    })
  })

  if (n < 1L) stop("holdout dataloader produced no batches.", call. = FALSE)
  out <- list(val_loss = total / n, n_rows = n)
  if (n_elems > 0L) {
    out$val_mae <- mae_total / n_elems
    out$val_rmse <- sqrt(rmse_total / n_elems)
  }
  out
}

# ---------------------------------------------------------------------------
# Overfitting stop rule
# ---------------------------------------------------------------------------

# Least-squares slope of `values` against epoch index over the last `window`
# points: a trend of exactly delta per epoch returns delta whatever the window
# length. NA when fewer than two usable points.
.probotTrend <- function(values, window) {
  v <- as.numeric(tail(values, window))
  if (length(v) < 2L || !all(is.finite(v))) return(NA_real_)
  t <- seq_along(v) - mean(seq_along(v))
  sum(t * (v - mean(v))) / sum(t^2)
}

# Decide whether the holdout says to stop.
#
# The signal for over-training is asymmetric: the training loss keeps falling
# while the holdout loss flattens or turns upward. Both halves matter, which is
# why `train_trend_epsilon` exists -- a training loss that has itself stagnated
# is not over-fitting, and the existing plateau rule is better placed to stop
# for that.
#
# A short window cannot resolve a slope, so below 4 points the test degrades to
# "neither loss is moving", which is the conservative reading rather than a
# guess. `patience` counts consecutive flagged epochs so one noisy epoch does
# not end the run.
#
# `state` (the running counter) is threaded through because the loop owns it.
.probotHoldoutStop <- function(train_loss, val_loss, window, min_delta,
                               train_trend_epsilon, patience, state) {
  w <- min(as.integer(window), length(val_loss))
  train_trend <- .probotTrend(train_loss, w)
  val_trend <- .probotTrend(val_loss, w)

  if (is.na(train_trend) || is.na(val_trend)) {
    return(list(state = 0L, stop = FALSE, train_trend = train_trend,
                val_trend = val_trend, window = w))
  }

  informative <- w >= 4L
  train_improving <- if (informative) train_trend < -train_trend_epsilon else
    train_trend <= 0
  val_stalled <- if (informative) val_trend > -min_delta else val_trend >= 0

  state <- if (train_improving && val_stalled) state + 1L else 0L
  list(state = state, stop = state >= as.integer(patience),
       train_trend = train_trend, val_trend = val_trend, window = w)
}

# ---------------------------------------------------------------------------
# Shared training loop
# ---------------------------------------------------------------------------

# Append a new run's history to an existing one.
#
# Base rbind() requires identical columns, which broke resuming once the holdout
# split was added: a history recorded with holdout_fraction = 0 has no val_loss,
# and rbind() died with "numbers of columns of arguments do not match" without
# naming the cause. Union the columns and leave absent ones NA, which is the
# honest reading -- that epoch really has no verification number.
.probotBindHistory <- function(old, new) {
  cols <- union(names(old), names(new))
  for (nm in setdiff(cols, names(old))) {
    old[[nm]] <- rep(NA_real_, nrow(old))
  }
  for (nm in setdiff(cols, names(new))) {
    new[[nm]] <- rep(NA_real_, nrow(new))
  }
  # Align column order so rbind() pairs like with like.
  rbind(old[, cols, drop = FALSE], new[, cols, drop = FALSE])
}

.probotTrainLoop <- function(
  train_fn,
  model,
  dataloader,
  optimizer,
  epochs = 100,
  checkpoint_dir = NULL,
  checkpoint_every = 10,
  history = NULL,
  verbose = TRUE,
  early_stop = TRUE,
  stop_window = 20,
  stop_delta = 1e-2,
  checkpoint_prefix = "checkpoint",
  val = NULL,
  score_fn = NULL,
  holdout_stop = TRUE,
  holdout_min_delta = 1e-3,
  holdout_patience = 5L,
  ...
) {
  if (!is.null(val) && !is.null(val$train_dataloader)) {
    # The trainer already did the split; train on the holdout-free rows.
    dataloader <- val$train_dataloader
  }
  if (!is.null(val) && is.null(score_fn)) {
    stop("internal error: a validation set needs a score_fn to be scored.",
         call. = FALSE)
  }

  stop_window <- max(2L, as.integer(stop_window))
  holdout_patience <- max(1L, as.integer(holdout_patience))

  history_list <- vector("list", epochs)
  loss_history <- numeric()
  val_history <- numeric()
  stop_state <- 0L
  stop_reason <- NULL
  best_epoch <- NULL
  best_val <- Inf

  if (!is.null(val) && verbose) {
    cat(sprintf("Split %d train / %d verify rows%s\n", val$n_train, val$n_val,
                if (is.null(val$fraction)) "" else
                  sprintf(" (holdout_fraction = %g)", val$fraction)))
  }

  for (epoch in seq_len(epochs)) {
    metrics <- train_fn(model, dataloader, optimizer, ...)

    # Normalise metrics to a list with at least loss
    if (is.list(metrics) && !is.null(metrics$loss)) {
      loss_val <- metrics$loss
    } else {
      loss_val <- metrics
    }

    # Holdout pass: eval mode, no gradients, training mode restored.
    if (!is.null(val)) {
      vs <- .probotValScore(model, val, score_fn)
      # n_rows stays out of `metrics`: it is constant, so it would otherwise add
      # a redundant column to every history row.
      metrics <- c(metrics, vs[setdiff(names(vs), "n_rows")])
      val_history <- c(val_history, vs$val_loss)
      if (is.finite(vs$val_loss) && vs$val_loss < best_val) {
        best_val <- vs$val_loss
        best_epoch <- epoch
      }
    }

    # Store full metrics
    history_list[[epoch]] <- c(list(epoch = epoch), metrics)
    loss_history <- c(loss_history, loss_val)

    if (verbose && (epoch %% checkpoint_every == 0 || epoch == 1)) {
      .probotEpochLine(epoch, metrics, n_verify = if (is.null(val)) {
        NULL
      } else {
        val$n_val
      })
    }

    # Early stopping on the training-loss plateau.
    if (early_stop && length(loss_history) >= 2 * stop_window) {
      recent_mean <- mean(tail(loss_history, stop_window))
      previous_window <- tail(loss_history, 2 * stop_window)[seq_len(stop_window)]
      previous_mean <- mean(previous_window)
      improvement <- previous_mean - recent_mean

      # is.finite, not only the comparison: a NaN epoch makes `improvement` NaN
      # and `if (improvement > 0 & improvement < stop_delta)` throws "missing
      # value where TRUE/FALSE needed", aborting the run.
      if (is.finite(improvement) && improvement > 0 && improvement < stop_delta) {
        if (verbose) {
          cat(sprintf("\nEarly stopping at epoch %d: average training loss improved by only %.6f over last %d epochs (threshold = %.6f)\n\n",
                      epoch, improvement, stop_window, stop_delta))
        }
        stop_reason <- "train_loss_plateau"
        history_list <- history_list[seq_len(epoch)]
        break
      }
    }

    # Early stopping on the holdout: over-training is when the training loss
    # keeps improving but the verification loss no longer is.
    if (early_stop && holdout_stop && !is.null(val) && length(val_history) >= 2L) {
      hs <- .probotHoldoutStop(
        train_loss = loss_history, val_loss = val_history,
        window = stop_window, min_delta = holdout_min_delta,
        # Half the training threshold: "still improving" has to be a weaker bar
        # than the plateau rule's own, or the two rules fire on the same epoch
        # and neither says anything useful.
        train_trend_epsilon = stop_delta / 2,
        patience = holdout_patience, state = stop_state
      )
      stop_state <- hs$state

      if (hs$stop) {
        if (verbose) {
          cat(sprintf("\nStopping at epoch %d: training loss is still falling (%.3g per epoch over the last %d) but the verification loss is not (%.3g per epoch), flagged %d epoch(s) running.\n",
                      epoch, hs$train_trend, hs$window, hs$val_trend, stop_state))
          if (!is.null(best_epoch)) {
            cat(sprintf("  Best verification loss %.6f at epoch %d (now %.6f).\n",
                        best_val, best_epoch, tail(val_history, 1L)))
            cat("  The model is returned as trained at the stopping epoch; ",
                "res$validation$best_epoch names the better one.\n", sep = "")
          }
          cat("\n")
        }
        stop_reason <- "holdout_overfit"
        history_list <- history_list[seq_len(epoch)]
        break
      }
    }

    # Checkpointing
    if (!is.null(checkpoint_dir) && epoch %% checkpoint_every == 0) {
      torch_save(
        list(
          epoch = epoch,
          model = model$state_dict(),
          optimizer = optimizer$state_dict(),
          loss = loss_val
        ),
        file.path(checkpoint_dir, sprintf("%s_epoch_%03d.pt", checkpoint_prefix, epoch))
      )
    }
  }

  # Build history data.frame from common columns
  history_df <- do.call(rbind, lapply(history_list, function(x) {
    # Keep only common numeric columns
    df <- data.frame(epoch = x$epoch)
    for (nm in names(x)) {
      if (nm == "epoch") next
      val <- x[[nm]]
      if (is.numeric(val) && length(val) == 1) {
        df[[nm]] <- val
      }
    }
    df
  }))

  if (!is.null(history)) {
    history_df <- .probotBindHistory(history, history_df)
  }

  res <- list(model = model, history = history_df)

  if (!is.null(val)) {
    res$validation <- list(
      n_train = val$n_train,
      n_verify = val$n_val,
      verify_idx = val$val_idx,
      source = val$source,
      holdout_fraction = val$fraction,
      best_epoch = best_epoch,
      best_val_loss = if (is.null(best_epoch)) NULL else best_val,
      final_val_loss = if (length(val_history)) tail(val_history, 1L) else NULL,
      stop_reason = stop_reason
    )
  }

  res
}

# One verbose line per reported epoch, naming both the training and (when there
# is a holdout) the verification number, so a reader cannot mistake one for the
# other.
#
# The MDN arm keeps its original 3-decimal formatting; the others keep the
# 6-decimal format they had before the split existed.
.probotEpochLine <- function(epoch, metrics, n_verify = NULL) {
  if (is.null(metrics$loss)) {
    cat(sprintf("Epoch %d\n", epoch))
    return(invisible(NULL))
  }

  is_mdn <- !is.null(metrics$sigma) && !is.null(metrics$mix)
  dp <- if (is_mdn) 3L else 6L

  verify <- if (is.null(metrics$val_loss)) "" else sprintf(
    " | Verify Loss %s (%d rows)",
    sprintf(paste0("%.", dp, "f"), metrics$val_loss), n_verify)

  line <- sprintf("Epoch %d Train Loss %s%s", epoch,
                  sprintf(paste0("%.", dp, "f"), metrics$loss), verify)

  if (is_mdn) {
    line <- paste0(line, sprintf("  MAE %.3f RMSE %.3f Sigma %.3f MixSD %.3f Mix [%s]",
                                 metrics$mae, metrics$rmse, metrics$sigma,
                                 sd(metrics$mix),
                                 paste(sprintf("%.2f", metrics$mix),
                                       collapse = " ")))
    if (!is.null(metrics$val_mae)) {
      line <- paste0(line, sprintf("  | Verify MAE %.3f RMSE %.3f",
                                   metrics$val_mae, metrics$val_rmse))
    }
  } else if (!is.null(metrics$mae) && !is.null(metrics$rmse)) {
    line <- paste0(line, sprintf("  MAE %.6f RMSE %.6f", metrics$mae,
                                 metrics$rmse))
    if (!is.null(metrics$val_mae)) {
      line <- paste0(line, sprintf("  | Verify MAE %.6f RMSE %.6f",
                                   metrics$val_mae, metrics$val_rmse))
    }
  }

  cat(line, "\n", sep = "")
  invisible(NULL)
}
