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
  ...
) {
  history_list <- vector("list", epochs)
  loss_history <- numeric()

  for (epoch in seq_len(epochs)) {
    metrics <- train_fn(model, dataloader, optimizer, ...)

    # Normalise metrics to a list with at least loss
    if (is.list(metrics) && !is.null(metrics$loss)) {
      loss_val <- metrics$loss
    } else {
      loss_val <- metrics
    }

    # Store full metrics
    history_list[[epoch]] <- c(list(epoch = epoch), metrics)
    loss_history <- c(loss_history, loss_val)

    if (verbose && (epoch %% checkpoint_every == 0 || epoch == 1)) {
      # Print loss and optional metrics
      if (!is.null(metrics$loss)) {
        if (!is.null(metrics$mae) && !is.null(metrics$rmse) && !is.null(metrics$sigma) && !is.null(metrics$mix)) {
          # MDN-style printing
          cat(sprintf("Epoch %d Loss %.3f MAE %.3f RMSE %.3f Sigma %.3f MixSD %.3f Mix [%s]\n",
                      epoch, metrics$loss, metrics$mae, metrics$rmse, metrics$sigma, sd(metrics$mix), paste(sprintf("%.2f", metrics$mix), collapse = " ")))
        } else if (!is.null(metrics$mae) && !is.null(metrics$rmse)) {
          cat(sprintf("Epoch %d Loss %.6f MAE %.6f RMSE %.6f\n", epoch, metrics$loss, metrics$mae, metrics$rmse))
        } else {
          cat(sprintf("Epoch %d Loss %.6f\n", epoch, metrics$loss))
        }
      } else {
        cat(sprintf("Epoch %d\n", epoch))
      }
    }

    # Early stopping
    if (early_stop && length(loss_history) >= 2 * stop_window) {
      recent_mean <- mean(tail(loss_history, stop_window))
      previous_window <- tail(loss_history, 2 * stop_window)[seq_len(stop_window)]
      previous_mean <- mean(previous_window)
      improvement <- previous_mean - recent_mean

      if (improvement > 0 & improvement < stop_delta) {
        if (verbose) {
          cat(sprintf("\nEarly stopping at epoch %d: average loss improved by only %.6f over last %d epochs (threshold = %.6f)\n\n",
                      epoch, improvement, stop_window, stop_delta))
        }
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
    history_df <- rbind(history, history_df)
  }

  list(model = model, history = history_df)
}
