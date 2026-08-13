.probotChooseDevice <- function(device) {
  if (is.null(device)) {
    device <- if (backends_mps_is_available()) "mps" else "cpu"
  }
  torch_device(device)
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
  checkpoint_prefix = "checkpoint"
) {
  history_list <- vector("list", epochs)
  loss_history <- numeric()

  for (epoch in seq_len(epochs)) {
    metrics <- train_fn(model, dataloader, optimizer)

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
      # Print loss if available
      if (!is.null(metrics$loss)) {
        cat(sprintf("Epoch %d Loss %.6f\n", epoch, metrics$loss))
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
      if (is.numeric(val)) {
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
