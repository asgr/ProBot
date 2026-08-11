probotSingleEpochFlow <- function(model,
                                  dataloader,
                                  optimizer,
                                  loss_fn = probotLossNF) {
  model$train()
  running_loss <- 0

  coro::loop(for (batch in dataloader) {
    optimizer$zero_grad()

    current_loss <- loss_fn(
      true_theta = batch[[2]],
      context_x = batch[[1]],
      model = model
    )

    current_loss$backward()
    optimizer$step()

    running_loss <- running_loss + current_loss$item() * length(batch[[1]])
  })

  list(
    loss = running_loss / length(dataloader$dataset)
  )
}

probotMultiEpochFlow <- function(model,
                                 dataloader,
                                 optimizer,
                                 epochs = 100,
                                 loss_fn = probotLossNF,
                                 checkpoint_dir = NULL,
                                 checkpoint_every = 10,
                                 history = NULL,
                                 verbose = TRUE,
                                 early_stop = TRUE,
                                 stop_window = 20,
                                 stop_delta = 1e-2) {
  history_list <- vector("list", epochs)
  loss_history <- numeric()

  for (epoch in seq_len(epochs)) {
    metrics <- probotSingleEpochFlow(
      model = model,
      dataloader = dataloader,
      optimizer = optimizer,
      loss_fn = loss_fn
    )

    history_list[[epoch]] <- c(list(epoch = epoch), metrics)
    loss_history <- c(loss_history, metrics$loss)

    if (verbose && (epoch %% checkpoint_every == 0 || epoch == 1)) {
      cat(sprintf("Epoch %d Loss %.3f\n", epoch, metrics$loss))
    }

    if (early_stop && length(loss_history) >= 2 * stop_window) {
      recent_mean <- mean(tail(loss_history, stop_window))
      previous_window <- tail(loss_history, 2 * stop_window)[seq_len(stop_window)]
      previous_mean <- mean(previous_window)
      improvement <- previous_mean - recent_mean

      if (improvement < stop_delta) {
        if (verbose) {
          cat(sprintf(
            paste0(
              "\nEarly stopping at epoch %d: ",
              "average loss improved by only %.6f ",
              "over the last %d epochs ",
              "(threshold = %.6f)\n\n"
            ),
            epoch,
            improvement,
            stop_window,
            stop_delta
          ))
        }

        history_list <- history_list[seq_len(epoch)]
        break
      }
    }

    if (!is.null(checkpoint_dir) &&
        epoch %% checkpoint_every == 0) {
      torch_save(
        list(
          epoch = epoch,
          model = model$state_dict(),
          optimizer = optimizer$state_dict(),
          loss = metrics$loss
        ),
        file.path(
          checkpoint_dir,
          sprintf("flow_checkpoint_epoch_%03d.pt", epoch)
        )
      )
    }
  }

  history_df <- do.call(rbind, lapply(history_list, function(x) {
    data.frame(
      epoch = x$epoch,
      loss = x$loss
    )
  }))

  if (!is.null(history)) {
    history_df <- rbind(history, history_df)
  }

  list(model = model, history = history_df)
}
