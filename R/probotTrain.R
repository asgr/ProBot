probotDataLoader <- function(input,
                             output,
                             batch = 1024,
                             shuffle = TRUE,
                             train_idx = NULL) {
  if (is.null(train_idx)) {
    train_idx = 1:nrow(input)
  }

  input_train <- torch_tensor(input[train_idx, ], dtype = torch_float())

  output_train <- torch_tensor(output[train_idx, ], dtype = torch_float())

  dataset <- tensor_dataset(input_train, output_train)

  dataloader <- dataloader(dataset, batch_size = batch, shuffle = shuffle)

  return(dataloader)
}

probotSingleEpochMDN <- function(model,
                                 dataloader,
                                 optimizer,
                                 mdn_components,
                                 lambda = 0.01) {
  model$train()

  running_loss <- 0
  running_mae <- 0
  running_rmse <- 0
  running_sigma <- 0

  running_mix <- numeric(mdn_components)

  n_batches <- 0

  coro::loop(for (batch in dataloader) {
    optimizer$zero_grad()

    output_pred <- model(batch[[1]])

    p <- .probotUnpackMDN(output_pred, mdn_components)

    weights <- nnf_softmax(p$logits, dim = 2)

    mu_mix <- (weights$unsqueeze(3) * p$mu)$sum(dim = 2)

    mse_loss <- ((batch[[2]] - mu_mix)^2)$mean()

    mae <- (batch[[2]] - mu_mix)$abs()$mean()$item()

    rmse <- ((batch[[2]] - mu_mix)^2)$mean()$sqrt()$item()

    mean_sigma <- (10^torch_clamp(p$log10_sigma, min = -5, max = 5))$mean()$item()

    mix_use <- apply(as.array(weights), 2, mean)

    loss <- probotLossMDN(batch[[2]], output_pred, mdn_components) + lambda * mse_loss

    loss$backward()

    optimizer$step()

    running_loss <- running_loss + loss$item()
    running_mae <- running_mae + mae
    running_rmse <- running_rmse + rmse
    running_sigma <- running_sigma + mean_sigma

    running_mix <- running_mix + mix_use

    n_batches <- n_batches + 1
  })

  list(
    loss = running_loss / n_batches,
    mae = running_mae / n_batches,
    rmse = running_rmse / n_batches,
    sigma = running_sigma / n_batches,
    mix = running_mix / n_batches
  )
}

probotTrainMDN <- function(model,
                           dataloader,
                           optimizer,
                           epochs = 100,
                           mdn_components,
                           lambda = 0.01,
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
    metrics <- probotSingleEpochMDN(
      model = model,
      dataloader = dataloader,
      optimizer = optimizer,
      mdn_components = mdn_components,
      lambda = lambda
    )

    history_list[[epoch]] <- c(list(epoch = epoch), metrics)

    loss_history <- c(loss_history, metrics$loss)

    if (verbose) {
      cat(
        sprintf(
          paste0(
            "Epoch %d ",
            "Loss %.3f ",
            "MAE %.3f ",
            "RMSE %.3f ",
            "Sigma %.3f ",
            "MixSD %.3f ",
            "Mix [%s]\n"
          ),
          epoch,
          metrics$loss,
          metrics$mae,
          metrics$rmse,
          metrics$sigma,
          sd(metrics$mix),
          paste(sprintf("%.2f", metrics$mix), collapse = " ")
        )
      )

    }

    # Early stopping based on rolling loss averages
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
          sprintf("mdn_checkpoint_epoch_%03d.pt", epoch)
        )
      )
    }
  }

  history_df <- do.call(rbind, lapply(history_list, function(x) {
    data.frame(
      epoch = x$epoch,
      loss = x$loss,
      mae = x$mae,
      rmse = x$rmse,
      sigma = x$sigma,
      mix_sd = sd(x$mix)
    )

  }))

  if (!is.null(history)) {
    history_df <- rbind(history, history_df)
  }

  list(model = model, history = history_df)
}
