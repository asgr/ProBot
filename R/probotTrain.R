probotSingleEpochFlow <- function(model, dataloader, optimizer,
                                  loss_fn = probotLossNF) {
  model$train()
  running_loss <- 0

  coro::loop(for (batch in dataloader) {
    optimizer$zero_grad()

    # batch[[1]] = true_theta (parameters), batch[[2]] = context_x (observations)
    current_loss <- loss_fn(batch[[1]], batch[[2]], model)
    current_loss$backward()
    optimizer$step()

    running_loss <- running_loss + current_loss$item() * batch[[1]]$size(1)
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
                                  stop_delta = 1e-4) {
  history_list <- vector("list", epochs)
  loss_history <- numeric()

  for (epoch in seq_len(epochs)) {
    metrics <- probotSingleEpochFlow(
      model = model,
      dataloader = dataloader,
      optimizer = optimizer,
      loss_fn = loss_fn
    )

    history_list[[epoch]] <- list(epoch = epoch, loss = metrics$loss)
    loss_history <- c(loss_history, metrics$loss)

    if (verbose && (epoch %% checkpoint_every == 0 || epoch == 1)) {
      cat(sprintf("Epoch %d  Loss %.4f\n", epoch, metrics$loss))
    }

    if (early_stop && length(loss_history) >= 2 * stop_window) {
      recent_mean   <- mean(tail(loss_history, stop_window))
      previous_mean <- mean(tail(loss_history, 2 * stop_window)[seq_len(stop_window)])
      improvement   <- previous_mean - recent_mean

      if (improvement < stop_delta) {
        if (verbose) {
          cat(sprintf(
            "\nEarly stopping at epoch %d: average loss improved by only %.6f over the last %d epochs (threshold = %.6f)\n\n",
            epoch, improvement, stop_window, stop_delta
          ))
        }
        history_list <- history_list[seq_len(epoch)]
        break
      }
    }

    if (!is.null(checkpoint_dir) && epoch %% checkpoint_every == 0) {
      torch_save(
        list(
          epoch     = epoch,
          model     = model$state_dict(),
          optimizer = optimizer$state_dict(),
          loss      = metrics$loss
        ),
        file.path(checkpoint_dir, sprintf("flow_checkpoint_epoch_%03d.pt", epoch))
      )
    }
  }

  history_df <- do.call(rbind, lapply(history_list, function(x) {
    data.frame(epoch = x$epoch, loss = x$loss)
  }))

  if (!is.null(history)) {
    history_df <- rbind(history, history_df)
  }

  list(model = model, history = history_df)
}

probotSingleEpochMDN <- function(model, dataloader, optimizer, mdn_components, 
                                 loss_fn = probotLossMDN, lambda = 0) {
  model$train()
  running_loss <- 0; running_mae <- 0; running_rmse <- 0
  running_sigma <- 0; running_mix <- numeric(mdn_components); n_batches <- 0
  
  coro::loop(for (batch in dataloader) {
    optimizer$zero_grad()
    output_pred <- model(batch[[1]])
    
    # 1. Compute primary loss (tensor)
    current_loss <- loss_fn(batch[[2]], output_pred, mdn_components)
    
    # Unpack for metrics & mixture mean
    p <- .probotUnpackMDN(output_pred, mdn_components)
    weights <- nnf_softmax(p$logits, dim = 2)
    mu_mix <- (weights$unsqueeze(3) * p$mu)$sum(dim = 2)
    
    # 2. Compute MSE as a tensor to preserve gradient flow
    mse_loss <- ((batch[[2]] - mu_mix)^2)$mean()
    
    # 3. Blend losses on tensors BEFORE backward()
    if (lambda > 0) {
      current_loss <- current_loss * (1 - lambda) + mse_loss * lambda
    }
    
    # 4. Backpropagate & step
    current_loss$backward()
    optimizer$step()
    
    # 5. Safe to detach for logging AFTER gradients are captured
    running_loss <- running_loss + current_loss$item() * length(batch[[1]])
    running_mae <- running_mae + torch::torch_abs(batch[[2]] - mu_mix)$sum()$item()
    running_rmse <- running_rmse + ((batch[[2]] - mu_mix)^2)$sum()$item()
    
    # Sigma/Mix tracking (unchanged)
    mean_sigma <- (10^torch_clamp(p$log10_sigma, min = -5, max = 5))$mean()$item()
    mix_use <- apply(as.array(weights), 2, mean)
    running_sigma <- running_sigma + mean_sigma
    running_mix <- running_mix + mix_use
    n_batches <- n_batches + 1
  })
  
  list(
    loss = running_loss / length(dataloader$dataset),
    mae = running_mae / length(dataloader$dataset),
    rmse = sqrt(running_rmse / length(dataloader$dataset)),
    sigma = running_sigma / n_batches,
    mix = running_mix / n_batches
  )
}

probotTrainMDN <- function(model,
                           dataloader,
                           optimizer,
                           epochs = 100,
                           mdn_components,
                           loss_fn = probotLossMDN,
                           lambda = 0,
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
      loss_fn = loss_fn,
      lambda = lambda
    )

    history_list[[epoch]] <- c(list(epoch = epoch), metrics)

    loss_history <- c(loss_history, metrics$loss)

    if (verbose && (epoch %% checkpoint_every == 0 || epoch == 1)) {
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
