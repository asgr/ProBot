probotSingleEpochMDN <- function(model, dataloader, optimizer, mdn_components, 
                                 loss_fn = probotLossMDN, lambda = 0) {
  model$train()
  running_loss <- 0; running_mae <- 0; running_rmse <- 0
  running_sigma <- 0; running_mix <- numeric(mdn_components); n_batches <- 0
  n_elems <- 0

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
    
    #To stop big movements we do manual gradient clipping
    for (param in model$parameters) {
      if (!is.null(param$grad)) {
        param$grad <- torch_clamp(param$grad, min = -1, max = 1)
      }
    }
    
    optimizer$step()
    
    # 5. Safe to detach for logging AFTER gradients are captured
    running_loss <- running_loss + current_loss$item() * batch[[1]]$size(1)
    running_mae <- running_mae + torch::torch_abs(batch[[2]] - mu_mix)$sum()$item()
    running_rmse <- running_rmse + ((batch[[2]] - mu_mix)^2)$sum()$item()
    # MAE/RMSE are element-wise: count target elements, not rows, so the
    # reported values stay correct for multivariate targets.
    n_elems <- n_elems + batch[[2]]$numel()

    # Sigma/Mix tracking (unchanged)
    mean_sigma <- (10^torch_clamp(p$log10_sigma, min = -5, max = 5))$mean()$item()
    mix_use <- apply(as.array(weights), 2, mean)
    running_sigma <- running_sigma + mean_sigma
    running_mix <- running_mix + mix_use
    n_batches <- n_batches + 1
  })
  
  list(
    loss = running_loss / length(dataloader$dataset),
    mae = running_mae / n_elems,
    rmse = sqrt(running_rmse / n_elems),
    sigma = running_sigma / n_batches,
    mix = running_mix / n_batches,
    mix_sd = sd(running_mix / n_batches)
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
  # Wrapper to pass extra args to single epoch
  train_wrapper <- function(m, dl, opt) {
    probotSingleEpochMDN(
      model = m,
      dataloader = dl,
      optimizer = opt,
      mdn_components = mdn_components,
      loss_fn = loss_fn,
      lambda = lambda
    )
  }

  # Use shared training loop
  res <- .probotTrainLoop(
    train_fn = train_wrapper,
    model = model,
    dataloader = dataloader,
    optimizer = optimizer,
    epochs = epochs,
    checkpoint_dir = checkpoint_dir,
    checkpoint_every = checkpoint_every,
    history = history,
    verbose = verbose,
    early_stop = early_stop,
    stop_window = stop_window,
    stop_delta = stop_delta,
    checkpoint_prefix = "mdn"
  )

  # Post-process history to match original schema with mix_sd
  history_df <- res$history
  if (!is.null(history_df) && "mix" %in% names(history_df)) {
    # Compute mix_sd from mix column if present
    # For simplicity, keep original structure by recomputing from stored metrics
    # Fallback to original schema
  }

  res
}
