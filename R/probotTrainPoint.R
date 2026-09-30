probotSingleEpochPoint <- function(model, dataloader, optimizer, loss_fn = nnf_mse_loss) {
  model$train()
  running_loss <- 0
  running_mae <- 0
  running_rmse <- 0
  n_batches <- 0
  n_elems <- 0

  coro::loop(for (batch in dataloader) {
    optimizer$zero_grad()
    output_pred <- model(batch[[1]])
    current_loss <- loss_fn(output_pred, batch[[2]])
    current_loss$backward()
    
    #To stop big movements we do manual gradient clipping
    for (param in model$parameters) {
      if (!is.null(param$grad)) {
        param$grad <- torch_clamp(param$grad, min = -1, max = 1)
      }
    }
    
    optimizer$step()
    
    running_loss <- running_loss + current_loss$item() * batch[[1]]$size(1)
    diff <- torch_abs(batch[[2]] - output_pred)
    running_mae <- running_mae + diff$sum()$item()
    running_rmse <- running_rmse + ((batch[[2]] - output_pred)^2)$sum()$item()
    n_batches <- n_batches + 1
    # MAE/RMSE are element-wise: count target elements, not rows, so the
    # reported values stay correct for multivariate targets.
    n_elems <- n_elems + batch[[2]]$numel()
  })

  list(
    loss = running_loss / length(dataloader$dataset),
    mae = running_mae / n_elems,
    rmse = sqrt(running_rmse / n_elems)
  )
}

probotTrainPoint <- function(model,
                              dataloader,
                              optimizer,
                              epochs = 100,
                              loss_fn = nnf_mse_loss,
                              checkpoint_dir = NULL,
                              checkpoint_every = 10,
                              history = NULL,
                              verbose = TRUE,
                              early_stop = TRUE,
                              stop_window = 20,
                              stop_delta = 1e-2,
                              holdout_fraction = 0.1,
                              val_dataloader = NULL,
                              val_batch = NULL,
                              split_seed = NULL,
                              holdout_stop = TRUE,
                              holdout_min_delta = 1e-3,
                              holdout_patience = 5L) {
  val <- .probotValSetup(dataloader,
                         holdout_fraction = holdout_fraction,
                         val_dataloader = val_dataloader,
                         val_batch = val_batch,
                         split_seed = split_seed)

  res <- .probotTrainLoop(
    train_fn = probotSingleEpochPoint,
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
    checkpoint_prefix = "point",
    loss_fn = loss_fn,
    val = val,
    score_fn = if (is.null(val)) NULL else
      .probotValScorer("point", loss_fn),
    holdout_stop = holdout_stop,
    holdout_min_delta = holdout_min_delta,
    holdout_patience = holdout_patience
  )
  res
}
