probotSingleEpochPoint <- function(model, dataloader, optimizer, loss_fn = nnf_mse_loss) {
  model$train()
  running_loss <- 0
  running_mae <- 0
  running_rmse <- 0
  n_batches <- 0
  
  coro::loop(for (batch in dataloader) {
    optimizer$zero_grad()
    output_pred <- model(batch[[1]])
    current_loss <- loss_fn(output_pred, batch[[2]])
    current_loss$backward()
    optimizer$step()
    
    running_loss <- running_loss + current_loss$item() * batch[[1]]$size(1)
    diff <- torch_abs(batch[[2]] - output_pred)
    running_mae <- running_mae + diff$sum()$item()
    running_rmse <- running_rmse + ((batch[[2]] - output_pred)^2)$sum()$item()
    n_batches <- n_batches + 1
  })
  
  list(
    loss = running_loss / length(dataloader$dataset),
    mae = running_mae / length(dataloader$dataset),
    rmse = sqrt(running_rmse / length(dataloader$dataset))
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
                              stop_delta = 1e-2) {
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
    loss_fn = loss_fn
  )
  # Ensure mae/rmse columns are present in history
  res
}
