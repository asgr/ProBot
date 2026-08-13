probotSingleEpochFlow <- function(model,
                                  dataloader,
                                  optimizer,
                                  loss_fn = probotLossNF) {
  model$train()
  running_loss <- 0

  coro::loop(for (batch in dataloader) {
    optimizer$zero_grad()

    current_loss <- loss_fn(
      output_true = batch[[2]],
      output_pred = batch[[1]],
      model = model
    )

    current_loss$backward()
    optimizer$step()

    running_loss <- running_loss + current_loss$item() * batch[[1]]$size(1)
  })

  list(
    loss = running_loss / length(dataloader$dataset)
  )
}

probotTrainFlow <- function(model,
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
  .probotTrainLoop(
    train_fn = probotSingleEpochFlow,
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
    checkpoint_prefix = "flow",
    loss_fn = loss_fn
  )
}
