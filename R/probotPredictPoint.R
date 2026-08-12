probotPredictPoint <- function(input, model, device = NULL) {
  
  if (is.null(device)) {
    if (length(model$parameters) > 0) {
      device <- model$parameters[[1]]$device
    } else {
      device <- if (backends_mps_is_available()) torch_device("mps") else torch_device("cpu")
    }
  }
  
  model$eval()
  
  with_no_grad({
    output <- model(
      torch_tensor(
        input,
        dtype = torch_float(),
        device = device
      )
    )
  })
  
  as.matrix(output$to(device = "cpu"))
}

probotPredictPointScaled <- function(input, model, col_means, col_sds, device = NULL) {
  pred_scaled <- probotPredictPoint(input, model, device)
  probotScaleBackward(pred_scaled, col_means, col_sds)
}
