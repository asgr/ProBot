probotDataLoader <- function(input,
                              output,
                              batch = 1024,
                              shuffle = TRUE,
                              train_idx = NULL,
                              device = NULL) { # Added device parameter with a safe default
  if (is.null(train_idx)) {
    train_idx = 1:nrow(input)
  }
  
  #Determine device: Use provided device, or auto-detect MPS / CPU fallback
  device_torch <- .probotChooseDevice(device)
  
  # Allocate tensors directly onto the specified target hardware
  input_train <- torch_tensor(input[train_idx, ], dtype = torch_float(), device = device_torch)
  
  output_train <- torch_tensor(output[train_idx, ], dtype = torch_float(), device = device_torch)
  
  dataset <- tensor_dataset(input_train, output_train)
  
  dataloader <- dataloader(dataset, batch_size = batch, shuffle = shuffle)
  
  return(dataloader)
}
