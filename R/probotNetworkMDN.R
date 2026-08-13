probotMakeMDN <- function(input_dim, output_dim, mdn_components,
                          hidden_dims = c(128, 256, 256), activation = nnf_relu, 
                          dropout = 0, device = NULL) {
  
  nn_module(
    "mlp",
    initialize = function() {
      self$activation_fn <- activation
      self$dropout_rate  <- dropout
      
      dims <- c(input_dim, hidden_dims, mdn_components * (2 * output_dim + 1))
      
      layer_list <- list()
      for (i in seq_len(length(dims) - 1)) {
        layer_list[[i]] <- nn_linear(dims[i], dims[i + 1])
      }
      
      self$layers <- nn_module_list(layer_list)
      
      #Determine device: Use provided device, or auto-detect MPS / CPU fallback
      target_device <- .probotChooseDevice(device)
      
      # 3. CRITICAL: Automatically move the entire module structure to the target hardware
      self$to(device = target_device)
    },
    
    forward = function(x) {
      n_layers <- length(self$layers)
      for (i in seq_len(n_layers - 1)) {
        x <- self$layers[[i]](x)
        x <- self$activation_fn(x)
        if (self$dropout_rate > 0) {
          x <- nnf_dropout(x, p = self$dropout_rate, training = self$training)
        }
      }
      x <- self$layers[[n_layers]](x)
      x
    }
  )
}

.probotUnpackMDN <- function(output, mdn_components){

  batch_size <- output$size(1)

  output_features <- output$size(2)

  output_dim <- (output_features / mdn_components - 1) / 2

  stopifnot(output_dim == as.integer(output_dim))

  output_dim <- as.integer(output_dim)

  output <- output$view(
    c(batch_size, mdn_components, 2 * output_dim + 1)
  )

  list(
    mu = output[,,1:output_dim],
    log10_sigma =
      output[,,(output_dim + 1):(2 * output_dim)],
    logits =
      output[,,2 * output_dim + 1]
  )
}
