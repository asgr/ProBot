probotMakePoint <- function(input_dim, output_dim,
                            hidden_dims = c(128, 256, 256),
                            activation = nnf_relu,
                            dropout = 0,
                            device = NULL) {
  
  nn_module(
    "mlp_point",
    initialize = function() {
      self$activation_fn <- activation
      self$dropout_rate <- dropout
      
      dims <- c(input_dim, hidden_dims, output_dim)
      
      layer_list <- list()
      for (i in seq_len(length(dims) - 1)) {
        layer_list[[i]] <- nn_linear(dims[i], dims[i + 1])
      }
      
      self$layers <- nn_module_list(layer_list)
      
      target_device <- .probotChooseDevice(device)
      
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
