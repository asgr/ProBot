probotMakeMDN <- function(input_dim, output_dim, mdn_components, hidden_dims = c(128, 256, 256), activation = nnf_relu) {
  nn_module(
    "mlp",
    initialize = function() {
      dims <- c(input_dim, hidden_dims, mdn_components * (2 * output_dim + 1))
      self$layers <- nn_module_list()
      for (i in seq_len(length(dims) - 1)) {
        self$layers$append(nn_linear(dims[i], dims[i + 1]))
      }
    },

    forward = function(x) {
      n_layers <- length(self$layers)
      for (i in seq_len(n_layers - 1)) {
        x <- self$layers[[i]](x)  # Fixed subsetting and activation call
        x <- activation(x)
      }
      x <- self$layers[[n_layers]](x) # Fixed final layer execution
      x
    }
  )
}

.probotUnpackMDN <- function(output, n_components){

  batch_size <- output$size(1)

  output_features <- output$size(2)

  output_dim <- (output_features / n_components - 1) / 2

  stopifnot(output_dim == as.integer(output_dim))

  output_dim <- as.integer(output_dim)

  output <- output$view(
    c(batch_size, n_components, 2 * output_dim + 1)
  )

  list(
    mu = output[,,1:output_dim],
    log10_sigma =
      output[,,(output_dim + 1):(2 * output_dim)],
    logits =
      output[,,2 * output_dim + 1]
  )
}

probotLossMDN <- function(
    output_true,
    output_pred,
    n_components
){

  output_dim <- output_true$size(2)

  p <- .probotUnpackMDN(
    output_pred,
    n_components
  )

  mu <- p$mu

  log10_sigma <- torch_clamp(
    p$log10_sigma,
    min = -5,
    max = 5
  )

  sigma <- 10^log10_sigma

  y_true <- output_true$unsqueeze(2)

  y_true <- y_true$expand(
    c(
      y_true$size(1),
      n_components,
      output_dim
    )
  )

  z <- (y_true - mu) / sigma

  log_prob <- -0.5 * (
    z^2 +
      2 * log10_sigma * log(10) +
      log(2 * pi)
  )

  log_prob <- log_prob$sum(dim = 3)

  log_pi <- nnf_log_softmax(
    p$logits,
    dim = 2
  )

  log_mix_prob <- torch_logsumexp(
    log_pi + log_prob,
    dim = 2
  )

  -log_mix_prob$mean()
}

