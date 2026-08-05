probotLossMDN <- function(
    output_true,
    output_pred,
    mdn_components
){

  if (length(output_true$shape) == 1) output_true <- output_true$unsqueeze(2)

  output_dim <- output_true$size(2)

  p <- .probotUnpackMDN(
    output_pred,
    mdn_components
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
      mdn_components,
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

probotLossMSE <- function(
    output_true,
    output_pred,
    mdn_components
) {
  # Ensure output_true has dimensions (batch_size, output_dim)
  if (length(output_true$shape) == 1) {
    output_true <- output_true$unsqueeze(2)
  }

  # Unpack the raw network output into MDN parameters
  p <- .probotUnpackMDN(output_pred, mdn_components)

  # Convert logits to normalized component weights
  weights <- nnf_softmax(p$logits, dim = 2)

  # Compute the expected value (weighted mean of all components)
  mu_mix <- (weights$unsqueeze(3) * p$mu)$sum(dim = 2)

  # Return classic Mean Squared Error
  ((output_true - mu_mix)^2)$mean()
}
