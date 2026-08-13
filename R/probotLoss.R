probotLossMDN <- function(
    output_true,
    output_pred,
    mdn_components
){

  if (length(output_true$shape) == 1) output_true <- output_true$unsqueeze(2)

  output_dim <- output_true$size(2)

  p <- .probotUnpackMDN(output_pred, mdn_components)

  mu <- p$mu

  log10_sigma <- torch_clamp(p$log10_sigma, min = -5, max = 5)

  sigma <- 10^log10_sigma

  y_true <- output_true$unsqueeze(2)

  y_true <- y_true$expand(c(y_true$size(1), mdn_components, output_dim))

  z <- (y_true - mu) / sigma

  log_prob <- -0.5 * (z^2 + 2 * log10_sigma * log(10) + log(2 * pi))

  log_prob <- log_prob$sum(dim = 3)

  log_pi <- nnf_log_softmax(p$logits, dim = 2)

  log_mix_prob <- torch_logsumexp(log_pi + log_prob, dim = 2)

  -log_mix_prob$mean()
}

probotLossNF <- function(output_true, output_pred, model) {
  # output_true: (Batch, D) - The true parameters from your simulation
  # output_pred: (Batch, C) - The observed data / conditioning variables

  out <- model$forward(output_true, output_pred)
  z <- out$z
  log_det_jac <- out$log_det_jac
  
  # Base distribution: Standard Normal N(0, I)
  # log p_base(z) = -0.5 * sum(z^2 + log(2*pi)) across dimensions
  log_p_z <- -0.5 * (z^2 + log(2 * pi))$sum(dim = 2)$unsqueeze(1)
  
  # Total log likelihood: log p(true_theta | x) = log p_base(z) + log|det(J)|
  log_likelihood <- log_p_z + log_det_jac
  
  # Return Negative Log Likelihood (to be minimized by optimizer)
  -log_likelihood$mean()
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

probotLossMAE <- function(
    output_true,
    output_pred,
    mdn_components
) {
  if (length(output_true$shape) == 1) output_true <- output_true$unsqueeze(2)
  
  p <- .probotUnpackMDN(output_pred, mdn_components)
  weights <- nnf_softmax(p$logits, dim = 2)
  mu_mix <- (weights$unsqueeze(3) * p$mu)$sum(dim = 2)
  
  torch::torch_abs(output_true - mu_mix)$mean()
}

probotLossMAPE <- function(
    output_true,
    output_pred, 
    mdn_components
) {
  if (length(output_true$shape) == 1) output_true <- output_true$unsqueeze(2)
  
  p <- .probotUnpackMDN(output_pred, mdn_components)
  weights <- nnf_softmax(p$logits, dim = 2)
  mu_mix <- (weights$unsqueeze(3) * p$mu)$sum(dim = 2)
  
  # Prevent division by zero
  denom <- torch::torch_clamp(torch::torch_abs(output_true), min = 1e-8)
  loss <- torch::torch_abs((output_true - mu_mix) / denom)$mean()
  
  return(loss * 100) # Returned as a percentage
}

probotLossHuber <- function(
    output_true, 
    output_pred, 
    mdn_components, 
    delta = 1.0
) {
  if (length(output_true$shape) == 1) output_true <- output_true$unsqueeze(2)
  
  p <- .probotUnpackMDN(output_pred, mdn_components)
  weights <- nnf_softmax(p$logits, dim = 2)
  mu_mix <- (weights$unsqueeze(3) * p$mu)$sum(dim = 2)
  
  error <- output_true - mu_mix
  abs_error <- torch::torch_abs(error)
  
  quad_part <- 0.5 * (error^2)
  linear_part <- delta * (abs_error - 0.5 * delta)
  loss <- torch::torch_where(abs_error <= delta, quad_part, linear_part)
  
  loss$mean()
}
