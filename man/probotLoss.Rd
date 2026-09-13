\name{probotLoss}
\alias{probotLoss}
\alias{probotLossMDN}
\alias{probotLossNF}
\alias{probotLossMSE}
\alias{probotLossMAE}
\alias{probotLossMAPE}
\alias{probotLossHuber}
\title{Compute Loss Values for Model Outputs}
\description{
Compute loss values for different network types. \code{probotLossMDN} computes mean negative log-likelihood of targets under a Gaussian mixture. \code{probotLossNF} computes the negative log likelihood loss for training a normalizing flow model. \code{probotLossMSE} computes Mean Squared Error. \code{probotLossMAE} computes Mean Absolute Error. \code{probotLossMAPE} computes Mean Absolute Percentage Error (as a percentage). \code{probotLossHuber} computes the Huber loss.
}
\usage{
probotLossMDN(output_true, output_pred, model)

probotLossNF(output_true, output_pred, model)

probotLossMSE(output_true, output_pred, model)

probotLossMAE(output_true, output_pred, model)

probotLossMAPE(output_true, output_pred, model)

probotLossHuber(output_true, output_pred, model, delta = 1.0)
}
\arguments{
  \item{output_true}{Torch tensor of true targets with shape
  \code{batch x output_dim}.}

  \item{output_pred}{Torch tensor of raw MDN outputs from the network.}

  \item{model}{The model that produced \code{output_pred}. For the MDN family
  this is the module from \code{\link{probotMakeMDN}}; it is used only to read
  off the number of mixture components, because the raw output width
  \code{K * (2D + 1)} does not identify \code{K} and \code{D} on its own. For
  \code{probotLossNF} it is the flow from \code{\link{probotMakeFlow}}, used to
  compute the forward transformation and Jacobian. For \code{probotLossNF},
  \code{output_true} is the parameters tensor (\code{batch x D}) and
  \code{output_pred} is the conditioning data (\code{batch x C}).}

  \item{delta}{Threshold for the Huber loss: errors below \code{delta} are penalized quadratically, errors above are penalized linearly. Only used by \code{probotLossHuber}. Default is 1.0.}
}
\details{
The MDN family of loss functions (\code{probotLossMDN}, \code{probotLossMSE},
\code{probotLossMAE}, \code{probotLossMAPE}, \code{probotLossHuber}) all unpack
the raw network output into predicted means, scales, and mixture logits using
an internal helper. The number of components is read from \code{model} rather
than passed by the caller: an MDN head emits \code{K * (2D + 1)} values, which
does not identify \code{K} and \code{D} separately (25 units is equally a
5-component network over 2 parameters and a 1-component network over 12), so
the tensor alone is ambiguous. \code{probotMakeMDN()} stores both numbers, and
\code{probotLoad()} reconstructs them, so any model built by this package can
answer the question. See \code{\link{probotMakeMDN}}.
For \code{probotLossMDN}, numerically stable log-sum-exp
aggregation over components yields the mean negative log-likelihood.
For \code{probotLossMSE}, \code{probotLossMAE}, \code{probotLossMAPE}, and
\code{probotLossHuber}, the predicted mixture-mean (weighted average of
component means) is compared to the true targets using squared error,
absolute error, absolute percentage error, or Huber loss respectively.

\code{probotLossNF} applies the flow's forward transformation to map parameters
to latent space and combines the base Gaussian log-likelihood with the
log-determinant of the Jacobian to give the exact parameter log-likelihood.
}
\value{
A scalar torch tensor containing the loss value.
}
\author{
Aaron Robotham
}
\seealso{
\code{\link{probotMakeMDN}}, \code{\link{probotMakeFlow}}, \code{\link{probotSingleEpochMDN}}, \code{\link{probotSingleEpochFlow}}
}
\examples{
if (requireNamespace("torch", quietly = TRUE)) {
  torch::torch_manual_seed(1)

  ## MDN loss
  model_mdn <- probotMakeMDN(input_dim = 2, output_dim = 1, mdn_components = 2, device = "cpu")()
  x <- torch::torch_randn(c(4, 2))
  y <- torch::torch_randn(c(4, 1))

  pred <- model_mdn(x)
  loss_mdn <- probotLossMDN(y, pred, model_mdn)
  as.numeric(loss_mdn$item())

  ## Flow loss
  model_nf <- probotMakeFlow(input_dim = 3, output_dim = 2, n_layers = 2, device = "cpu")()
  theta <- torch::torch_randn(c(4, 2), device = "cpu")
  ctx <- torch::torch_randn(c(4, 3), device = "cpu")

  loss_nf <- probotLossNF(theta, ctx, model_nf)
  as.numeric(loss_nf$item())
}
}
