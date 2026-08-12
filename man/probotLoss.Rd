\name{probotLoss}
\alias{probotLoss}
\alias{probotLossMDN}
\alias{probotLossNF}
\alias{probotLossMSE}
\alias{probotLossMAE}
\alias{probotLossMAPE}
\alias{probotLossHuber}
\title{Compute Loss Values for an MDN Output}
\description{
Compute loss values for different networks. \code{probotLossMDN} computes mean negative log-likelihood of targets under a Gaussian mixture. \code{probotLossFlow} computes the negative log likelihood loss for training a normalizing flow model. \code{probotLossMSE} computes Mean Squared Error. \code{probotLossMAE} computes Mean Absolute Error. \code{probotLossMAPE} computes Mean Absolute Percentage Error (as a percentage). \code{probotLossHuber} computes the Huber loss.
}
\usage{
probotLossMDN(output_true, output_pred, mdn_components)

probotLossNF(output_true, output_pred, model)

probotLossMSE(output_true, output_pred, mdn_components)

probotLossMAE(output_true, output_pred, mdn_components)

probotLossMAPE(output_true, output_pred, mdn_components)

probotLossHuber(output_true, output_pred, mdn_components, delta = 1.0)
}
\arguments{
  \item{output_true}{Torch tensor of true targets with shape
  \code{batch x output_dim}.}

  \item{output_pred}{Torch tensor of raw MDN outputs from the network.}

  \item{mdn_components}{Number of Gaussian mixture components used by the model.}
  
  \item{model}{The flow model to use (see \code{\link{probotMakeFlow}}) to compute loss.}

  \item{delta}{Threshold for the Huber loss: errors below \code{delta} are penalized quadratically, errors above are penalized linearly. Only used by \code{probotLossHuber}. Default is 1.0.}
}
\details{
All functions unpack the raw network output into predicted means, scales, and mixture logits using an internal helper. For \code{probotLossMDN}, numerically stable log-sum-exp aggregation over components yields the mean negative log-likelihood. For \code{probotLossMSE}, \code{probotLossMAE}, \code{probotLossMAPE}, and \code{probotLossHuber}, the predicted mixture-mean (weighted average of component means) is compared to the true targets using squared error, absolute error, absolute percentage error, or Huber loss respectively.
}
\value{
A scalar torch tensor containing the loss value.
}
\author{
Aaron Robotham
}
\seealso{
\code{\link{probotMakeMDN}}, \code{\link{probotSingleEpochMDN}}
}
\examples{
if (requireNamespace("torch", quietly = TRUE)) {
  torch::torch_manual_seed(1)

  model <- probotMakeMDN(input_dim = 2, output_dim = 1, mdn_components = 2, device='cpu')()
  x <- torch::torch_randn(c(4, 2))
  y <- torch::torch_randn(c(4, 1))

  pred <- model(x)
  loss <- probotLossMDN(y, pred, mdn_components = 2)

  as.numeric(loss$item())
}
}
