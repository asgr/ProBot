\name{probotLoss}
\alias{probotLoss}
\alias{probotLossMDN}
\alias{probotLossMSE}
\alias{probotLossMAE}
\alias{probotLossMAPE}
\alias{probotLossHuber}
\title{Compute Negative Log-Likelihood for an MDN Output}
\description{
Various useful loss functions. Compute the Mixture Density Network (MDN) loss as mean negative log-likelihood of targets under a Gaussian mixture. Compute the Mean Standard Error (MSE) loss. Compute the Mean Absolute Error (MAE) loss. Compute the Mean Absolute Percentage Error (MAPE) loss. Compute the Huber loss.
}
\usage{
probotLossMDN(output_true, output_pred, mdn_components)

probotLossMSE(output_true, output_pred, mdn_components)

probotLossMAE(output_true, output_pred, mdn_components)

probotLossMAPE(output_true, output_pred, mdn_components)

probotLossHuber(output_true, output_pred, mdn_components)
}
\arguments{
  \item{output_true}{Torch tensor of true targets with shape
  \code{batch x output_dim}.}

  \item{output_pred}{Torch tensor of raw MDN outputs from the network.}

  \item{mdn_components}{Number of Gaussian mixture components used by the model.}
}
\details{
The function unpacks predicted means, log-scales, and mixture logits; applies numerically stable log-sum-exp aggregation over components; and returns the average negative log-likelihood (\code{probotLossMDN}) or MSE (\code{probotLossMSE}).
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

  model <- probotMakeMDN(input_dim = 2, output_dim = 1, mdn_components = 2)()
  x <- torch::torch_randn(c(4, 2))
  y <- torch::torch_randn(c(4, 1))

  pred <- model(x)
  loss <- probotLossMDN(y, pred, mdn_components = 2)

  as.numeric(loss$item())
}
}
