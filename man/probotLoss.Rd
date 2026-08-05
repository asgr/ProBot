\name{probotLossMDN}
\alias{probotLossMDN}
\title{Compute Negative Log-Likelihood for an MDN Output}
\description{
Compute the Mixture Density Network loss as mean negative log-likelihood of
targets under a Gaussian mixture.
}
\usage{
probotLossMDN(output_true, output_pred, mdn_components)
}
\arguments{
  \item{output_true}{Torch tensor of true targets with shape
  \code{batch x output_dim}.}

  \item{output_pred}{Torch tensor of raw MDN outputs from the network.}

  \item{mdn_components}{Number of Gaussian mixture components used by the model.}
}
\details{
The function unpacks predicted means, log-scales, and mixture logits; applies numerically stable log-sum-exp aggregation over components; and returns the average negative log-likelihood.
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
