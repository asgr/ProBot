\name{probotLossNF}
\alias{probotLossNF}
\title{Compute Normalizing Flow Loss (Negative Log Likelihood)}
\description{
Compute the negative log likelihood loss for training a normalizing flow model.
The loss evaluates how well the flow maps observed parameters to a standard normal
base distribution, conditioned on input features.
}
\usage{
probotLossNF(true_theta, context_x, model, device = NULL)
}
\arguments{
   \item{true_theta}{Torch tensor of shape \eqn{Batch \times D} containing the true parameters.}

   \item{context_x}{Torch tensor of shape \eqn{Batch \times C} containing the observed data or conditioning variables.}

   \item{model}{An instantiated normalizing flow \code{\link[torch]{nn_module}} object
   (e.g., from \code{\link[=probotNetworkFlow]{probotNetworkFlow()}}).}

   \item{device}{Character string or \code{\link[torch]{torch_device}} object specifying
   the computing device: \code{"cpu"}, \code{"cuda"}, or \code{"mps"}. If \code{NULL}
   (default), the device is automatically inferred from \code{model}; if the model has
   no parameters, falls back to \code{"cpu"}.}
}
\details{
The loss is computed as the negative log likelihood under the change-of-variables formula:

\deqn{\log p(\theta \mid x) = \log p_{\text{base}}(z) + \log |\det(J)|}

where \eqn{z} is the transformed latent representation,
\eqn{p_{\text{base}}(z)} is a standard normal distribution, and
\eqn{\log |\det(J)|} is the accumulated log determinant of the Jacobian from all
coupling layers.

Both \code{true_theta} and \code{context_x} are automatically moved to the specified
device to avoid device mismatch errors.

The function returns the mean negative log likelihood, which can be directly minimized
by an optimizer (e.g., via \code{torch::nnf_cross_entropy()} or manual backprop).
}
\value{
A scalar \code{\link[torch]{torch_tensor}} containing the mean negative log likelihood.
}
\author{
Aaron Robotham
}
\seealso{
\code{\link{probotNetworkFlow}}, \code{\link{probotSamplePostNF}}, \code{\link{probotLossMDN}}
}
\examples{
\dontrun{
## Create a flow model and compute loss
flow <- probotNetworkFlow(dim_theta = 2, dim_x = 3)()

theta <- torch::torch_randn(c(10, 2))
x <- torch::torch_randn(c(10, 3))

loss <- probotLossNF(theta, x, flow)
loss$item()
}
}
