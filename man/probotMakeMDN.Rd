\name{probotMakeMDN}
\alias{probotMakeMDN}
\title{Construct a Fully Connected Mixture Density Network}
\description{
Create a torch \code{nn_module} generator for a multilayer perceptron whose
output parameterizes a Gaussian mixture.
}
\usage{
probotMakeMDN(
  input_dim,
  output_dim,
  mdn_components,
  hidden_dims = c(128, 256, 256),
  activation = nnf_relu,
  dropout = 0
)
}
\arguments{
  \item{input_dim}{Number of predictor features.}

  \item{output_dim}{Number of target dimensions.}

  \item{mdn_components}{Number of Gaussian mixture components in model.}

  \item{hidden_dims}{Integer vector of hidden layer widths.}

  \item{activation}{Activation function applied between hidden layers. Stored
  as \code{$activation_fn} on the instantiated module.}

  \item{dropout}{Dropout probability applied after each hidden activation
  during training. Set to \code{0} (the default) to disable dropout.}
}
\details{
The final layer size is \code{mdn_components * (2 * output_dim + 1)} to encode,
per component, \code{output_dim} means, \code{output_dim} log10 standard
deviations, and one mixture logit. The activation function and dropout rate are
stored as \code{$activation_fn} and \code{$dropout_rate} on the instantiated
module for easy inspection and reuse.
}
\value{
A torch module generator; instantiate with \code{probotMakeMDN(... )()}.
}
\author{
Aaron Robotham
}
\seealso{
\code{\link{probotPredictMDN}}, \code{\link{probotLossMDN}}
}
\examples{
if (requireNamespace("torch", quietly = TRUE)) {
  net <- probotMakeMDN(input_dim = 3, output_dim = 2, mdn_components = 2)
  model <- net()

  x <- torch::torch_randn(c(5, 3))
  out <- model(x)

  as.integer(out$size(2))
}
}
