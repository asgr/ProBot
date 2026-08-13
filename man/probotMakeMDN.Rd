\name{probotMakeMDN}
\alias{probotMakeMDN}
\title{Construct a Mixture Density Network Module}
\description{
Construct an \code{\link[torch]{nn_module}} implementing a Multi-Layer Perceptron
with a Mixture Density Network (MDN) output layer. The network maps input features
to a mixture of Gaussian components over the output parameters.
}
\usage{
probotMakeMDN(
  input_dim,
  output_dim,
  mdn_components,
  hidden_dims = c(128, 256, 256),
  activation = nnf_relu,
  dropout = 0,
  device = NULL
)
}
\arguments{
    \item{input_dim}{Number of input features.}

    \item{output_dim}{Number of output parameters (each modeled as a mixture distribution).}

    \item{mdn_components}{Number of Gaussian mixture components in the MDN output layer.}

    \item{hidden_dims}{Numeric vector specifying the size of each hidden layer.
     Defaults to \code{c(128, 256, 256)}.}

    \item{activation}{Activation function applied after each hidden layer.
     Defaults to \code{\link[torch]{nnf_relu}}. Also accepts \code{\link[torch]{nnf_silu}} or
     \code{\link[torch]{nnf_gelu}}.}

    \item{dropout}{Dropout rate (fraction of activations to zero out during training). Defaults to 0 (no dropout).}

    \item{device}{Character string specifying the computing device: \code{"cpu"}, \code{"cuda"}, or \code{"mps"}. If \code{NULL} (default), auto-detects \code{"mps"} on Apple Silicon systems, otherwise falls back to \code{"cpu"}}
}
\details{
The final output layer has \code{mdn_components * (2 * output_dim + 1)} units,
encoding \code{output_dim} means, \code{output_dim} log10-sigmas, and 1 logit
per mixture component. The output is unpacked by \code{\link[=probotPredictMDN]{probotPredictMDN()}}.

When \code{device = NULL}, the network attempts to use MPS (Apple GPU acceleration) if
available, otherwise falls back to CPU. Pass an explicit device string to override.
}
\value{
An \code{\link[torch]{nn_module}} that must be called as a function
(e.g., \code{probotMakeMDN(2, 2, 3)()}) to instantiate the model object.
}
\author{
Aaron Robotham
}
\seealso{
\code{\link{probotPredictMDN}}, \code{\link{probotTrainMDN}}
}
\examples{
\dontrun{
## Create and inspect model on default device (auto-detected)
## Note the explicit () required to trigger model creation
model <- probotMakeMDN(input_dim = 2, output_dim = 2, mdn_components = 3, device = "cpu")()
}
}
