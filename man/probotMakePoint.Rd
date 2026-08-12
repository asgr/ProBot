\name{probotMakePoint}
\alias{probotMakePoint}
\title{Construct a Point Estimate MLP Module}
\description{
Construct an \code{\link[torch]{nn_module}} implementing a Multi-Layer Perceptron for point estimation.
The network maps input features to point predictions via a standard regression output layer.
}
\usage{
probotMakePoint(
  input_dim,
  output_dim,
  hidden_dims = c(128, 256, 256),
  activation = nnf_relu,
  dropout = 0,
  device = NULL
)
}
\arguments{
    \item{input_dim}{Number of input features.}
    
    \item{output_dim}{Number of output parameters to predict.}
    
    \item{hidden_dims}{Numeric vector specifying the size of each hidden layer.
     Defaults to \code{c(128, 256, 256)}.}
    
    \item{activation}{Activation function applied after each hidden layer.
     Defaults to \code{\link[torch]{nnf_relu}}. Also accepts \code{\link[torch]{nnf_silu}} or
     \code{\link[torch]{nnf_gelu}}.}
    
    \item{dropout}{Dropout rate (fraction of activations to zero out during training).
     Defaults to 0 (no dropout).}
    
    \item{device}{Character string specifying the computing device: \code{"cpu"},
     \code{"cuda"}, or \code{"mps"}. If \code{NULL} (default), auto-detects
     \code{"mps"} on Apple Silicon systems, otherwise falls back to \code{"cpu"}}
}
\details{
The final output layer has \code{output_dim} units producing point predictions.
When \code{device = NULL}, the network attempts to use MPS (Apple GPU acceleration) if
available, otherwise falls back to CPU. Pass an explicit device string to override.
}
\value{
An \code{\link[torch]{nn_module}} that must be called as a function
(e.g., \code{probotMakePoint(2, 2)()}) to instantiate the model object.
}
\author{
Aaron Robotham
}
\seealso{
\code{\link{probotTrainPoint}}, \code{\link{probotPredictPoint}}
}
\examples{
\dontrun{
## Create and inspect model on default device (auto-detected)
model_fn <- probotMakePoint(input_dim = 2, output_dim = 2)
model <- model_fn()

## Explicitly place on GPU/MPS
model_gpu <- probotMakePoint(
  input_dim = 2,
  output_dim = 2,
  device = "mps"
)()
}
}
