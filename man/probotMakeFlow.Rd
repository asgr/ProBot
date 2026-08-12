\name{probotMakeFlow}
\alias{probotMakeFlow}
\title{Construct a Normalizing Flow Network Module}
\description{
Construct an \code{\link[torch]{nn_module}} implementing a stack of coupling layers
(normalizing flow) for learning a conditional probability distribution. The network
maps parameters to a latent space (forward) and vice versa (inverse).
}
\usage{
probotMakeFlow(dim_theta, dim_x, n_layers = 4, hidden_dim = 32, soft_clamp = 4,   
  device = NULL)
}
\arguments{
    \item{dim_theta}{Total dimensionality of the parameter space.}

    \item{dim_x}{Dimensionality of the context/input features used to condition the flow.}

    \item{n_layers}{Number of coupling layers to stack. Defaults to 4.}

    \item{hidden_dim}{Number of units in each hidden layer of the internal shift/scale network within each coupling layer. Defaults to 32.}
    
    \item{soft_clamp}{Encourages narrower posteriors for larger values. Should probably be between 1-5.}
    
    \item{device}{Character specifying the computing device: \code{"cpu"}, \code{"cuda"}, or \code{"mps"}. If \code{NULL} (default), the device is automatically inferred from \code{model}; if the model has no parameters, falls back to \code{"mps"} if available otherwise \code{"cpu"}.}
}
\details{
The network registers \code{n_layers} coupling layers directly, so parameters are
tracked by torch and device placement is handled automatically.

The forward pass maps parameters \eqn{\theta} to latent space \eqn{z}, accumulating
the log determinant of the Jacobian across all layers. The inverse pass maps from
latent space back to parameter space by iterating through the coupling layers in
reverse order.
}
\value{
An instantiated \code{\link[torch]{nn_module}} object ready for use.
The call creates the module directly (no additional parentheses needed).
}
\author{
Aaron Robotham
}
\seealso{
\code{\link{probotCouplingLayer}}, \code{\link{probotLossNF}}, \code{\link{probotSamplePostNF}}
}
\examples{
if (requireNamespace("torch", quietly = TRUE)) {
## Create a flow network with 4 coupling layers
flow <- probotMakeFlow(dim_theta = 2, dim_x = 3, device='cpu')()

## Forward and inverse passes require torch tensors
theta <- torch::torch_randn(c(10, 2))
x <- torch::torch_randn(c(10, 3))

out <- flow$forward(theta, x)
names(out)   # "z" and "log_det_jac"
}
}
