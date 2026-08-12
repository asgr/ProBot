\name{probotCouplingLayer}
\alias{probotCouplingLayer}
\title{Construct a Coupling Layer Module for Normalizing Flows}
\description{
Construct an \code{\link[torch]{nn_module}} implementing a single coupling layer
for normalizing flows. The layer splits the parameter space in half, uses one half
combined with context features to predict an affine transformation (shift and scale)
applied to the other half, and then swaps the halves for full mixing across layers.
}
\usage{
probotCouplingLayer(dim_theta, dim_x, hidden_dim = 32,    soft_clamp = 3, device = NULL)
}
\arguments{
    \item{dim_theta}{Total dimensionality of the parameter space (i.e., number of output parameters).}

    \item{dim_x}{Dimensionality of the context/input features used to condition the transformation.}

    \item{hidden_dim}{Number of units in each hidden layer of the internal shift/scale network. Defaults to 32.}
    
    \item{soft_clamp}{Encourages narrower posteriors for larger values. Should probably be between 1-5.}
    
    \item{device}{Character specifying the computing device: \code{"cpu"}, \code{"cuda"}, or \code{"mps"}. If \code{NULL} (default), the device is automatically inferred from \code{model}; if the model has no parameters, falls back to \code{"mps"} if available otherwise \code{"cpu"}.}
}
\details{
The coupling layer splits \code{dim_theta} into two halves (\code{d1} and \code{d2}),
concatenates the first half with the context features, and passes the result through
a small MLP (two hidden layers with GeLU activation) to predict shift and log-scale
parameters for the second half. The log-scale is clamped via \code{\link[torch]{torch_tanh}}
for numerical stability.

The forward pass applies the affine transformation
\eqn{z_2 = (\theta_2 - t) \cdot \exp(s)} and returns the transformed latent
representation along with the log determinant of the Jacobian.

The inverse pass reverses the transformation:
\eqn{\theta_2 = z_2 / \exp(s) + t}, enabling posterior sampling from the base distribution.

Halves are swapped after each transformation to ensure full mixing across subsequent
coupling layers.
}
\value{
An instantiated \code{\link[torch]{nn_module}} object ready for use.
The call creates the module directly (no additional parentheses needed).
}
\author{
Aaron Robotham
}
\seealso{
\code{\link{probotMakeFlow}}, \code{\link{probotLossNF}}
}
\examples{
if (requireNamespace("torch", quietly = TRUE)) {
## Create a coupling layer for 2 parameters conditioned on 3 context features
layer <- probotCouplingLayer(dim_theta = 2, dim_x = 3, device='cpu')()

## Forward and inverse passes require torch tensors
theta <- torch::torch_randn(c(10, 2))
x <- torch::torch_randn(c(10, 3))

out <- layer$forward(theta, x)
names(out)   # "z" and "log_det_jac"
}
}
