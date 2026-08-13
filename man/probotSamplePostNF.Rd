\name{probotSamplePostNF}
\alias{probotSamplePostNF}
\title{Sample Posterior Parameters from a Normalizing Flow Model}
\description{
Draw samples from the posterior distribution of parameters conditioned on observed data,
using a trained normalizing flow model. Samples are drawn from a standard normal base
distribution and transformed through the inverse flow.
}
\usage{
probotSamplePostNF(
  context_x,
  model,
  n_samples = 5000,
  col_means = NULL,
  col_sds = NULL,
  col_names = NULL,
  dim_theta = NULL
)
}
\arguments{
   \item{context_x}{Torch tensor of context/input features. Can be 1D (single observation)
   or 2D (\eqn{Batch \times C}).}

   \item{model}{An instantiated normalizing flow \code{\link[torch]{nn_module}} object
   (e.g., from \code{\link{probotMakeFlow}}). The model is set
   to evaluation mode automatically.}

   \item{n_samples}{Number of posterior samples to draw. Defaults to 5000.}

   \item{col_means}{Numeric vector of column means for inverse scaling (unstandardizing)
   the sampled parameters. If provided, samples are transformed back to the original
   parameter space using \code{\link[=probotScaleBackward]{probotScaleBackward()}}.}

   \item{col_sds}{Numeric vector of column standard deviations for inverse scaling.
   Ignored if \code{col_means} is \code{NULL}.}

   \item{col_names}{Character vector of column names to assign to the output matrix.}

   \item{dim_theta}{Integer specifying the parameter dimensionality. Only needed when
   \code{col_means} is \code{NULL} (i.e., when no scaling is applied). If both
   \code{dim_theta} and \code{col_means} are \code{NULL}, the function throws an error.}
}
\details{
The model is set to evaluation mode (\code{model$eval()}) and all
transformations are computed inside a \code{\link[torch]{with_no_grad}} block
to avoid unnecessary memory allocations.

If \code{context_x} is 1D, it is expanded to match \code{n_samples} rows so that
each sample is conditioned on the same observed data. If \code{context_x} already
has \code{n_samples} rows, it is used as-is.

When \code{col_means} is provided, the sampled parameters are automatically
unstandardized using \code{\link{probotScaleBackward}},
which reverses the scaling applied during training.
}
\value{
A numeric matrix of shape \eqn{n\_samples \times D}, where \eqn{D} is the number
of parameters. Column names are set from \code{col_names} if provided.
}
\author{
Aaron Robotham
}
\seealso{
\code{\link{probotMakeFlow}}, \code{\link{probotLossNF}},
\code{\link{probotScaleBackward}}, \code{\link{probotSamplePostMDN}}
}
\examples{
\dontrun{
## Sample posterior from a trained flow model
x <- torch::torch_tensor(c(1.2, 0.5))

samples <- probotSamplePostNF(
  context_x = x,
  model = flow_model,
  n_samples = 5000,
  col_means = c(mean1, mean2),
  col_sds = c(sd1, sd2),
  col_names = c("param_A", "param_B")
)

dim(samples)  # 5000 x 2
}
}
