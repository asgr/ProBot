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
  input,
  model,
  n_samples = 5000,
  col_means = NULL,
  col_sds = NULL,
  col_names = NULL,
  output_dim = NULL,
  device = NULL,
  batch_size = NULL,
  verbose = FALSE,
  point_estimate = FALSE
)
}
\arguments{
   \item{input}{Numeric vector (single observation) or matrix (multiple observations, where rows are observations and columns are parameters), or a torch tensor. Accepts raw R vectors/matrices which are automatically converted to tensors.}

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

   \item{output_dim}{Integer specifying the output parameter dimensionality. Only needed when
   \code{col_means} is \code{NULL} (i.e., when no scaling is applied). If both
   \code{output_dim} and \code{col_means} are \code{NULL}, the function throws an error.}

   \item{device}{Torch device string (e.g., "cpu", "mps"). Defaults to \code{NULL},
   which auto-detects from the model's parameters.}

   \item{batch_size}{Number of observations processed per chunk in multi-observation
   mode. If \code{NULL} (default), it is chosen automatically so that roughly
   \code{2e6} rows (\code{batch_size * n_samples}) of latent space are held in
   memory at once. Lower this on machines with less RAM, or raise it if more
   headroom is available. Ignored in single-observation mode.}

   \item{verbose}{Logical; if \code{TRUE} and processing in multiple chunks,
   print a progress message at regular intervals.}

   \item{point_estimate}{Logical. If \code{TRUE}, return one deterministic
   point estimate per row of \code{input} instead of drawing samples: the
   location head \eqn{\mu(x)} for a \code{\link{probotFlowLoc}} model, or the
   inverse of \code{z = 0} (the base distribution's mode) for a plain flow. One
   sweep over \code{input}, no \code{n_samples} cost, and the result scales with
   the number of rows rather than \code{rows * n_samples}. \code{col_means} /
   \code{col_sds} unscale it as usual; \code{n_samples} and \code{batch_size} are
   ignored, and \code{verbose} warns if it was set. Defaults to \code{FALSE}.}
}
\details{
The model is set to evaluation mode (\code{model$eval()}) and all
transformations are computed inside a \code{\link[torch]{with_no_grad}} block
to avoid unnecessary memory allocations.

If \code{input} is an R vector or matrix, it is automatically converted to a
torch tensor and placed on the same device as the model. If \code{input} is
1D, it is expanded to match \code{n_samples} rows so that each sample is
conditioned on the same observed data.

When \code{col_means} is provided, the sampled parameters are automatically
unstandardized on-device using a single fused broadcast, which reverses the
scaling applied during training.

In multi-observation mode, observations are processed in chunks of
\code{batch_size} rows to keep peak memory bounded, so very large samples
(millions of observations) can be handled without exhausting memory. Only the
accumulated output array and one chunk of intermediate tensors are held at a
time.
}
\value{
If \option{input} is a vector, returns a numeric matrix of shape \eqn{n\_samples \times D}, where \eqn{D} is the number
of parameters. Column names are set from \code{col_names} if provided.

If \option{input} is a matrix, returns a numeric array of shape
\eqn{n\_samples \times D \times Nrow}, where \eqn{D} is the number of
parameters and \eqn{Nrow} is the number of rows in \option{input}. The
\code{Sample}, \code{Parameter}, and \code{Observation} dimension names are
set from \code{col_names} and the row/observation indices if provided.

With \code{point_estimate = TRUE}, a matrix of shape \eqn{Nrow \times D} (or a
length-\eqn{D} vector when \code{input} is a vector). The single-row and
multi-row paths use different float32 GEMM kernels, so a one-row result matches
the corresponding row of a larger batch only to about \eqn{10^{-6}} relative.
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
## Sample posterior from a trained flow model, passing a plain R vector
samples <- probotSamplePostNF(
  input = c(1.2, 0.5),
  model = flow_model,
  n_samples = 5000,
  col_means = c(mean1, mean2),
  col_sds = c(sd1, sd2),
  col_names = c("param_A", "param_B")
)

dim(samples)  # 5000 x 2
}
}
