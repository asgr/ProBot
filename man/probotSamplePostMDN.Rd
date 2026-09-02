\name{probotPost}
\alias{probotPost}
\alias{probotSamplePostMDN}
\alias{probotMarginalPostMDN}
\title{Sample and Summarize Posteriors from MDN Outputs}
\description{
Draw posterior samples from a trained Mixture Density Network given conditioning
inputs, or compute marginal posterior means and standard deviations from a set of
MDN outputs.
}
\usage{
probotSamplePostMDN(
  input,
  model,
  mdn_components,
  n_samples = 5000,
  col_means = NULL,
  col_sds = NULL,
  col_names = NULL,
  device = NULL,
  batch_size = NULL,
  verbose = FALSE
)

probotMarginalPostMDN(mdn_output, col_means, col_sds, col_names = NULL)
}
\arguments{
  \item{input}{Numeric vector (single observation) or matrix (multiple observations,
  rows = observations), or a torch tensor, of conditioning inputs. R vectors and
  matrices are automatically converted to tensors and placed on the model's device.
  These must be scaled in the same way as the training inputs.}

  \item{model}{An instantiated MDN \code{\link[torch]{nn_module}} object (e.g., from
  \code{\link{probotMakeMDN}}). The model is set to evaluation mode automatically.}

  \item{mdn_components}{Number of mixture components used when the model was trained.}

  \item{n_samples}{Number of posterior samples to draw per observation. Defaults to 5000.}

  \item{col_means}{Optional numeric vector of centering values used for back-transforming
  the MDN output, as stored by \code{\link{probotScaleForward}}. If \code{NULL} (default),
  samples are returned on the scaled (standardised) output scale.}

  \item{col_sds}{Optional numeric vector of scale values used for back-transforming the
  MDN output. Must be provided whenever \code{col_means} is provided.}

  \item{col_names}{Character vector of column names to assign to the returned columns.}

  \item{device}{Torch device string (e.g., \code{"cpu"}, \code{"mps"}). Defaults to
  \code{NULL}, which auto-detects from the model's parameters.}

  \item{batch_size}{Number of observations processed per chunk. If \code{NULL} (default),
  it is chosen automatically so that roughly \code{2e6} draw rows
  (\code{batch_size * n_samples}) are held in memory at once; peak memory also scales with
  the number of parameters. Lower this on machines with less RAM, or raise it if more
  headroom is available.}

  \item{verbose}{Logical; if \code{TRUE} and processing in multiple chunks, print a
  progress message at regular intervals. Defaults to \code{FALSE}.}

  \item{mdn_output}{List returned by \code{probotPredictMDN()}.}
}
\details{
\code{probotSamplePostMDN()} runs the forward pass itself, so a single call replaces the
former \code{probotPredictMDN()} + \code{probotSamplePostMDN(mdn_output, index = i)} pair.
Component labels are sampled from the softmax weights and Gaussian draws taken per
component, matching the \code{log10_sigma} soft clamp applied during training (sigma is
capped at 1e5). When \code{col_means} and \code{col_sds} are supplied, draws are
back-transformed to physical units.

All randomness comes from the R global RNG, so calling \code{\link{set.seed}} beforehand
makes the draws reproducible.

Observations are processed in chunks of \code{batch_size} rows, which bounds the
intermediate working set. The full output array (roughly 8 bytes per element of
\code{n_samples * D * N}) is allocated up front and dominates memory when sampling many
observations. Note that a given \code{set.seed()} value
produces different draws depending on how many observations are passed in a single call,
since the mixture draws are consumed in chunks; for exact reproduction of a single
observation's samples, pass that observation alone.
}
\value{
\code{probotSamplePostMDN()}: if \code{input} is a vector or a single-row matrix, a numeric
matrix of shape \eqn{n\_samples \times D}{n_samples x D}. If \code{input} has multiple rows,
a numeric array of shape \eqn{n\_samples \times D \times N}{n_samples x D x N}, with
dimension names \code{Sample}, \code{Parameter}, and \code{Observation} set when
\code{col_names} is given. Here \eqn{D} is the model's output dimension.

\code{probotMarginalPostMDN()}: List with \code{post_mean} and \code{post_sd} matrices.
}
\author{
Aaron Robotham
}
\seealso{
\code{\link{probotPredictMDN}}, \code{\link{probotScaleBackward}},
\code{\link{probotSamplePostNF}}, \code{\link{probotAssess}}
}
\examples{
if (requireNamespace("torch", quietly = TRUE)) {
  set.seed(1)
  torch::torch_manual_seed(1)

  model <- probotMakeMDN(input_dim = 2, output_dim = 1, mdn_components = 2)()
  x <- matrix(rnorm(8), ncol = 2)

  # Single observation: a vector in, an (n_samples x 1) matrix out
  draws <- probotSamplePostMDN(
    input = x[1, ],
    model = model,
    mdn_components = 2,
    n_samples = 20,
    col_means = 0,
    col_sds = 1,
    col_names = "theta"
  )
  dim(draws)

  # Multiple observations: an (n_samples x 1 x nrow) array out
  many <- probotSamplePostMDN(x, model, mdn_components = 2, n_samples = 20)
  dim(many)

  # The marginal summary still works from a precomputed prediction
  pred <- probotPredictMDN(x, model, mdn_components = 2)
  marg <- probotMarginalPostMDN(
    mdn_output = pred,
    col_means = 0,
    col_sds = 1,
    col_names = "theta"
  )
  dim(marg$post_mean)
}
}
