\name{probotAssess}
\alias{probotCRPS}
\alias{probotPIT}
\alias{probotTARP}
\title{Posterior Predictive Assessment: CRPS, PIT, and TARP}
\description{
Assess the calibration and sharpness of posterior samples drawn from a
Mixture Density Network (MDN) or a normalising flow. \code{probotCRPS()} computes
the Continuous Ranked Probability Score, \code{probotPIT()} computes Probability
Integral Transform values, and \code{probotTARP()} computes Test of Accuracy with
Random Points coverage values.
}
\usage{
probotCRPS(
  input,
  model,
  mdn_components = NULL,
  params,
  n_test = NULL,
  n_samples = 1e4,
  col_means = NULL,
  col_sds = NULL,
  col_names = NULL,
  batch_size = NULL,
  verbose = TRUE
)

probotPIT(
  input,
  model,
  mdn_components = NULL,
  params,
  n_test = NULL,
  n_samples = 1e4,
  col_means = NULL,
  col_sds = NULL,
  col_names = NULL,
  batch_size = NULL,
  verbose = TRUE
)

probotTARP(
  input,
  model,
  mdn_components = NULL,
  params,
  n_test = NULL,
  n_samples = 1e4,
  col_means = NULL,
  col_sds = NULL,
  col_names = NULL,
  batch_size = NULL,
  verbose = TRUE
)
}
\arguments{
  \item{input}{Numeric matrix of conditioning inputs, with one row per test
    observation and the same columns used to train \code{model}. Rows beyond
    \code{n_test} are ignored, so it is not necessary to subset beforehand.
    A bare vector is treated as a single observation.}

  \item{model}{An instantiated \code{\link[torch]{nn_module}} object: either an
    MDN (from \code{\link{probotMakeMDN}}) or a normalising flow (from
    \code{\link{probotMakeFlow}}, \code{\link{probotFlowRealNVP}},
    \code{\link{probotFlowMAF}} or \code{\link{probotFlowNSF}}). The model class
    decides which sampler is used.}

  \item{mdn_components}{Number of mixture components used when \code{model} was
    trained. Required for MDNs; ignored (and may be left \code{NULL}) for flows.}

  \item{params}{Numeric matrix of shape \eqn{n_\text{test} \times n_\text{params}}
    containing the true parameter values for each test observation. The number of
    columns must match the model's parameter dimension.}

  \item{n_test}{Number of values to compute (i.e. the first \code{n_test} rows
    of \code{params}). Defaults to \code{NULL} (all \code{nrow(params)} rows)
    for all three functions; otherwise capped at \code{nrow(params)}.}

  \item{n_samples}{Number of posterior samples to draw per observation.
    Defaults to \code{1e4}.}

  \item{col_means}{Optional centering values used for back-transforming model
    output, as returned by \code{\link{probotScaleForward}}. If omitted, results
    are computed on the model's (scaled) output scale; \code{params} must then
    be on that same scale.}

  \item{col_sds}{Optional scale values used for back-transforming model output,
    as returned by \code{\link{probotScaleForward}}. Must be given together with
    \code{col_means}.}

  \item{col_names}{Optional character vector of parameter names. For
    \code{probotCRPS} and \code{probotPIT}, assigned as column names
    of the returned matrix; for \code{probotTARP} it is used only for the
    internal sample array and the returned vector is unnamed.}

  \item{batch_size}{Number of observations sampled per chunk, passed to the
    underlying sampler. If \code{NULL} (default) the chunk size is
    chosen automatically to bound memory.}

  \item{verbose}{Logical. If \code{TRUE} (default), forwards progress reporting
    to the underlying sampler.}
}
\details{
  \strong{CRPS:} For each test observation \eqn{i} and parameter dimension
  \eqn{j}, CRPS is computed from \eqn{S} = \code{n_samples} posterior samples
  \eqn{z^{(1)}, \dots, z^{(S)}} and the true value \eqn{\theta_{ij}}:

  \deqn{\text{CRPS}_{ij} = \frac{1}{S} \sum_{s=1}^{S} |\theta_{ij} - z^{(s)}|
    - \frac{1}{2 S^2} \sum_{s,t=1}^{S} |z^{(s)} - z^{(t)}|}

  The second term is the spread penalty and is efficiently computed using
  sorted samples. Lower CRPS values indicate better-calibrated and sharper
  predictive distributions.

  \strong{PIT:} For each test observation \eqn{i} and parameter dimension
  \eqn{j}, the PIT value is the fraction of posterior samples whose
  \eqn{j}-th coordinate is less than or equal to the true value
  \eqn{\theta_{ij}}:

  \deqn{\text{PIT}_{ij} = \frac{1}{S} \sum_{s=1}^{S}
    \mathbf{1}\!\left[\tilde\theta_{ij}^{(s)} \le \theta_{ij}\right]}

  Under a well-calibrated posterior, PIT values for each parameter should be
  uniformly distributed on \eqn{[0, 1]}.

  \strong{TARP:} For each test observation \eqn{i}, a unit-length direction
  vector \eqn{\mathbf{d}_i \in \mathbb{R}^{n_\text{params}}} is drawn uniformly
  at random by normalising an isotropic Gaussian draw. The posterior samples and
  the true parameters are then projected onto this direction:

  \deqn{p^{(s)}_i = \tilde{\boldsymbol\theta}_i^{(s)\top}\mathbf{d}_i, \quad
    t_i = \boldsymbol\theta_i^\top\mathbf{d}_i}

  and the TARP value is

  \deqn{\text{TARP}_i = \frac{1}{S}\sum_{s=1}^{S}
    \mathbf{1}\!\left[p^{(s)}_i \le t_i\right].}

  Under a well-calibrated posterior, TARP values should be uniformly
  distributed on \eqn{[0, 1]}.

  All three functions sample the first \code{n_test} rows of \code{input} with a
  single batched call to the sampler that matches \code{model}:
  \code{\link{probotSamplePostMDN}} for MDNs,
  \code{\link{probotSamplePostNF}} for flows. Sampling proceeds in chunks of
  \code{batch_size} observations, so only one
  \eqn{n\_samples \times D \times \text{chunk}}{n_samples x D x chunk} array of
  draws is held in memory at a time, and the statistics above are computed
  vectorially over each chunk. Peak memory therefore tracks \code{batch_size}
  rather than \code{n_test}; the returned matrix tracks only
  \eqn{n_\text{test} \times D}. Lower \code{batch_size} if memory becomes restrictive,
  at the cost of more (smaller) sampling passes.
}
\value{
  \code{probotCRPS()} returns a numeric matrix of shape
  \eqn{n_\text{test} \times n_\text{params}} containing the CRPS value for
  every (observation, parameter) pair.

  \code{probotPIT()} returns a numeric matrix of shape
  \eqn{n_\text{test} \times n_\text{params}} containing the PIT value for
  every (observation, parameter) pair.

  \code{probotTARP()} returns a numeric vector of length
  \eqn{n_\text{test}} containing the TARP coverage value for each test
  observation.
}
\author{
  Aaron Robotham
}
\seealso{
  \code{\link{probotSamplePostMDN}}, \code{\link{probotSamplePostNF}},
  \code{\link{probotPredictMDN}}, \code{\link{probotScaleForward}}
}
\examples{
if (requireNamespace("torch", quietly = TRUE)) {
  set.seed(42)
  torch::torch_manual_seed(42)

  model <- probotMakeMDN(input_dim = 2, output_dim = 2, mdn_components = 3,
                         hidden_dims = c(16, 16))()
  x <- matrix(rnorm(20), ncol = 2)

  # True parameter values (10 observations x 2 parameters)
  params <- matrix(rnorm(20), nrow = 10, ncol = 2)

  ## CRPS
  crps <- probotCRPS(
    input = x,
    model = model,
    mdn_components = 3,
    params = params,
    n_samples = 100,
    col_means = c(0, 0),
    col_sds = c(1, 1),
    col_names = c("param1", "param2"),
    verbose = FALSE
  )
  dim(crps)

  ## PIT
  pit <- probotPIT(
    input = x,
    model = model,
    mdn_components = 3,
    params = params,
    n_samples = 100,
    col_means = c(0, 0),
    col_sds = c(1, 1),
    col_names = c("param1", "param2"),
    verbose = FALSE
  )
  dim(pit)

  ## TARP
  tarp <- probotTARP(
    input = x,
    model = model,
    mdn_components = 3,
    params = params,
    n_samples = 100,
    col_means = c(0, 0),
    col_sds = c(1, 1),
    verbose = FALSE
  )
  length(tarp)

  # The same three calls work unchanged on a normalising flow, with
  # mdn_components left out.
  flow <- probotMakeFlow(input_dim = 2, output_dim = 2, n_layers = 2,
                         hidden_dim = 16, device = "cpu")()

  pit_flow <- probotPIT(
    input = x,
    model = flow,
    params = params,
    n_samples = 100,
    col_means = c(0, 0),
    col_sds = c(1, 1),
    col_names = c("param1", "param2"),
    verbose = FALSE
  )
  dim(pit_flow)
}
}
