\name{probotAssess}
\alias{probotCRPS}
\alias{probotPIT}
\alias{probotTARP}
\title{Posterior Predictive Assessment: CRPS, PIT, and TARP}
\description{
Assess the calibration and sharpness of posterior samples drawn from a
Mixture Density Network (MDN). \code{probotCRPS()} computes the Continuous
Ranked Probability Score, \code{probotPIT()} computes Probability Integral
Transform values, and \code{probotTARP()} computes Test of Accuracy with
Random Points coverage values.
}
\usage{
probotCRPS(
  mdn_output,
  params,
  n_test = NULL,
  n_samples = 1e4,
  col_means,
  col_sds,
  col_names = NULL,
  verbose = TRUE
)

probotPIT(
  mdn_output,
  params,
  n_test = NULL,
  n_samples = 1e4,
  col_means,
  col_sds,
  col_names = NULL,
  verbose = TRUE
)

probotTARP(
  mdn_output,
  params,
  n_test = NULL,
  n_samples = 1e4,
  col_means,
  col_sds,
  col_names = NULL,
  verbose = TRUE
)
}
\arguments{
  \item{mdn_output}{List returned by \code{\link{probotPredictMDN}()}, containing
    \code{logits}, \code{mu}, and \code{log10_sigma} tensors.}

  \item{params}{Numeric matrix of shape \eqn{n_\text{test} \times n_\text{params}}
    containing the true parameter values for each test observation.}

  \item{n_test}{Number of values to compute. For \code{probotCRPS()} and
    \code{probotPIT()}, defaults to \code{NULL} (all rows). For
    \code{probotTARP()}, defaults to \code{1e4}. In all cases, cannot exceed
    \code{nrow(params)}.}

  \item{n_samples}{Number of posterior samples to draw per observation.
    Defaults to \code{1e4}.}

  \item{col_means}{Centering values used for back-transforming MDN output,
    as returned by \code{\link{probotScaleForward}}.}

  \item{col_sds}{Scale values used for back-transforming MDN output,
    as returned by \code{\link{probotScaleForward}}.}

  \item{col_names}{Optional character vector of parameter names. For
    \code{probotCRPS} and \code{probotPIT}, assigned as column names
    of the returned matrix. For \code{probotTARP()}, passed through to
    \code{\link{probotSamplePostMDN}().}}

  \item{verbose}{Logical. If \code{TRUE} (default), prints progress to the
    console every 1000 observations.}
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
  \code{\link{probotSamplePostMDN}}, \code{\link{probotPredictMDN}},
  \code{\link{probotScaleForward}}
}
\examples{
if (requireNamespace("torch", quietly = TRUE)) {
  set.seed(42)
  torch::torch_manual_seed(42)

  model <- probotMakeMDN(input_dim = 2, output_dim = 2, mdn_components = 3)()
  x <- matrix(rnorm(20), ncol = 2)
  pred <- probotPredictMDN(x, model, mdn_components = 3)

  # True parameter values (10 observations x 2 parameters)
  params <- matrix(rnorm(20), nrow = 10, ncol = 2)

  ## CRPS
  crps <- probotCRPS(
    mdn_output = pred,
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
    mdn_output = pred,
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
    mdn_output = pred,
    params = params,
    n_samples = 100,
    col_means = c(0, 0),
    col_sds = c(1, 1),
    verbose = FALSE
  )
  length(tarp)
}
}
