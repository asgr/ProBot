\name{probotLossEval}
\alias{probotLossEval}
\title{Score a Trained Model's Loss on Arbitrary Data}
\description{
Evaluate the loss of a trained model over any set of context/target rows
without computing gradients or modifying the model's parameters. The natural
use is scoring a held-out split alongside the training loss to check for
overfitting; it is the read-only counterpart to \code{\link{probotDataLoader}}
followed by one gradient-free pass.

The model type is inferred from the object, so the same call works for MDN,
Point and normalising flow models, and for all three flow architectures
including a residual location head.
}
\usage{
probotLossEval(input, output, model, mdn_components = NULL, idx = NULL,
               batch = 4096L, loss_fn = NULL, device = NULL,
               verbose = FALSE, per_row = FALSE)
}
\arguments{
  \item{input}{Matrix, data frame or torch tensor of conditioning context with
  \code{input_dim} columns. Row order must match \code{output}.}

  \item{output}{Matrix, data frame or torch tensor of target parameters with
  \code{output_dim} columns. These are the \emph{targets} (theta), not a
  model's raw output; the network is run internally.}

  \item{model}{A trained \code{nn_module} from \code{\link{probotMakeMDN}},
  \code{\link{probotMakePoint}}, \code{\link{probotMakeFlow}} or
  \code{\link{probotLoad}}.}

  \item{mdn_components}{Number of Gaussian mixture components. Required for an
  MDN whose constructor did not store it, and ignored otherwise.}

  \item{idx}{Optional integer vector of rows of \code{input} and \code{output}
  to score. Equivalent to subsetting both matrices beforehand, and the more
  memory-efficient route for a large matrix that is already in RAM.}

  \item{batch}{Rows per forward pass. The reported loss is weighted by rows, so
  it does not depend on this value beyond float32 rounding.}

  \item{loss_fn}{Scoring function. Defaults to \code{\link{probotLossMDN}} for
  an MDN, \code{\link{torch::nnf_mse_loss}} for a Point model and
  \code{\link{probotLossNF}} for a flow, matching the trainers' defaults. Any
  function with the same signature as one of those may be supplied.}

  \item{device}{Where to place the tensors. Defaults to the device the model
  already sits on, so the two always agree.}

  \item{verbose}{Print a one-line summary of the score.}

  \item{per_row}{Also compute the loss of every individual row and return a
  robust summary of its distribution. See \strong{Robust scoring} below. Only
  supported for a model type's default loss; a custom \code{loss_fn} warns and
  leaves the result as the five-field list.}
}
\details{
The argument order follows \code{\link{probotDataLoader}} and
\code{\link{probotPredictMDN}} -- context first, targets second -- which is the
reverse of the \code{probotLoss*()} functions, whose first argument is
\code{output_true}. Passing the two the wrong way round still type-checks as a
pair of matrices, so the stored \code{input_dim} and \code{output_dim} are
checked against the column counts and an error names the swap.

A bare vector is read as one column of values, not as a single observation; pass
\code{matrix(x, nrow = 1)} for one row of context.

For a Point model the underlying \code{torch::nnf_mse_loss()} averages over
\emph{elements}, not rows, and requires the model's output layer to be as wide
as the targets; it fails outright if they differ. Each type's default therefore
reproduces exactly what the matching trainer reports as its epoch loss, which is
what makes a held-out score comparable with a training score.

No parameters are updated and gradients are not tracked. The model is placed in
eval mode for the pass and returned to whichever mode it was in afterwards, so
the call is safe between training epochs.

Like the trainers' reported loss, the value is computed on whatever scale the
targets are supplied on. Standardise them with \code{\link{probotScaleForward}}
first if that is how the model was trained, and compare a held-out score only
against a training score from the same scale.

Negative values are expected for a continuous density: with many dimensions the
log-Jacobian term of a flow, or the log-normalisation of an MDN, can outweigh
the penalty, so the absolute level carries no meaning. Only the gap between two
scores on the same data does.

\strong{Robust scoring.} \code{loss} is a \emph{mean} over rows, and a
negative log-likelihood has no upper bound, so a single unusual row can move it
by orders of magnitude without saying anything about the model. For a flow,
\code{loss = -log p(theta|x)} decomposes as
\code{0.5 * ||z||^2 + (D/2) * log(2*pi) - log|det J|}, and the first term grows
quadratically in the latent norm: one observation whose context sits far enough
off the training support to reach \code{||z|| ~ 6000} contributes
\code{1.8e7 / n} to the batch mean. A mean is the wrong summary for that
distribution, which is why \code{per_row = TRUE} reports the median, spread and
maximum alongside it, plus a count of rows beyond 10 MADs of the median. A mean
far above the median with \code{n_extreme > 0} indicates tail rows, not a model
regression.

The summary needs a second forward pass for a flow, because
\code{\link{probotLossNF}} returns only the reduced mean, so the feature is
opt-in.
}
\value{
A list with components
  \item{loss}{Mean per-row loss over the scored rows.}
  \item{n}{Number of rows scored.}
  \item{batches}{Number of forward passes used.}
  \item{model_type}{One of \code{"mdn"}, \code{"point"} or \code{"flow"}.}
  \item{loss_name}{Name of the loss function that produced \code{loss}, or
  \code{"<custom>"} for an unrecognised one.}

When \code{per_row = TRUE} and the loss is a model type's default, the list
additionally carries
  \item{loss_median}{Median per-row loss -- the robust centre.}
  \item{loss_p90}{90th percentile of the per-row losses.}
  \item{loss_max}{Worst single row.}
  \item{loss_mad}{Median absolute deviation of the per-row losses, rescaled to
  an sd under normality. \code{NA} when the median is not finite or the spread
  is zero.}
  \item{extreme_threshold}{\code{loss_median + 10 * loss_mad}; \code{Inf} when
  \code{loss_mad} is not usable.}
  \item{n_extreme}{Rows above \code{extreme_threshold}. A small count with a
  large gap between \code{loss} and \code{loss_median} means the mean is being
  driven by those rows.}
  \item{row_loss}{Numeric vector of the loss of each scored row, in the order
  they were scored (so indexed by \code{idx} when supplied).}
}
\references{
See \code{\link{probotLoss}} for the underlying batch loss functions, and
\code{\link{probotPIT}}, \code{\link{probotCRPS}} and \code{\link{probotTARP}}
for diagnostics that assess calibration rather than fit.
}
\author{
Aaron Robotham
}
\seealso{
\code{\link{probotLoss}}, \code{\link{probotDataLoader}},
\code{\link{probotTrainFlow}}, \code{\link{probotLoad}}
}
\examples{
if (requireNamespace("torch", quietly = TRUE)) {
  set.seed(42)

  # A small regression problem: context x, targets theta.
  x <- matrix(rnorm(400 * 3), 400, 3)
  theta <- matrix(rnorm(400 * 2), 400, 2)

  point <- probotMakePoint(input_dim = 3, output_dim = 2,
                           hidden_dims = c(32, 32))()
  flow <- probotMakeFlow(input_dim = 3, output_dim = 2, n_layers = 2,
                         hidden_dim = 16)()

  # Score everything, then score a held-out subset.
  probotLossEval(x, theta, point)
  probotLossEval(x, theta, flow, idx = 301:400)

  # The same value either way round: idx is just subsetting.
  identical(
    probotLossEval(x, theta, point, idx = 1:50)$loss,
    probotLossEval(x[1:50, ], theta[1:50, ], point)$loss
  )

  # A mean NLL can be dominated by one bad row. per_row = TRUE exposes that.
  full <- probotLossEval(x, theta, flow, per_row = TRUE)
  c(mean = full$loss, median = full$loss_median, max = full$loss_max,
    n_extreme = full$n_extreme)

  # Rows whose loss sits 10 MADs above the median, i.e. the tail rows that
  # would drag the mean without changing the median at all.
  which(full$row_loss > full$extreme_threshold)
}
}
