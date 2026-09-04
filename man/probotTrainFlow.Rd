\name{probotTrainFlow}
\alias{probotSingleEpochFlow}
\alias{probotTrainFlow}
\title{Train Normalizing Flow Models for One Epoch or Many}
\description{
Run one optimisation epoch for a flow model or iterate multiple epochs with
optional early stopping and checkpoint saving.
}
\usage{
probotSingleEpochFlow(model, dataloader, optimizer, loss_fn = probotLossNF,
  lambda = 0, point = "centre", n_point_samples = 32)

probotTrainFlow(model, dataloader, optimizer, epochs = 100,
  loss_fn = probotLossNF, lambda = 0, point = "centre",
  n_point_samples = 32, checkpoint_dir = NULL, checkpoint_every = 10,
  history = NULL, verbose = TRUE, early_stop = TRUE, stop_window = 20,
  stop_delta = 1e-2)
}
\arguments{
  \item{model}{Instantiated torch module returned by \code{probotMakeFlow(... )()}.}

  \item{dataloader}{Torch data loader yielding context inputs and parameter targets.}

  \item{optimizer}{Torch optimizer configured with model parameters.}

  \item{loss_fn}{Loss function to use for optimisation.}

  \item{lambda}{Scalar mixing weight in \code{[0, 1]} blending the likelihood
  loss with a mean-squared-error term, as
  \code{(1 - lambda) * loss_fn + lambda * mse}. The default of \code{0} gives
  pure maximum-likelihood training and reproduces earlier behaviour exactly.
  See \bold{Loss blending} below.}

  \item{point}{Which point estimate the MSE term regresses against, when
  \code{lambda > 0}: \code{"centre"} (the default) uses the deterministic
  inverse of the base distribution's mode, \code{"mean"} uses the average of
  \code{n_point_samples} posterior draws, and \code{"loc"} uses the location
  head of a \code{\link{probotFlowLoc}} model (built with
  \code{loc_head = TRUE} in \code{\link{probotMakeFlow}}). Ignored when
  \code{lambda = 0}. See \bold{Point estimators} below.}

  \item{n_point_samples}{Number of inverse draws averaged when
  \code{point = "mean"}. Must be a whole number of at least 1, and is only
  validated when \code{lambda > 0}.}

  \item{epochs}{Maximum number of epochs for \code{probotTrainFlow()}.}

  \item{checkpoint_dir}{Optional directory to save periodic \code{.pt} checkpoints.}

  \item{checkpoint_every}{Save checkpoint every this many epochs.}

  \item{history}{Optional existing history data frame to append to.}

  \item{verbose}{Logical; print per-epoch loss when \code{TRUE}.}

  \item{early_stop}{Logical; enable rolling-window early stopping.}

  \item{stop_window}{Window size (in epochs) for early stopping comparison.}

  \item{stop_delta}{Minimum required improvement in rolling mean loss.}
}
\details{
The data loader should provide observed summaries or features as inputs and the
corresponding parameters as targets. Early stopping compares mean loss over the
latest \code{stop_window} epochs to the previous window.

Gradients are clipped to \code{[-1, 1]} in value (via
\code{\link[torch]{nn_utils_clip_grad_value_}}) before each optimizer step.
Flow training is sensitive to large updates because the negative-log-likelihood
loss can spike; clipping mitigates this at the cost of slower, more stable
progression.

\subsection{Loss blending}{

Setting \code{lambda} above 0 adds a mean-squared-error contribution to the
cost, which encourages better point estimates from the trained model. This
mirrors the \code{lambda} argument of \code{\link{probotTrainMDN}}, with one
important difference in how the point estimate is obtained.

An MDN exposes an analytic posterior mean (the weighted mean of its mixture
components) for free. A flow's forward pass maps parameters to latent space and
yields no network output to regress against, so the MSE term must instead be
built from the inverse map. The two terms are combined as tensors \emph{before}
\code{backward()}, so both contribute gradients in a single pass.

\code{mae} and \code{rmse} are reported only when the MSE term is active, so a
\code{lambda = 0} run keeps the historical \code{epoch}/\code{loss} history
schema and is numerically identical to a run without the argument at all.
}

\subsection{Point estimators}{

The \code{point} argument selects which quantity the MSE term targets. They
answer different questions, and the right choice depends on what you intend to
do with the fitted model.

\itemize{
  \item \code{"centre"} (default) evaluates \code{model$inverse(z = 0, x)}, the
  image of the base distribution's mode. It is deterministic, and for a
  symmetric conditional posterior coincides with the posterior mean. For a
  skewed posterior it sits nearer the mode and is therefore biased under
  squared-error loss; strictly it is a Jacobian-weighted centre rather than
  either the mean or the marginal mode.
  \item \code{"mean"} averages \code{n_point_samples} draws through the
  inverse, giving an unbiased estimate of \code{E[theta | x]} -- the
  Bayes-optimal point under squared error, and the closest flow analogue to an
  MDN's mixture mean. It is the better choice when a skewed posterior makes the
  centre a poor summary. The cost is Monte-Carlo noise: both the estimate and
  its gradient fluctuate from batch to batch, and the whole blend becomes
  stochastic.
  \item \code{"loc"} returns \code{model$location(x)}, the location head of a
  \code{\link{probotFlowLoc}} model. It requires \code{loc_head = TRUE} and
  errors otherwise. This is the cheapest and least noisy of the three: no
  inverse sweep, no Monte-Carlo noise, and the gradient reaches the head
  directly instead of through the shared conditioner. Whether \code{mu(x)} is a
  good summary of the posterior is then the flow's responsibility -- the NLL
  term is what makes the residual carry the mass, so \code{"loc"} is meant to be
  used with a \code{lambda} small enough that the likelihood still dominates.
}

\code{point = "mean"} draws are batched into a single inverse call with each
context row repeated \code{n_point_samples} times, so the cost is roughly
\code{n_point_samples} inverse sweeps amortised over the batch rather than one
per observation. Measured on a 256-row CPU batch, \code{n_point_samples = 32}
added about half again to the per-step cost for the \code{realnvp} and
\code{nsf} coupling styles and about twofold for \code{maf}, whose inverse runs
\code{output_dim} sequential masked-MLP evaluations. Raising it to 128 moved
those to roughly 2.5x and 3x.

The default of 32 is a knee rather than a optimum: with common random numbers
the deviation of a length-\code{m} partial mean from the full-pool mean scales
as \code{sqrt(1/m - 1/M)}, so gains fall off quickly past 32 while cost keeps
climbing. Raise it when the posterior is heavily skewed or the fit is
unstable; lower it if \code{maf} sampling cost dominates the epoch.

In an illustrative lognormal example (posterior mean well above the mode, 40
epochs, held-out rows) both blends improved on pure likelihood training on
both summaries, but each favoured the estimator it named: \code{point =
"mean"} gave the best posterior-mean accuracy (RMSE 1.98 against 2.42 for
\code{point = "centre"}, from 3.11 with no blend), while \code{point =
"centre"} gave the best centre accuracy (2.32 against 2.92 for \code{point =
"mean"}). Pick \code{point} to match the summary you will actually report, and
expect very large \code{lambda} to bias the posterior shape away from maximum
likelihood.
}
}
\value{
\itemize{
  \item \code{probotSingleEpochFlow()}: List with \code{loss} and, when
  \code{lambda > 0}, \code{mae} and \code{rmse} measured against the estimator
  named by \code{point} (element-wise, over all target elements).
  \item \code{probotTrainFlow()}: List with trained \code{model} and
  \code{history} data frame. The history data frame contains columns
  \code{epoch} and \code{loss}, plus \code{mae} and \code{rmse} when
  \code{lambda > 0}.
}
}
\author{
Aaron Robotham
}
\seealso{
\code{\link{probotDataLoader}}, \code{\link{probotLossNF}},
\code{\link{probotMakeFlow}}, \code{\link{probotSamplePostNF}},
\code{\link{probotTrainMDN}}
}
\examples{
if (requireNamespace("torch", quietly = TRUE)) {
  torch::torch_manual_seed(1)

  x <- matrix(rnorm(60), ncol = 3)
  y <- matrix(rnorm(40), ncol = 2)

  loader <- probotDataLoader(x, y, batch = 5, shuffle = FALSE)
  model <- probotMakeFlow(input_dim = 3, output_dim = 2)()
  opt <- torch::optim_adam(model$parameters, lr = 0.01)

  metrics <- probotSingleEpochFlow(
    model = model,
    dataloader = loader,
    optimizer = opt
  )

  metrics$loss

  # Blend the likelihood loss with a centre-MSE term
  blended <- probotSingleEpochFlow(
    model = model,
    dataloader = loader,
    optimizer = opt,
    lambda = 0.5
  )

  blended[c("loss", "mae", "rmse")]

  # Or target the posterior mean instead of the centre, at the cost of
  # averaging several inverse draws per observation each step. Fresh model, so
  # the two runs start from the same place.
  torch::torch_manual_seed(1)
  model2 <- probotMakeFlow(input_dim = 3, output_dim = 2)()
  means <- probotSingleEpochFlow(
    model = model2,
    dataloader = loader,
    optimizer = torch::optim_adam(model2$parameters, lr = 0.01),
    lambda = 0.5,
    point = "mean",
    n_point_samples = 32
  )

  means[c("loss", "mae", "rmse")]
}
}
