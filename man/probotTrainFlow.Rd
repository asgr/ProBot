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
yields no network output to regress against, so the MSE term instead compares
the true parameters against the \emph{deterministic centre} of the learned
conditional posterior, \code{model$inverse(z = 0, x)}. For the standard normal
base distribution used here, \code{z = 0} is the base distribution's mode, so
the estimate is a Jacobian-weighted centre of the posterior rather than its
mean. Under the blend the two terms are combined as tensors \emph{before}
\code{backward()}, so both contribute gradients in a single pass.

Computing the centre costs one inverse sweep per batch. This is cheap for the
\code{realnvp} and \code{nsf} styles, whose inverse is a single vectorised
coupling sweep, but noticeably more expensive for \code{maf}, whose inverse
runs \code{output_dim} sequential masked-MLP evaluations.

The trade-off is mostly relevant when training is budget-limited. On a well
converged fit the likelihood term alone already places the centre accurately,
and \code{lambda} changes little; the benefit shows up early in training and
where the likelihood surface is difficult. Very large \code{lambda} can bias
the posterior shape away from maximum likelihood, since the centre-MSE term
does not target the posterior mean.

When \code{lambda} is 0 the loss and metrics are unchanged from a run without
the argument at all; \code{mae} and \code{rmse} are reported only when the MSE
term is active, so the history data frame keeps its \code{epoch}/\code{loss}
schema by default.
}
}
\value{
\itemize{
  \item \code{probotSingleEpochFlow()}: List with \code{loss}, and, when
  \code{lambda > 0}, \code{mae} and \code{rmse} for the centre estimate
  (element-wise, over all target elements).
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
}
}
