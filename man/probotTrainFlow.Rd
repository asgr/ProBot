\name{probotTrainFlow}
\alias{probotSingleEpochFlow}
\alias{probotTrainFlow}
\title{Train Normalizing Flow Models for One Epoch or Many}
\description{
Run one optimisation epoch for a flow model or iterate multiple epochs with
optional early stopping and checkpoint saving.
}
\usage{
probotSingleEpochFlow(model, dataloader, optimizer, loss_fn = probotLossNF)

probotTrainFlow(model, dataloader, optimizer, epochs = 100,
  loss_fn = probotLossNF, checkpoint_dir = NULL, checkpoint_every = 10,
  history = NULL, verbose = TRUE, early_stop = TRUE, stop_window = 20,
  stop_delta = 1e-2)
}
\arguments{
  \item{model}{Instantiated torch module returned by \code{probotMakeFlow(... )()}.}

  \item{dataloader}{Torch data loader yielding context inputs and parameter targets.}

  \item{optimizer}{Torch optimizer configured with model parameters.}

  \item{loss_fn}{Loss function to use for optimisation.}

  \item{epochs}{Maximum number of epochs for \code{probotMultiEpochFlow()}.}

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
}
\value{
\itemize{
  \item \code{probotSingleEpochFlow()}: List with \code{loss}.
  \item \code{probotMultiEpochFlow()}: List with trained \code{model} and
  \code{history} data frame. The history data frame contains columns
  \code{epoch} and \code{loss}.
}
}
\author{
Aaron Robotham
}
\seealso{
\code{\link{probotDataLoader}}, \code{\link{probotLossNF}},
\code{\link{probotMakeFlow}}, \code{\link{probotSamplePostNF}}
}
\examples{
if (requireNamespace("torch", quietly = TRUE)) {
  torch::torch_manual_seed(1)

  x <- matrix(rnorm(60), ncol = 3)
  y <- matrix(rnorm(40), ncol = 2)

  loader <- probotDataLoader(x, y, batch = 5, shuffle = FALSE)
  model <- probotMakeFlow(dim_theta = 2, dim_x = 3)()
  opt <- torch::optim_adam(model$parameters, lr = 0.01)

  metrics <- probotSingleEpochFlow(
    model = model,
    dataloader = loader,
    optimizer = opt
  )

  metrics$loss
}
}
