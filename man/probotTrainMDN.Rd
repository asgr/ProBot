\name{probotTrainMDN}
\alias{probotSingleEpochMDN}
\alias{probotTrainMDN}
\title{Train Mixture Density Networks for One Epoch or Many}
\description{
Run one optimization epoch for an MDN or iterate multiple epochs with optional
early stopping and checkpoint saving.
}
\usage{
probotSingleEpochMDN(model, dataloader, optimizer, mdn_components,
  loss_fn = probotLossMDN, lambda = 0)

probotTrainMDN(model, dataloader, optimizer, epochs = 100, mdn_components,
  loss_fn = probotLossMDN, lambda = 0, checkpoint_dir = NULL,
  checkpoint_every = 10, history = NULL, verbose = TRUE, early_stop = TRUE,
  stop_window = 20, stop_delta = 1e-2)
}
\arguments{
  \item{model}{Instantiated torch module returned by \code{probotMakeMDN(... )()}.}

  \item{dataloader}{Torch data loader yielding \code{$x} and \code{$y} batches.}

  \item{optimizer}{Torch optimizer configured with model parameters.}

  \item{mdn_components}{Number of Gaussian mixture components in the model.}

  \item{loss_fn}{Loss function to use for optimisation.}
  
  \item{lambda}{Optional scalar mixing value to combine loss with MSE. Setting this above 0 can help model collapse.}

  \item{epochs}{Maximum number of epochs for \code{probotTrainMDN()}.}

  \item{checkpoint_dir}{Optional directory to save periodic \code{.pt} checkpoints.}

  \item{checkpoint_every}{Save checkpoint every this many epochs.}

  \item{history}{Optional existing history data frame to append to.}

  \item{verbose}{Logical; print per-epoch metrics when \code{TRUE}.}

  \item{early_stop}{Logical; enable rolling-window early stopping.}

  \item{stop_window}{Window size (in epochs) for early stopping comparison.}

  \item{stop_delta}{Minimum required improvement in rolling mean loss.}
}
\details{
Training metrics include loss, MAE, RMSE, average sigma, and average mixture
weights. Early stopping compares mean loss over the latest
\code{stop_window} epochs to the previous window.
}
\value{
\itemize{
  \item \code{probotSingleEpochMDN()}: List with \code{loss}, \code{mae},
  \code{rmse}, \code{sigma}, and \code{mix}.
  \item \code{probotTrainMDN()}: List with trained \code{model} and
  \code{history} data frame. The history data frame contains columns
  \code{epoch}, \code{loss}, \code{mae}, \code{rmse}, \code{sigma}, and
  \code{mix_sd} (standard deviation of mixture weights across components).
}
}
\author{
Aaron Robotham
}
\seealso{
\code{\link{probotDataLoader}}, \code{\link{probotLossMDN}},
\code{\link{probotSave}}
}
\examples{
if (requireNamespace("torch", quietly = TRUE)) {
  torch::torch_manual_seed(1)

  x <- matrix(rnorm(60), ncol = 3)
  y <- matrix(rnorm(40), ncol = 2)

  loader <- probotDataLoader(x, y, batch = 5, shuffle = FALSE)
  model <- probotMakeMDN(input_dim = 3, output_dim = 2, mdn_components = 2)()
  opt <- torch::optim_adam(model$parameters, lr = 0.01)

  metrics <- probotSingleEpochMDN(
    model = model,
    dataloader = loader,
    optimizer = opt,
    mdn_components = 2
  )

  metrics$loss
}
}
