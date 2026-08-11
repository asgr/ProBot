\name{probotSingleEpochFlow}
\alias{probotSingleEpochFlow}
\alias{probotMultiEpochFlow}
\title{Train Normalizing Flow Networks for One Epoch or Many}
\description{
Run one optimization epoch for a normalizing flow network or iterate multiple
epochs with optional early stopping and checkpoint saving.
}
\usage{
probotSingleEpochFlow(model, dataloader, optimizer,
  loss_fn = probotLossNF)

probotMultiEpochFlow(model, dataloader, optimizer, epochs = 100,
  loss_fn = probotLossNF, checkpoint_dir = NULL, checkpoint_every = 10,
  history = NULL, verbose = TRUE, early_stop = TRUE,
  stop_window = 20, stop_delta = 1e-4)
}
\arguments{
  \item{model}{Instantiated torch module returned by \code{probotNetworkFlow(...)}.}

  \item{dataloader}{Torch data loader created with \code{probotDataLoader()} where
    \code{batch[[1]]} contains the parameter samples (\code{true_theta}) and
    \code{batch[[2]]} contains the conditioning context (\code{context_x}).}

  \item{optimizer}{Torch optimizer configured with model parameters.}

  \item{loss_fn}{Loss function to use for optimisation. Defaults to
    \code{probotLossNF}.}

  \item{epochs}{Maximum number of epochs for \code{probotMultiEpochFlow()}.}

  \item{checkpoint_dir}{Optional directory to save periodic \code{.pt} checkpoints.}

  \item{checkpoint_every}{Save a checkpoint and print progress every this many epochs.}

  \item{history}{Optional existing history data frame to append to.}

  \item{verbose}{Logical; print per-epoch loss when \code{TRUE}.}

  \item{early_stop}{Logical; enable rolling-window early stopping.}

  \item{stop_window}{Window size (in epochs) for early stopping comparison.}

  \item{stop_delta}{Minimum required improvement in rolling mean loss to continue
    training.}
}
\details{
The dataloader must be constructed so that the first element of each batch is the
parameter tensor (\code{true_theta}) and the second element is the context tensor
(\code{context_x}). When using \code{probotDataLoader()}, pass the parameter
matrix as \code{input} and the observation matrix as \code{output}.

Early stopping compares the mean negative log-likelihood loss over the most recent
\code{stop_window} epochs to the previous window of the same length, and halts
training if the improvement falls below \code{stop_delta}.
}
\value{
\itemize{
  \item \code{probotSingleEpochFlow()}: List with a single element \code{loss}
    (mean negative log-likelihood for the epoch).
  \item \code{probotMultiEpochFlow()}: List with trained \code{model} and
    \code{history} data frame containing columns \code{epoch} and \code{loss}.
}
}
\author{
Aaron Robotham
}
\seealso{
\code{\link{probotNetworkFlow}}, \code{\link{probotLossNF}},
\code{\link{probotSamplePostNF}}, \code{\link{probotDataLoader}}
}
\examples{
if (requireNamespace("torch", quietly = TRUE)) {
  torch::torch_manual_seed(1)

  theta <- matrix(rnorm(200), ncol = 2)   # 100 samples, 2 parameters
  x_obs <- matrix(rnorm(300), ncol = 3)   # 100 samples, 3 context features

  loader <- probotDataLoader(theta, x_obs, batch = 20, shuffle = FALSE)
  flow   <- probotNetworkFlow(dim_theta = 2, dim_x = 3)
  opt    <- torch::optim_adam(flow$parameters, lr = 1e-3)

  metrics <- probotSingleEpochFlow(
    model      = flow,
    dataloader = loader,
    optimizer  = opt
  )

  metrics$loss
}
}
