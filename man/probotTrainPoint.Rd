\name{probotTrainPoint}
\alias{probotSingleEpochPoint}
\alias{probotTrainPoint}
\title{Train Point Estimate Model}
\description{
Train a point estimate MLP using mean squared error loss with optional early stopping and checkpointing.
}
\usage{
probotSingleEpochPoint(model, dataloader, optimizer, loss_fn = nnf_mse_loss)

probotTrainPoint(
  model,
  dataloader,
  optimizer,
  epochs = 100,
  loss_fn = nnf_mse_loss,
  checkpoint_dir = NULL,
  checkpoint_every = 10,
  history = NULL,
  verbose = TRUE,
  early_stop = TRUE,
  stop_window = 20,
  stop_delta = 1e-2
)
}
\arguments{
  \item{model}{An instantiated point estimate \code{\link[torch]{nn_module}} object.}
  
  \item{dataloader}{A \code{\link[torch]{dataloader}} providing training data.}
  
  \item{optimizer}{A torch optimizer (e.g., \code{optim_adam}).}
  
  \item{epochs}{Maximum number of training epochs.}
  
  \item{loss_fn}{Loss function. Defaults to MSE (\code{nnf_mse_loss}).}
  
  \item{checkpoint_dir}{Directory to save model checkpoints.}
  
  \item{checkpoint_every}{Save checkpoint every n epochs.}
  
  \item{history}{Optional data.frame of previous training history to prepend.}
  
  \item{verbose}{If TRUE, print progress.}
  
  \item{early_stop}{If TRUE, stop when loss improvement stalls.}
  
  \item{stop_window}{Number of epochs to average for early stopping.}
  
  \item{stop_delta}{Minimum improvement threshold.}
}
\value{
A list with \code{model} and \code{history} data.frame.
}
\author{
Aaron Robotham
}
\seealso{
\code{\link{probotMakePoint}}, \code{\link{probotPredictPoint}}, \code{\link{probotSingleEpochPoint}}
}
