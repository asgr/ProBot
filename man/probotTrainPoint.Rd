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
  stop_delta = 1e-2,
  holdout_fraction = 0.1,
  val_dataloader = NULL,
  val_batch = NULL,
  split_seed = NULL,
  holdout_stop = TRUE,
  holdout_min_delta = 1e-3,
  holdout_patience = 5L
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

  \item{holdout_fraction}{Fraction of the dataloader's rows held out for
    verification, scored once per epoch and reported beside the training loss.
    Defaults to 0.1. Set to 0 to train on every row and record no verification
    loss. Ignored when \code{val_dataloader} is supplied. Datasets with fewer
    than 10 rows warn and skip the split rather than trending a verification
    set too small to mean anything.}

  \item{val_dataloader}{Optional \code{\link{probotDataLoader}} used as the
    verification set verbatim, in place of a random split. Rows may overlap the
    training data; nothing is checked, since an explicit loader is a deliberate
    choice.}

  \item{val_batch}{Rows per verification batch. Defaults to the whole
    verification set in one batch. Affects speed and memory only: the
    verification loss is averaged row-weighted, so it is identical for any
    \code{val_batch}.}

  \item{split_seed}{Seed for the holdout draw. \code{NULL} (default) uses the
    current state of R's global RNG. A given value reproduces the same split
    without disturbing the caller's RNG stream.}

  \item{holdout_stop}{If TRUE (default) and a verification set exists, stop when
    the training loss is still improving while the verification loss is not.
    This is the over-training signal; see Details.}

  \item{holdout_min_delta}{A verification-loss trend more positive than
    \code{-holdout_min_delta} (per epoch) counts as "not improving".}

  \item{holdout_patience}{Consecutive epochs the over-training condition must
    hold before stopping, so that one noisy epoch does not end the run.}
}
\details{
  \subsection{Verification split}{
    By default 10 percent of the rows in \code{dataloader} are set aside before
    epoch 1, and the model trains on the remaining 90 percent. The verification
    rows are scored once per epoch in \code{eval()} mode with no gradient step,
    which adds one forward pass over roughly a tenth of the data to each epoch.
    \code{history} gains a \code{val_loss} column (and \code{val_mae},
    \code{val_rmse} when the model type reports an error metric), and
    \code{verbose} output names both numbers on every printed line so the two
    cannot be confused.

    The verification set is never reshuffled or truncated between epochs, so
    the epoch-to-epoch change in \code{val_loss} is a real trend rather than
    resampling noise. \code{probotLossEval()} scores the same rows afterwards
    if you want a per-row breakdown.
  }

  \subsection{Stopping rules}{
    Two independent rules can end a run early, both gated by \code{early_stop}.
    The older one, driven by \code{stop_window} and \code{stop_delta}, stops
    when the \emph{training} loss has plateaued. The newer one, driven by
    \code{holdout_stop}, \code{holdout_min_delta} and \code{holdout_patience},
    stops on the asymmetric signature of over-training: the training loss is
    still falling at a measurable rate while the verification loss has flattened
    or turned upward. The asymmetry matters. A training loss that has itself
    gone flat is not over-fitting but convergence, and the plateau rule is the
    one that fires there, so \code{holdout_stop} deliberately stays quiet and
    reports which rule fired in \code{validation$stop_reason}.

    Both rules read least-squares slopes over the last \code{stop_window}
    epochs, so a trend of exactly \code{delta} per epoch is detected regardless
    of window length.
  }

  \subsection{What the verification loss is measured on}{
    It is the \emph{same functional} as the training loss, including any
    \code{lambda} blend the trainer applies. Comparing a pure likelihood
    verification number against a blended training number would make the
    stopping rule compare incomparable trends.
  }

  \subsection{A note on reproducibility}{
    Splitting re-indexes the training rows into a compacted tensor, so a run
    with a holdout is not numerically identical to the same run with
    \code{holdout_fraction = 0} at the same seed. It also consumes one draw from
    the global RNG unless \code{split_seed} is given.

    A \code{history} recorded before a split existed (or with
    \code{holdout_fraction = 0}) can still be continued by a run that splits:
    the columns are unioned and the earlier epochs carry \code{NA} in
    \code{val_loss}, since those epochs genuinely have no verification number.
  }
}
\value{
A list with \code{model} and \code{history} data.frame. When a verification set
exists, a \code{validation} component is added with \code{n_train},
\code{n_verify}, \code{verify_idx} (\code{NULL} for an explicit
\code{val_dataloader}), \code{source}, \code{holdout_fraction},
\code{best_epoch}, \code{best_val_loss}, \code{final_val_loss} and
\code{stop_reason}.

\code{best_epoch} names the epoch with the lowest verification loss. The
returned \code{model} is the one trained at the \emph{stopping or final} epoch,
not the best one; re-run with a shorter \code{epochs} or restore a checkpoint to
recover it.
}
\author{
Aaron Robotham
}
\seealso{
\code{\link{probotMakePoint}}, \code{\link{probotPredictPoint}},
\code{\link{probotSingleEpochPoint}}, \code{\link{probotLossEval}},
\code{\link{probotDataLoader}}
}
\examples{
if (requireNamespace("torch", quietly = TRUE)) {
  torch::torch_manual_seed(1)
  set.seed(1)

  x <- matrix(rnorm(200 * 3), ncol = 3)
  y <- matrix(rnorm(200 * 2), ncol = 2)

  # 90/10 split by default; history carries train and verify loss
  model <- probotMakePoint(3, 2, hidden_dims = c(16, 16))()
  res <- probotTrainPoint(
    model = model,
    dataloader = probotDataLoader(x, y, batch = 32),
    optimizer = torch::optim_adam(model$parameters, lr = 1e-3),
    epochs = 5, verbose = FALSE, early_stop = FALSE, split_seed = 7
  )
  res$history
  res$validation$n_verify
}
}
