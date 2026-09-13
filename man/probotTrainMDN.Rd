\name{probotTrainMDN}
\alias{probotSingleEpochMDN}
\alias{probotTrainMDN}
\title{Train Mixture Density Networks for One Epoch or Many}
\description{
Run one optimization epoch for an MDN or iterate multiple epochs with optional
early stopping and checkpoint saving.
}
\usage{
probotSingleEpochMDN(model, dataloader, optimizer,
  loss_fn = probotLossMDN, lambda = 0)

probotTrainMDN(model, dataloader, optimizer, epochs = 100,
  loss_fn = probotLossMDN, lambda = 0, checkpoint_dir = NULL,
  checkpoint_every = 10, history = NULL, verbose = TRUE, early_stop = TRUE,
  stop_window = 20, stop_delta = 1e-2, holdout_fraction = 0.1,
  val_dataloader = NULL, val_batch = NULL, split_seed = NULL,
  holdout_stop = TRUE, holdout_min_delta = 1e-3, holdout_patience = 5L)
}
\arguments{
  \item{model}{Instantiated torch module returned by \code{probotMakeMDN(... )()}.}

  \item{dataloader}{Torch data loader yielding \code{$x} and \code{$y} batches.}

  \item{optimizer}{Torch optimizer configured with model parameters.}

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

  \item{holdout_fraction}{Fraction of the dataloader's rows held out for
    verification, scored once per epoch and reported beside the training loss.
    Defaults to 0.1. Set to 0 to train on every row and record no verification
    loss. Ignored when \code{val_dataloader} is supplied. Datasets with fewer
    than 10 rows warn and skip the split.}

  \item{val_dataloader}{Optional \code{\link{probotDataLoader}} used as the
    verification set verbatim, in place of a random split.}

  \item{val_batch}{Rows per verification batch. Defaults to the whole
    verification set at once; affects only speed and memory.}

  \item{split_seed}{Seed for the holdout draw. \code{NULL} (default) uses the
    current state of R's global RNG.}

  \item{holdout_stop}{If TRUE (default) and a verification set exists, stop when
    the training loss is still improving while the verification loss is not.}

  \item{holdout_min_delta}{A verification-loss trend more positive than
    \code{-holdout_min_delta} per epoch counts as "not improving".}

  \item{holdout_patience}{Consecutive epochs the over-training condition must
    hold before stopping.}
}
\details{
Training metrics include loss, MAE, RMSE, average sigma, and average mixture
weights. MAE and RMSE are element-wise means over all target elements
(\eqn{N \times ncol(y)}), not per-row means. Early stopping compares mean
loss over the latest \code{stop_window} epochs to the previous window.

When a verification set exists (the default), \code{history} gains a
\code{val_loss} column, and \code{val_mae}/\code{val_rmse} alongside it whenever
\code{lambda > 0} makes an error metric meaningful. The verification numbers use
the same \code{lambda} blend as training, so the two columns are directly
comparable. Printed lines name both explicitly, e.g.

\if{html}{\out{<pre>}}\if{latex}{\out{\begin{verbatim}}}
Epoch 1 Train Loss 2.998 | Verify Loss 2.777 (12 rows)  MAE 0.841 RMSE 1.037 Sigma 1.186 MixSD 0.008 Mix [0.34 0.33 0.33]
\if{html}{\out{</pre>}}\if{latex}{\out{\end{verbatim}}}

See \code{\link{probotTrainPoint}} for the full description of the split, the
two stopping rules and the reproducibility caveats.
}
\value{
\itemize{
  \item \code{probotSingleEpochMDN()}: List with \code{loss}, \code{mae},
  \code{rmse}, \code{sigma}, and \code{mix}.
  \item \code{probotTrainMDN()}: List with trained \code{model} and
  \code{history} data frame. The history data frame contains columns
  \code{epoch}, \code{loss}, \code{mae}, \code{rmse}, \code{sigma}, and
  \code{mix_sd} (standard deviation of mixture weights across components), plus
  \code{val_loss} (and \code{val_mae}, \code{val_rmse} when \code{lambda > 0})
  when a verification set exists.
  \item When a verification set exists, a third component \code{validation}
  reports \code{n_train}, \code{n_verify}, \code{verify_idx}, \code{source},
  \code{holdout_fraction}, \code{best_epoch}, \code{best_val_loss},
  \code{final_val_loss} and \code{stop_reason}. \code{best_epoch} names the
  epoch with the lowest verification loss; the returned \code{model} is the one
  trained at the stopping or final epoch, which is not necessarily that one.
}
}
\author{
Aaron Robotham
}
\seealso{
\code{\link{probotDataLoader}}, \code{\link{probotLossMDN}},
\code{\link{probotSave}}, \code{\link{probotLossEval}},
\code{\link{probotTrainPoint}}
}
\examples{
if (requireNamespace("torch", quietly = TRUE)) {
  torch::torch_manual_seed(1)
  set.seed(1)

  x <- matrix(rnorm(60 * 3), nrow = 60)
  y <- matrix(rnorm(60 * 2), nrow = 60)

  loader <- probotDataLoader(x, y, batch = 16)
  model <- probotMakeMDN(input_dim = 3, output_dim = 2, mdn_components = 2,
                         hidden_dims = c(16, 16))()
  opt <- torch::optim_adam(model$parameters, lr = 0.01)

  metrics <- probotSingleEpochMDN(
    model = model,
    dataloader = loader,
    optimizer = opt
  )

  metrics$loss

  fit <- probotTrainMDN(model, loader, opt, epochs = 5, verbose = FALSE,
                        early_stop = FALSE, split_seed = 7)
  fit$history
  fit$validation$best_epoch
}
}
