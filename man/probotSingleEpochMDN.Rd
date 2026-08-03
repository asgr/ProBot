\name{probotSingleEpochMDN}
\alias{probotSingleEpochMDN}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
A Capitalized Title (ideally limited to 65 characters)
}
\description{
%%  ~~ A concise (1-5 lines) description of what the function does. ~~
}
\usage{
probotSingleEpochMDN(model, dataloader, optimizer, n_components, lambda = 0.01)
probotTrainMDN (model, dataloader, optimizer, n_components, lambda = 0.01, epochs = 100,
  checkpoint_dir = NULL, checkpoint_every = 10, history = NULL, early_stop = TRUE,
  stop_window = 20, stop_delta = 1e-2, verbose = TRUE)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{model}{
%%     ~~Describe \code{model} here~~
}
  \item{dataloader}{
%%     ~~Describe \code{dataloader} here~~
}
  \item{optimizer}{
%%     ~~Describe \code{optimizer} here~~
}
  \item{n_components}{
%%     ~~Describe \code{n_components} here~~
}
  \item{lambda}{
%%     ~~Describe \code{lambda} here~~
}
  \item{epochs}{
%%     ~~Describe \code{lambda} here~~
}
  \item{checkpoint_dir}{
%%     ~~Describe \code{lambda} here~~
}
  \item{checkpoint_every}{
%%     ~~Describe \code{lambda} here~~
}
  \item{history}{
%%     ~~Describe \code{lambda} here~~
}
  \item{early_stop}{
%%     ~~Describe \code{lambda} here~~
}
  \item{stop_window}{
%%     ~~Describe \code{lambda} here~~
}
  \item{stop_delta}{
%%     ~~Describe \code{lambda} here~~
}
  \item{verbose}{
%%     ~~Describe \code{lambda} here~~
}
}
\details{
%%  ~~ If necessary, more details than the description above ~~
}
\value{
%%  ~Describe the value returned
%%  If it is a LIST, use
%%  \item{comp1 }{Description of 'comp1'}
%%  \item{comp2 }{Description of 'comp2'}
%% ...
}
\references{
%% ~put references to the literature/web site here ~
}
\author{
%%  ~~who you are~~
}
\note{
%%  ~~further notes~~
}

%% ~Make other sections like Warning with \section{Warning }{....} ~

\seealso{
%% ~~objects to See Also as \code{\link{help}}, ~~~
}
\examples{
##---- Should be DIRECTLY executable !! ----
##-- ==>  Define data, use random,
##--	or standard data sets, see data().

## The function is currently defined as
function (model, dataloader, optimizer, n_components, lambda = 0.01)
{
    model$train()
    running_loss <- 0
    running_mae <- 0
    running_rmse <- 0
    running_sigma <- 0
    running_mix <- numeric(n_components)
    n_batches <- 0
    coro::loop(for (batch in dataloader) {
        optimizer$zero_grad()
        output_pred <- model(batch$x)
        p <- .probotUnpackMDN(output_pred, n_components)
        weights <- nnf_softmax(p$logits, dim = 2)
        mu_mix <- (weights$unsqueeze(3) * p$mu)$sum(dim = 2)
        mse_loss <- ((batch$y - mu_mix)^2)$mean()
        mae <- (batch$y - mu_mix)$abs()$mean()$item()
        rmse <- ((batch$y - mu_mix)^2)$mean()$sqrt()$item()
        mean_sigma <- (10^torch_clamp(p$log10_sigma, min = -5,
            max = 5))$mean()$item()
        mix_use <- apply(as.array(weights), 2, mean)
        loss <- probotLossMDN(batch$y, output_pred, n_components)
        loss$backward()
        optimizer$step()
        running_loss <- running_loss + loss$item()
        running_mae <- running_mae + mae
        running_rmse <- running_rmse + rmse
        running_sigma <- running_sigma + mean_sigma
        running_mix <- running_mix + mix_use
        n_batches <- n_batches + 1
    })
    list(loss = running_loss/n_batches, mae = running_mae/n_batches,
        rmse = running_rmse/n_batches, sigma = running_sigma/n_batches,
        mix = running_mix/n_batches)
  }
}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory (show via RShowDoc("KEYWORDS")):
% \keyword{ ~kwd1 }
% \keyword{ ~kwd2 }
% Use only one keyword per line.
% For non-standard keywords, use \concept instead of \keyword:
% \concept{ ~cpt1 }
% \concept{ ~cpt2 }
% Use only one concept per line.
