\name{probotLossMDN}
\alias{probotLossMDN}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
A Capitalized Title (ideally limited to 65 characters)
}
\description{
%%  ~~ A concise (1-5 lines) description of what the function does. ~~
}
\usage{
probotLossMDN(output_true, output_pred, n_components)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{output_true}{
%%     ~~Describe \code{output_true} here~~
}
  \item{output_pred}{
%%     ~~Describe \code{output_pred} here~~
}
  \item{n_components}{
%%     ~~Describe \code{n_components} here~~
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
function (output_true, output_pred, n_components) 
{
    output_dim <- output_true$size(2)
    p <- .probotUnpackMDN(output_pred, n_components)
    mu <- p$mu
    log10_sigma <- torch_clamp(p$log10_sigma, min = -5, max = 5)
    sigma <- 10^log10_sigma
    y_true <- output_true$unsqueeze(2)
    y_true <- y_true$expand(c(y_true$size(1), n_components, output_dim))
    z <- (y_true - mu)/sigma
    log_prob <- -0.5 * (z^2 + 2 * log10_sigma * log(10) + log(2 * 
        pi))
    log_prob <- log_prob$sum(dim = 3)
    log_pi <- nnf_log_softmax(p$logits, dim = 2)
    log_mix_prob <- torch_logsumexp(log_pi + log_prob, dim = 2)
    -log_mix_prob$mean()
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
