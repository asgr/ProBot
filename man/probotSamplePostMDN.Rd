\name{probotPost}
\alias{probotSamplePostMDN}
\alias{probotMarginalPostMDN}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
A Capitalized Title (ideally limited to 65 characters)
}
\description{
%%  ~~ A concise (1-5 lines) description of what the function does. ~~
}
\usage{
probotSamplePostMDN(mdn_output, galaxy_index = 1, n_samples = 5000, col_means, col_sds,
  col_names = NULL)
probotMarginalPostMDN = function(mdn_output, col_means, col_sds, col_names = NULL)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{mdn_output}{
%%     ~~Describe \code{mdn_output} here~~
}
  \item{galaxy_index}{
%%     ~~Describe \code{galaxy_index} here~~
}
  \item{n_samples}{
%%     ~~Describe \code{n_samples} here~~
}
  \item{col_means}{
%%     ~~Describe \code{col_means} here~~
}
  \item{col_sds}{
%%     ~~Describe \code{col_sds} here~~
}
  \item{col_names}{
%%     ~~Describe \code{col_names} here~~
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
function (mdn_output, galaxy_index = 1, n_samples = 5000, col_means,
    col_sds, col_names = NULL)
{
    n_dim <- length(col_means)
    n_components <- length(as.numeric(mdn_output$logits[galaxy_index,
        ]))
    weights <- as.numeric(nnf_softmax(mdn_output$logits[galaxy_index,
        ], dim = 1))
    mu <- as.matrix(mdn_output$mu[galaxy_index, 1:n_components,
        1:n_dim])
    sigma <- 10^as.matrix(mdn_output$log10_sigma[galaxy_index,
        1:n_components, 1:n_dim])
    comp <- sample(1:n_components, n_samples, replace = TRUE,
        prob = weights)
    samples <- matrix(NA_real_, n_samples, n_dim)
    for (k in 1:n_components) {
        idx <- which(comp == k)
        if (length(idx) == 0)
            next
        samples[idx, ] <- matrix(rnorm(length(idx) * n_dim, mean = rep(mu[k,
            ], each = length(idx)), sd = rep(sigma[k, ], each = length(idx))),
            ncol = n_dim)
    }
    samples <- probotScaleBackward(samples, col_means, col_sds)
    colnames(samples) <- col_names
    return(samples)
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
