\name{probotMakeMDN}
\alias{probotMakeMDN}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
A Capitalized Title (ideally limited to 65 characters)
}
\description{
%%  ~~ A concise (1-5 lines) description of what the function does. ~~
}
\usage{
probotMakeMDN(input_dim, output_dim, mdn_components, hidden_dims = c(128, 256, 256), activation = nnf_relu)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{input_dim}{
%%     ~~Describe \code{input_dim} here~~
}
  \item{output_dim}{
%%     ~~Describe \code{output_dim} here~~
}
  \item{mdn_components}{
%%     ~~Describe \code{mdn_components} here~~
}
  \item{hidden_dims}{
%%     ~~Describe \code{hidden_dims} here~~
}
  \item{activation}{
%%     ~~Describe \code{activation} here~~
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
function (input_dim, output_dim, mdn_components, hidden_dims = c(128, 
    256, 256), activation = nnf_relu) 
{
    nn_module("mlp", initialize = function() {
        dims <- c(input_dim, hidden_dims, mdn_components * (2 * 
            output_dim + 1))
        self$layers <- nn_module_list()
        for (i in seq_len(length(dims) - 1)) {
            self$layers$append(nn_linear(dims[i], dims[i + 1]))
        }
    }, forward = function(x) {
        n_layers <- length(self$layers)
        for (i in seq_len(n_layers - 1)) {
            x <- self$layers[[i]](x)
            x <- activation(x)
        }
        x <- self$layers[[n_layers]](x)
        x
    })
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
