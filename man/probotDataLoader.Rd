\name{probotDataLoader}
\alias{probotDataLoader}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
A Capitalized Title (ideally limited to 65 characters)
}
\description{
%%  ~~ A concise (1-5 lines) description of what the function does. ~~
}
\usage{
probotDataLoader(input, output, batch = 1024, shuffle = TRUE, train_idx = NULL)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{input}{
%%     ~~Describe \code{input} here~~
}
  \item{output}{
%%     ~~Describe \code{output} here~~
}
  \item{batch}{
%%     ~~Describe \code{batch} here~~
}
  \item{shuffle}{
%%     ~~Describe \code{shuffle} here~~
}
  \item{train_idx}{
%%     ~~Describe \code{train_idx} here~~
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
function (input, output, batch = 1024, shuffle = TRUE, train_idx = NULL) 
{
    if (is.null(train_idx)) {
        train_idx = 1:dim(input)[1]
    }
    input_train <- torch_tensor(input[train_idx, ], dtype = torch_float())
    output_train <- torch_tensor(output[train_idx, ], dtype = torch_float())
    dataset <- tensor_dataset(x = input_train, y = output_train)
    dataloader <- dataloader(dataset, batch_size = batch, shuffle = shuffle)
    return(dataloader)
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
