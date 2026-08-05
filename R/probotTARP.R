probotTARP <- function(
    mdn_output,
    params,
    n_test = 1e4,
    n_samples = 1e4,
    col_means,
    col_sds,
    col_names = NULL,
    verbose = TRUE
){

  if(is.null(n_test)){
    n_test <- nrow(params)
  }else{
    n_test = pmin(n_test, nrow(params), na.rm=TRUE)
  }

  tarp <- numeric(n_test)

  for(i in 1:n_test){

    samples <- probotSamplePostMDN(
      mdn_output,
      index = i,
      n_samples = n_samples,
      col_means = col_means,
      col_sds = col_sds,
      col_names = col_names
    )

    # Random direction
    direction <- rnorm(ncol(samples))
    direction <- direction / sqrt(sum(direction^2))

    truth_proj <- sum(params[i, ] * direction)

    sample_proj <- as.vector(samples %*% direction)

    tarp[i] <- mean(sample_proj <= truth_proj)

    if(verbose && i %% 1000 == 0)
      cat(i, "\n")

  }

  return(tarp)
}
