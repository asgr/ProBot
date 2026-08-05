probotPIT <- function(
    mdn_output,
    params,
    n_test = NULL,
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

  n_params <- ncol(params)

  pit <- matrix(
    NA_real_,
    n_test,
    n_params
  )

  for(i in 1:n_test){

    samples <- probotSamplePostMDN(
      mdn_output,
      index = i,
      n_samples = n_samples,
      col_means = col_means,
      col_sds = col_sds,
      col_names = col_names
    )

    for(j in 1:n_params){
      pit[i, j] <- mean(samples[, j] <= params[i, j])
    }

    if(verbose && i %% 1000 == 0)
      cat(i, "\n")

  }

  colnames(pit) <- col_names

  return(pit)
}
