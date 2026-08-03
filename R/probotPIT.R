probotPIT <- function(
    mdn_output,
    params,
    n_samples = 1e4,
    col_means,
    col_sds,
    col_names = NULL,
    verbose = TRUE
){

  n_test <- nrow(params)
  n_params <- ncol(params)

  pit <- matrix(
    NA_real_,
    n_test,
    n_params
  )

  for(i in 1:n_test){

    samples <- probotSamplePostMDN(
      mdn_output,
      galaxy_index = i,
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
