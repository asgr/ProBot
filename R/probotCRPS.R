probotCRPS <- function(
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
   } else {
    n_test = pmin(n_test, nrow(params), na.rm = TRUE)
   }

  n_params <- ncol(params)

  crps <- matrix(NA_real_, n_test, n_params)

  # Weights for the sorted-sample penalty term: (2k - S - 1) / (2 * S^2)
  w <- (2:(n_samples + 1L) - (n_samples + 1)) / (2 * n_samples^2)

  for(i in seq_len(n_test)){

    samples <- probotSamplePostMDN(
      mdn_output,
      index = i,
      n_samples = n_samples,
      col_means = col_means,
      col_sds = col_sds,
      col_names = col_names
     )

    for(j in seq_len(n_params)){
      z <- sort(samples[, j])
      crps[i, j] <- mean(abs(params[i, j] - samples[, j])) - sum(w * z)
     }

    if(verbose && i %% 1000 == 0)
      cat(i, "\n")
   }

  colnames(crps) <- col_names

  return(crps)
}
