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

  for(i in seq_len(n_test)){

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

probotTARP <- function(
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
  
  tarp <- numeric(n_test)
  
  for(i in seq_len(n_test)){
    
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
  
  w <- .probotCRPSWeights(n_samples)
  
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
      crps[i, j] <- .probotCRPSSample(samples[, j], params[i, j], w)
    }

    if(verbose && i %% 1000 == 0)
      cat(i, "\n")
  }

  colnames(crps) <- col_names

  return(crps)
}

# Weights for the sorted-sample penalty term (2k - S - 1) / S^2.
# The CRPS estimator's spread term is
#   (1/(2 S^2)) * sum_i sum_j |z_i - z_j|,
# which for sorted z equals sum_k ((2k - S - 1) / S^2) * z_(k).
.probotCRPSWeights = function(n_samples) {
  (2 * seq_len(n_samples) - (n_samples + 1L)) / n_samples^2
}

# CRPS for a single parameter dimension from posterior samples.
.probotCRPSSample = function(samples, y, w) {
  mean(abs(y - samples)) - sum(w * sort(samples))
}

