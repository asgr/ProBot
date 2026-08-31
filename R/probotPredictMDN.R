probotPredictMDN <- function(input, model, mdn_components, device = NULL){

  if (is.null(device)) {
    if (length(model$parameters) > 0) {
      device <- model$parameters[[1]]$device
    } else {
      device <- if (backends_mps_is_available()) torch_device("mps") else torch_device("cpu")
    }
  }

  model$eval()

  with_no_grad({
    output <- model(
      torch_tensor(
        input,
        dtype = torch_float(),
        device = device
      )
    )
  })

  .probotUnpackMDN(output, mdn_components)
}

probotSamplePostMDN <- function(
    mdn_output,
    index = 1,
    n_samples = 5000,
    col_means,
    col_sds,
    col_names = NULL
){

  n_dim <- length(col_means)

  mdn_components <- length(
    as.numeric(
      mdn_output$logits[index,]
    )
  )

  weights <- as.numeric(
    nnf_softmax(
      mdn_output$logits[index,],
      dim = 1
    )
  )

  mu <- as.matrix(
    mdn_output$mu[
      index,
      1:mdn_components,
      1:n_dim
    ]
  )

  sigma <- 10^pmin(
    pmax(
      as.matrix(
        mdn_output$log10_sigma[
          index,
          1:mdn_components,
          1:n_dim
        ]
      ),
      -5
    ),
    5
  )

  comp <- sample(
    1:mdn_components,
    n_samples,
    replace = TRUE,
    prob = weights
  )

  samples <- matrix(
    NA_real_,
    n_samples,
    n_dim
  )

  for(k in 1:mdn_components){

    idx <- which(comp == k)

    if(length(idx) == 0)
      next

    samples[idx,] <- matrix(
      rnorm(
        length(idx) * n_dim,
        mean = rep(mu[k,], each = length(idx)),
        sd = rep(sigma[k,], each = length(idx))
      ),
      ncol = n_dim
    )
  }

  # ----------------------------------
  # Undo standardization
  # ----------------------------------

  samples <- probotScaleBackward(samples, col_means, col_sds)

  colnames(samples) <- col_names

  return(samples)
}

probotMarginalPostMDN = function(mdn_output,
                              col_means,
                              col_sds,
                              col_names = NULL){
  weights <- as.array(
    nnf_softmax(
      mdn_output$logits,
      dim = 2
    )
  )

  mu <- as.array(mdn_output$mu)

  sigma <- 10^pmin(pmax(as.array(mdn_output$log10_sigma), -5), 5)

  # ======================================================
  # Posterior mean
  # ======================================================

  N <- dim(mu)[1]
  K <- dim(mu)[2]
  D <- dim(mu)[3]

  post_mean_scaled <- matrix(0, N, D)

  for(k in 1:K){
    post_mean_scaled <- post_mean_scaled + weights[,k] * mu[,k,]
  }

  # ======================================================
  # Posterior variance
  # ======================================================

  second_moment <- matrix(0, N, D)

  for(k in 1:K){
    second_moment <- second_moment + weights[,k] * (sigma[,k,]^2 + mu[,k,]^2)
  }

  post_var_scaled <- second_moment - post_mean_scaled^2

  post_var_scaled[] <- pmax(post_var_scaled, 0)

  post_sd_scaled <- sqrt(post_var_scaled)

  # ======================================================
  # Convert back to physical units
  # ======================================================

  post_mean <- probotScaleBackward(post_mean_scaled, col_means, col_sds)
  post_sd <- probotScaleBackward(post_sd_scaled, 0, col_sds)

  colnames(post_sd) = col_names
  colnames(post_mean) = col_names

  return(list(post_mean=post_mean, post_sd=post_sd))
}
