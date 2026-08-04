probotScaleForward = function(input, col_means=NULL, col_sds=NULL){
  if(is.null(col_means)){
    col_means = collapse::fmean(input, na.rm = TRUE)
  }
  
  if(is.null(col_sds)){
    col_sds = collapse::fsd(input, na.rm = TRUE)
  }

  output = collapse::TRA(input, col_means, "-")
  output = collapse::TRA(output, col_sds, "/")

  attributes(output)$col_scale = list(col_means=col_means, col_sds=col_sds)

  return(output)
}

probotScaleBackward = function(input, col_means=NULL, col_sds=NULL){

  if(is.null(col_means)){
    col_means = attributes(input)$col_scale$col_means
  }

  if(is.null(col_sds)){
    col_sds = attributes(input)$col_scale$col_sds
  }

  if(is.null(col_means)){
    stop('Missing col_means either as argument or input attribute!')
  }

  if(is.null(col_sds)){
    stop('Missing col_sds either as argument or input attribute!')
  }

  if(length(col_means) == 1){
    col_means = rep(col_means, ncol(input))
  }

  if(length(col_sds) == 1){
    col_sds = rep(col_sds, ncol(input))
  }

  output = collapse::TRA(input, col_sds, "*")
  output = collapse::TRA(output, col_means, "+")

  return(output)
}
