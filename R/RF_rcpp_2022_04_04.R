######################################################
# AUTHOR: MELISSA MEEKER
# WORKING ON CODING THE GINI IMPURITY FOR RF ALGORITHM
# Update on April 8 --> Was having issues with global and local scopes which was causing the algorithm to break
# fixed by using variable name "bagged_feats" and "bagged_outcome" throughout
# all print lines commented out
# Update on May 14, 2021 --> Changing cut points to middle points rather than unique values
# Update on Feb 08, 2022 --> add option for gini impurity index
# Update on March 07, 2022 --> add calculateErrorBinary vs calculateErrorContinuous
######################################################

#######################################################################################################
#######################################################################################################
################################          FUNCTIONS          ##########################################
#######################################################################################################
#######################################################################################################

#function for random forest
#outcome: a vector of the outcome values
rf_cpp_par = function(outcome, features, n_feats = floor(sqrt(ncol(features))), ntrees = 100, nlevels = 5, min_node_size = 2, setseed = FALSE, binary_outcome = TRUE, n_cores = parallel::detectCores()-2){

  forest = list()
  
  if(binary_outcome){
    out_values = unique(outcome)
    outcome = as.numeric(ifelse(outcome==out_values[1],0,1))
  }
  
  # Construct cluster
  cl = parallel::makeCluster(n_cores)
  
  # After the function is run, close the cluster.
  on.exit(parallel::stopCluster(cl))
  
  # Register parallel backend
  doParallel::registerDoParallel(cl)
  
  
  # Compute estimates
  i = NULL
  estimates = foreach::foreach(i = 1:ntrees, .packages = "rfpar", .inorder = TRUE) %dopar% {
    RNGkind(sample.kind = "Rounding")
    if(setseed){    
      set.seed(i)
      bagged_sample = sample(c(1:nrow(features)), replace = TRUE, size = nrow(features))
    } else {
      bagged_sample = sample(c(1:nrow(features)), replace = TRUE, size = nrow(features))
    }
    
    bag_outcome = outcome[bagged_sample]
    bag_feats = features[bagged_sample,]
    
    
    if(binary_outcome){
      forest[[i]] = generate_class_tree_cpp(bag_outcome, bag_feats, n_feats, min_node_size = min_node_size, setseed = setseed)
      forest[[i]]$value = as.numeric(ifelse(forest[[i]]$value==0, out_values[1], out_values[2]))
    } else {
      forest[[i]] = generate_reg_tree_cpp(bag_outcome, bag_feats, n_feats, min_node_size = min_node_size, setseed = setseed)
    }
  }
  
  return(estimates)
}

rf_cpp = function(outcome, features, n_feats = floor(sqrt(ncol(features))), ntrees = 100, nlevels = 5, min_node_size = 2, setseed = FALSE, binary_outcome = TRUE){
  RNGkind(sample.kind = "Rounding")
  forest = list()
  
  if(binary_outcome){
    out_values = unique(outcome)
    outcome = as.numeric(ifelse(outcome==out_values[1],0,1))
  }
  
  for(i in 1:ntrees){

    if(setseed){
      set.seed(i)
      bagged_sample = sample(c(1:nrow(features)), replace = TRUE, size = nrow(features))
    } else {
      bagged_sample = sample(c(1:nrow(features)), replace = TRUE, size = nrow(features))
    }

    bag_outcome = outcome[bagged_sample]
    bag_feats = features[bagged_sample,]


    if(binary_outcome){
      forest[[i]] = generate_class_tree_cpp(bag_outcome, bag_feats, n_feats, min_node_size = min_node_size, setseed = setseed)
      forest[[i]]$value = as.numeric(ifelse(forest[[i]]$value==0, out_values[1], out_values[2]))
    } else {
      forest[[i]] = generate_reg_tree_cpp(bag_outcome, bag_feats, n_feats, min_node_size = min_node_size, setseed = setseed)
    }

  }
  
  return(forest)
}


