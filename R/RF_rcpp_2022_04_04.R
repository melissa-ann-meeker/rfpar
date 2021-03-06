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
rf_cpp_par = function(outcome, features, training_set, test_set, n_feats = floor(sqrt(ncol(features))), ntrees = 100, nlevels = 5, min_node_size = 2, setseed = FALSE, binary_outcome = TRUE, n_cores = parallel::detectCores()-2){

  is_training = c(1:length(outcome)) %in% training_set
  
  forest = list()
  
  training_outcome = outcome[training_set]
  training_features = features[training_set,]
  
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
  trees_in_order = FALSE
  if(setseed){
    trees_in_order = TRUE
  }
  
  
  estimates = foreach::foreach(i = 1:ntrees, .packages = "rfpar", .inorder = trees_in_order) %dopar% {
    RNGkind(sample.kind = "Rounding")
    if(setseed){    
      set.seed(i)
      bagged_sample = sample(c(1:nrow(training_features)), replace = TRUE, size = nrow(training_features))
    } else {
      bagged_sample = sample(c(1:nrow(training_features)), replace = TRUE, size = nrow(training_features))
    }
    
    bag_outcome = training_outcome[bagged_sample]
    bag_feats = training_features[bagged_sample,]
    
    
    if(binary_outcome){
      tree = generate_class_tree_cpp(bag_outcome, bag_feats, n_feats, min_node_size = min_node_size, setseed = setseed)
      colnames(tree) = c("ch1", "ch2", "feat_index", "split_value", "weighted_impurity", "impurity_1", "impurity_2", "prediction_value", "is_predictive", "node_size", "isCh1")
      tree[,"prediction_value"] = as.numeric(ifelse(tree[,"prediction_value"]==0, out_values[1], out_values[2]))
      forest[[i]] = tree
    } else {
      tree = generate_reg_tree_cpp(bag_outcome, bag_feats, n_feats, min_node_size = min_node_size, setseed = setseed)
      colnames(tree) = c("ch1", "ch2", "feat_index", "split_value", "weighted_impurity", "impurity_1", "impurity_2", "prediction_value", "is_predictive", "node_size", "isCh1")
      forest[[i]] = tree
    }
    
  }

  if(binary_outcome){
    predictions = predict_class_cpp(estimates, features)
    
    #Accuracy is calculated as if the predicted outcome equals the true outcome
    accuracy = outcome == predictions
    
    #Error 1: Resubstitution Estimation
    accuracy1 = accuracy[training_set]
    e1 = sum(accuracy1)/length(accuracy1)
    
    #Error 2: Test Sample Estimation
    accuracy2 = accuracy[test_set]
    e2 = sum(accuracy2)/length(accuracy2)
    
  } else {
    predictions = predict_reg_cpp(estimates, features)
    
    #Accuracy is calculated as square error
    accuracy = (outcome-predictions)^2

    #Error 1: Resubstitution Estimation
    e1 = mean(accuracy[training_set])
    
    #Error 2: Test Sample Estimation
    e2 = mean(accuracy[test_set])
  }
  
  accuracy = cbind(c(1:length(accuracy)), is_training, accuracy)
  colnames(accuracy) = c("observation", "is_training", "accuracy")

  
  errors = c(e1, e2)
  
  names(errors) = c("resubstitution", "test")
  
  return_items = list(estimates, accuracy, errors)
  names(return_items) = c("forest","accuracy", "errors")
  
  return(return_items)
}

rf_cpp = function(outcome, features, training_set, test_set, n_feats = floor(sqrt(ncol(features))), ntrees = 100, nlevels = 5, min_node_size = 2, setseed = FALSE, binary_outcome = TRUE){
  
  is_training = c(1:length(outcome)) %in% training_set
  
  RNGkind(sample.kind = "Rounding")
  forest = list()
  
  training_outcome = outcome[training_set]
  training_features = features[training_set,]

  
  if(binary_outcome){
    out_values = unique(outcome)
    outcome = as.numeric(ifelse(outcome==out_values[1],0,1))
  }
  
  for(i in 1:ntrees){
    
    if(setseed){
      set.seed(i)
      bagged_sample = sample(c(1:nrow(training_features)), replace = TRUE, size = nrow(training_features))
    } else {
      bagged_sample = sample(c(1:nrow(training_features)), replace = TRUE, size = nrow(training_features))
    }
    
    bag_outcome = training_outcome[bagged_sample]
    bag_feats = training_features[bagged_sample,]
    
    
    if(binary_outcome){
      tree = generate_class_tree_cpp(bag_outcome, bag_feats, n_feats, min_node_size = min_node_size, setseed = setseed)
      colnames(tree) = c("ch1", "ch2", "feat_index", "split_value", "weighted_impurity", "impurity_1", "impurity_2", "prediction_value", "is_predictive", "node_size", "isCh1")
      tree[,"prediction_value"] = as.numeric(ifelse(tree[,"prediction_value"]==0, out_values[1], out_values[2]))
      forest[[i]] = tree
    } else {
      tree = generate_reg_tree_cpp(bag_outcome, bag_feats, n_feats, min_node_size = min_node_size, setseed = setseed)
      colnames(tree) = c("ch1", "ch2", "feat_index", "split_value", "weighted_impurity", "impurity_1", "impurity_2", "prediction_value", "is_predictive", "node_size", "isCh1")
      forest[[i]] = tree
    }
    
  }
  
  if(binary_outcome){
    predictions = predict_class_cpp(forest, features)
    
    #Accuracy is calculated as if the predicted outcome equals the true outcome
    accuracy = outcome == predictions
    
    #Error 1: Resubstitution Estimation
    accuracy1 = accuracy[training_set]
    e1 = sum(accuracy1)/length(accuracy1)
    
    #Error 2: Test Sample Estimation
    accuracy2 = accuracy[test_set]
    e2 = sum(accuracy2)/length(accuracy2)
    
  } else {
    predictions = predict_reg_cpp(forest, features)
    
    #Accuracy is calculated as square error
    accuracy = (outcome-predictions)^2
    
    #Error 1: Resubstitution Estimation
    e1 = mean(accuracy[training_set])
    
    #Error 2: Test Sample Estimation
    e2 = mean(accuracy[test_set])
  }
  
  accuracy = cbind(c(1:length(accuracy)), is_training, accuracy)
  colnames(accuracy) = c("observation", "is_training", "accuracy")
  
  
  errors = c(e1, e2)
  
  names(errors) = c("resubstitution", "test")
  
  return_items = list(forest, accuracy, errors)
  names(return_items) = c("forest","accuracy", "errors")
  
  return(return_items)
}

