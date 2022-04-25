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
################################     GENERIC FUNCTIONS       ##########################################
#######################################################################################################

calculate_mse_r = function(x){
  mean_x = mean(x)
  error = mean(x)-x
  sq_error = error^2
  return(mean(sq_error))
}

calculate_gini_r = function(v){
  n = length(v)
  p1 = sum(v)/n
  p2 = 1 - p1
  gini_impurity = 1 - (p1^2+p2^2)
  return(gini_impurity)
}

numeric_mode_r = function(v) {
  uniqv <- unique(v)
  uniqv[which.max(tabulate(match(v, uniqv)))]
}

#######################################################################################################
################################     FOREST GENERATION       ##########################################
#######################################################################################################

rf_r_par = function(outcome, features, training_set, test_set, n_feats = floor(sqrt(ncol(features))), ntrees = 100, min_node_size = 5, setseed = FALSE, binary_outcome = TRUE, n_cores = parallel::detectCores()-2){
  
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
  trees_in_order = FALSE
  if(setseed){
    trees_in_order = TRUE
  }
  
  estimates = foreach::foreach(i = 1:ntrees, .packages = "rfpar", .inorder = trees_in_order) %dopar% {
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
      tree = generate_class_tree_r(bag_outcome, bag_feats, n_feats, min_node_size = min_node_size, setseed = setseed)
      colnames(tree) = c("ch1", "ch2", "feat_index", "split_value", "weighted_impurity", "impurity_1", "impurity_2", "prediction_value", "is_predictive", "node_size", "isCh1")
      tree[,"prediction_value"] = as.numeric(ifelse(tree[,"prediction_value"]==0, out_values[1], out_values[2]))
      forest[[i]] = tree
    } else {
      tree = generate_reg_tree_r(bag_outcome, bag_feats, n_feats, min_node_size = min_node_size, setseed = setseed)
      colnames(tree) = c("ch1", "ch2", "feat_index", "split_value", "weighted_impurity", "impurity_1", "impurity_2", "prediction_value", "is_predictive", "node_size", "isCh1")
      forest[[i]] = tree
    }
  }
  
  
  if(binary_outcome){
    predictions = predict_class_cpp(estimates, features)
    
    #Error 1: Resubstitution Estimation
    accuracy1 = outcome[training_set]==predictions[training_set]
    e1 = sum(accuracy1)/length(accuracy1)
    
    #Error 2: Test Sample Estimation
    accuracy2 = outcome[test_set]==predictions[test_set]
    e2 = sum(accuracy2)/length(accuracy2)
    
  } else {
    predictions = predict_reg_cpp(estimates, features)
    
    #Error 1: Resubstitution Estimation
    e1 = mean((outcome[training_set]-predictions[training_set])^2)
    
    #Error 2: Test Sample Estimation
    e2 = mean((outcome[test_set]-predictions[test_set])^2)
  }
  
  predictions = cbind(c(1:length(predictions)), predictions)
  colnames(predictions) = c("observation", "obs_error")
  
  
  errors = c(e1, e2)
  
  names(errors) = c("resubstitution", "test")
  
  return_items = list(estimates, predictions, errors)
  names(return_items) = c("forest","predictions", "errors")
  
  return(return_items)
}
  
  
rf_r = function(outcome, features, training_set, test_set, n_feats = floor(sqrt(ncol(features))), ntrees = 100, min_node_size = 5, setseed = FALSE, binary_outcome = TRUE){
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
      tree = generate_class_tree_r(bag_outcome, bag_feats, n_feats, min_node_size = min_node_size, setseed = setseed)
      colnames(tree) = c("ch1", "ch2", "feat_index", "split_value", "weighted_impurity", "impurity_1", "impurity_2", "prediction_value", "is_predictive", "node_size", "isCh1")
      tree[,"prediction_value"] = as.numeric(ifelse(tree[,"prediction_value"]==0, out_values[1], out_values[2]))
      forest[[i]] = tree
    } else {
      tree = generate_reg_tree_r(bag_outcome, bag_feats, n_feats, min_node_size = min_node_size, setseed = setseed)
      colnames(tree) = c("ch1", "ch2", "feat_index", "split_value", "weighted_impurity", "impurity_1", "impurity_2", "prediction_value", "is_predictive", "node_size", "isCh1")
      forest[[i]] = tree
    }
    
  }
  
  if(binary_outcome){
    predictions = predict_class_cpp(forest, features)
    
    #Error 1: Resubstitution Estimation
    accuracy1 = outcome[training_set]==predictions[training_set]
    e1 = sum(accuracy1)/length(accuracy1)
    
    #Error 2: Test Sample Estimation
    accuracy2 = outcome[test_set]==predictions[test_set]
    e2 = sum(accuracy2)/length(accuracy2)
    
  } else {
    predictions = predict_reg_cpp(forest, features)  
    
    #Error 1: Resubstitution Estimation
    e1 = mean((outcome[training_set]-predictions[training_set])^2)
    
    #Error 2: Test Sample Estimation
    e2 = mean((outcome[test_set]-predictions[test_set])^2)
  }
  
  predictions = cbind(c(1:length(predictions)), predictions)
  colnames(predictions) = c("observation", "obs_error")
  
  errors = c(e1, e2)
  
  names(errors) = c("resubstitution", "test")
  
  return_items = list(forest, predictions, errors)
  names(return_items) = c("forest","predictions", "errors")
  
  return(return_items)
  
}

#######################################################################################################
############################     REGRESSION TREE GENERATION       #####################################
#######################################################################################################

make_reg_decision_rule_r = function(setseed, outcome, features, n_feats, seed, min_node_size){
  #print(nrow(features))
  feats_to_try = vector("numeric", length = n_feats)
  if(setseed){
    set.seed(seed+1000)
    feats_to_try = sample.int(n = ncol(features), size = n_feats, replace = FALSE)
  } else {
    feats_to_try = sample.int(n = ncol(features), size = n_feats, replace = FALSE)
  }
  
  splits_for_all_features = vector(mode = "list", length = n_feats)
  is_valid_feature = rep(FALSE, times = n_feats)
  
  for(index in c(1:n_feats)){
    feature = features[,feats_to_try[index]]
    unique_values = unique(feature)
    ord = order(unique_values)
    splits = roll_mean(unique_values[ord], n = 2)
    
    counts = table(feature)

    bin = counts[1]
    while(bin<min_node_size){
      splits = tail(splits, -1)
      counts = tail(counts, -1)
      bin = bin+counts[1]
    }

    if(length(splits)>0){
      bin = counts[length(counts)]
      while(bin<min_node_size){
        splits = head(splits, -1)
        counts = head(counts, -1)
        bin = bin+counts[length(counts)]
      }
    }

    if(length(splits)>0){
      is_valid_feature[index] = TRUE
      splits_for_all_features[[index]] = splits
    }
  }
  
  splits_for_all_features = splits_for_all_features[is_valid_feature]
  feats_to_try = feats_to_try[is_valid_feature]
  
  #This boolean will tell us if there is a valid split OR if it will become a predictive node
  has_valid_feature = length(splits_for_all_features)>0
  
  #A matrix to store impurity data for each valid split
  #Columns are -> feature(0), split(1), weighted impurity(2), left impurity(3), right impurity(4)
  results = matrix(NA, nrow = length(feats_to_try), ncol = 5)
  
  if(has_valid_feature){
    for(f in c(1:length(feats_to_try))){
      valid_splits = splits_for_all_features[[f]]
      feature = features[,feats_to_try[f]]
      #print(sort(feature))
      n_valid_splits = length(valid_splits)

      split_data = matrix(NA, nrow = n_valid_splits, ncol = 5)
      split_data[,1] = rep(feats_to_try[f], times = n_valid_splits)
      
      #calculate impurity for each valid split
      for(s in c(1:n_valid_splits)){
        split = valid_splits[s]
        #print(split)

        #split data into two groups based on the split
        group1 = which(feature<=split)
        group2 = which(feature> split)

        n1 = length(group1)
        n2 = length(group2)
        #print(c(n1,n2))
        split_data[s,2] = split
        
        #left impurity or impurity 1
        split_data[s,4] = calculate_mse_r(outcome[group1])
        
        #right impurity or impurity 2
        split_data[s,5] = calculate_mse_r(outcome[group2])
        
        #weighted impurity
        split_data[s,3] = (n1/length(feature))*split_data[s,4]+(n2/length(feature))*split_data[s,5]
        #print(split_data[s,])
      }
      
      #return the row of the split that minimizes weighted impurity
      #print(split_data[which(is.na(split_data[,3])),])
      #print(split_data[which(split_data[,3]==min(split_data[,3])),])
      features_which_minimize_impurity = which(split_data[,3]==min(split_data[,3]))
      if(length(features_which_minimize_impurity)==1){
        results[f,] = split_data[features_which_minimize_impurity,]
      } else {
        results[f,] = split_data[features_which_minimize_impurity[1],]
      }
      #results[f,] = ifelse(length(features_which_minimize_impurity)==1,split_data[features_which_minimize_impurity,],split_data[features_which_minimize_impurity[1],])
    }
  }
  
  
  #print(results)
  if(has_valid_feature){

    best = results[which(results[,3]==min(results[,3])),]
    if(length(best)>5){
      best = best[1,]
    }
  } else {
    best = rep(NA, times = 5)
  }
  
  
  return(best)
}



generate_reg_tree_r = function(bagged_outcome, bagged_feats, n_feats, min_node_size, setseed){
  
  #####################################
  ######### initializing data #########
  #####################################
  
  #initialize some integer values
  sample_size = length(bagged_outcome)
  num_nodes = 2*sample_size+1
  
  #matrix to indicate sample sets
  sample_set = matrix(0, nrow = num_nodes, ncol = sample_size)
  sample_set[1,] = 1
  
  #matrix to demonstrate tree structure
  tree = matrix(NA, nrow = num_nodes, ncol = 11)
  colnames(tree) = c("ch1", "ch2", "feat_index", "split_value", "weighted_impurity", "impurity_1", "impurity_2", "prediction_value", "is_predictive", "node_size", "isCh1")
  
  node_i = 1
  
  #####################################
  ############# functions #############
  #####################################
  
  build_reg_nodes_r = function(is_ch1){
    this_node = node_i
    set = which(sample_set[node_i,]==1)
    #print(sample_set[1:10,1:10])
    node_size = length(set)
    #print(set)
    #print( bagged_feats[set,41])
    split_info = make_reg_decision_rule_r(setseed, bagged_outcome[set], bagged_feats[set,], n_feats, node_i, min_node_size)
    #print(split_info)
    split_feature = split_info[1]
    split_value = split_info[2]
    
    #if there is no valid decision rule:
    if(all(is.na(split_info))){
      prediction = mean(bagged_outcome[set])
      #add predictive node
      tree[node_i,] <<- c(NA, NA, NA, NA, NA, NA, NA, prediction, TRUE, node_size, is_ch1)
    } else {
      #add decision node
      tree[node_i,] <<- c(NA, NA, split_info[1], split_info[2], split_info[3], split_info[4], split_info[5], NA, FALSE, node_size, is_ch1)
      
      ch1_sample = set[which(bagged_feats[set, split_feature]<=split_value)]
      ch2_sample = set[which(bagged_feats[set, split_feature]> split_value)]
      
      ch1_sample_size = length(ch1_sample)
      ch2_sample_size = length(ch2_sample)
      #print(c(ch1_sample_size,ch2_sample_size))
      
      add_ch1_pred_node = (ch1_sample_size<2*min_node_size) | (split_info[4]==0)
      add_ch2_pred_node = (ch2_sample_size<2*min_node_size) | (split_info[5]==0)
      
      #_______________
      # add children
      #_______________
      
      #child 1
      node_i <<- node_i + 1
      sample_set[node_i,ch1_sample] <<- 1
      if(add_ch1_pred_node){
        #adding child 1 predictive node
        tree[this_node,1] <<- node_i
        prediction = mean(bagged_outcome[ch1_sample])
        tree[node_i,] <<- c(NA, NA, NA, NA, NA, NA, NA, prediction, TRUE, ch1_sample_size, TRUE)
      } else {
        #create decision rule -> call recursive function
        tree[this_node,1] <<- node_i
        build_reg_nodes_r(TRUE)
      }
      
      #child 2
      node_i <<- node_i + 1
      sample_set[node_i,ch2_sample] <<- 1
      if(add_ch2_pred_node){
        #adding child 1 predictive node
        tree[this_node,2] <<- node_i
        prediction = mean(bagged_outcome[ch2_sample])
        tree[node_i,] <<- c(NA, NA, NA, NA, NA, NA, NA, prediction, TRUE, ch2_sample_size, FALSE)
      } else {
        #create decision rule -> call recursive function
        tree[this_node,2] <<- node_i
        build_reg_nodes_r(FALSE)
      }
    }
    #end of function
  }
  
  build_reg_nodes_r(TRUE)
  
  tree = tree[c(1:node_i),]
  
  tree[,1:2] = tree[,1:2]-1
  tree[,5:8] = round(tree[,5:8], digits = 4)
  
  
  return(tree)
}

#######################################################################################################
##########################     CLASSIFICATION TREE GENERATION       ###################################
#######################################################################################################

make_class_decision_rule_r = function(setseed, outcome, features, n_feats, seed, min_node_size){
  #print(nrow(features))
  feats_to_try = vector("numeric", length = n_feats)
  if(setseed){
    set.seed(seed+1000)
    feats_to_try = sample.int(n = ncol(features), size = n_feats, replace = FALSE)
  } else {
    feats_to_try = sample.int(n = ncol(features), size = n_feats, replace = FALSE)
  }
  
  splits_for_all_features = vector(mode = "list", length = n_feats)
  is_valid_feature = rep(FALSE, times = n_feats)
  
  for(index in c(1:n_feats)){
    feature = features[,feats_to_try[index]]
    unique_values = unique(feature)
    ord = order(unique_values)
    splits = roll_mean(unique_values[ord], n = 2)
    
    counts = table(feature)
    
    bin = counts[1]
    while(bin<min_node_size){
      splits = tail(splits, -1)
      counts = tail(counts, -1)
      bin = bin+counts[1]
    }
    
    if(length(splits)>0){
      bin = counts[length(counts)]
      while(bin<min_node_size){
        splits = head(splits, -1)
        counts = head(counts, -1)
        bin = bin+counts[length(counts)]
      }
    }
    
    if(length(splits)>0){
      is_valid_feature[index] = TRUE
      splits_for_all_features[[index]] = splits
    }
  }
  
  splits_for_all_features = splits_for_all_features[is_valid_feature]
  feats_to_try = feats_to_try[is_valid_feature]
  
  #This boolean will tell us if there is a valid split OR if it will become a predictive node
  has_valid_feature = length(splits_for_all_features)>0
  
  #A matrix to store impurity data for each valid split
  #Columns are -> feature(0), split(1), weighted impurity(2), left impurity(3), right impurity(4)
  results = matrix(NA, nrow = length(feats_to_try), ncol = 5)
  
  if(has_valid_feature){
    for(f in c(1:length(feats_to_try))){
      valid_splits = splits_for_all_features[[f]]
      feature = features[,feats_to_try[f]]
      #print(sort(feature))
      n_valid_splits = length(valid_splits)
      
      split_data = matrix(NA, nrow = n_valid_splits, ncol = 5)
      split_data[,1] = rep(feats_to_try[f], times = n_valid_splits)
      
      #calculate impurity for each valid split
      for(s in c(1:n_valid_splits)){
        split = valid_splits[s]
        #print(split)
        
        #split data into two groups based on the split
        group1 = which(feature<=split)
        group2 = which(feature> split)
        
        n1 = length(group1)
        n2 = length(group2)
        #print(c(n1,n2))
        split_data[s,2] = split
        
        #left impurity or impurity 1
        split_data[s,4] = calculate_gini_r(outcome[group1])
        
        #right impurity or impurity 2
        split_data[s,5] = calculate_gini_r(outcome[group2])
        
        #weighted impurity
        split_data[s,3] = (n1/length(feature))*split_data[s,4]+(n2/length(feature))*split_data[s,5]
        #print(split_data[s,])
      }
      
      #return the row of the split that minimizes weighted impurity
      #print(split_data[which(is.na(split_data[,3])),])
      #print(split_data[which(split_data[,3]==min(split_data[,3])),])
      features_which_minimize_impurity = which(split_data[,3]==min(split_data[,3]))
      if(length(features_which_minimize_impurity)==1){
        results[f,] = split_data[features_which_minimize_impurity,]
      } else {
        results[f,] = split_data[features_which_minimize_impurity[1],]
      }
      #results[f,] = ifelse(length(features_which_minimize_impurity)==1,split_data[features_which_minimize_impurity,],split_data[features_which_minimize_impurity[1],])
    }
  }
  
  
  #print(results)
  if(has_valid_feature){
    
    best = results[which(results[,3]==min(results[,3])),]
    if(length(best)>5){
      best = best[1,]
    }
  } else {
    best = rep(NA, times = 5)
  }
  return(best)
}



generate_class_tree_r = function(bagged_outcome, bagged_feats, n_feats, min_node_size, setseed){
  
  #####################################
  ######### initializing data #########
  #####################################
  
  #initialize some integer values
  sample_size = length(bagged_outcome)
  num_nodes = 2*sample_size+1
  
  #matrix to indicate sample sets
  sample_set = matrix(0, nrow = num_nodes, ncol = sample_size)
  sample_set[1,] = 1
  
  #matrix to demonstrate tree structure
  tree = matrix(NA, nrow = num_nodes, ncol = 11)
  colnames(tree) = c("ch1", "ch2", "feat_index", "split_value", "weighted_impurity", "impurity_1", "impurity_2", "prediction_value", "is_predictive", "node_size", "isCh1")
  
  node_i = 1
  
  #####################################
  ############# functions #############
  #####################################
  
  build_class_nodes_r = function(is_ch1){
    this_node = node_i
    set = which(sample_set[node_i,]==1)
    node_size = length(set)

    split_info = make_class_decision_rule_r(setseed, bagged_outcome[set], bagged_feats[set,], n_feats, node_i, min_node_size)
    split_feature = split_info[1]
    split_value = split_info[2]
    
    #if there is no valid decision rule:
    if(all(is.na(split_info))){
      prediction = numeric_mode_r(bagged_outcome[set])
      #add predictive node
      tree[node_i,] <<- c(NA, NA, NA, NA, NA, NA, NA, prediction, TRUE, node_size, is_ch1)
    } else {
      #add decision node
      tree[node_i,] <<- c(NA, NA, split_info[1], split_info[2], split_info[3], split_info[4], split_info[5], NA, FALSE, node_size, is_ch1)
      
      ch1_sample = set[which(bagged_feats[set, split_feature]<=split_value)]
      ch2_sample = set[which(bagged_feats[set, split_feature]> split_value)]
      
      ch1_sample_size = length(ch1_sample)
      ch2_sample_size = length(ch2_sample)
      #print(c(ch1_sample_size,ch2_sample_size))
      
      add_ch1_pred_node = (ch1_sample_size<2*min_node_size) | (split_info[4]==0)
      add_ch2_pred_node = (ch2_sample_size<2*min_node_size) | (split_info[5]==0)
      
      #_______________
      # add children
      #_______________
      
      #child 1
      node_i <<- node_i + 1
      sample_set[node_i,ch1_sample] <<- 1
      if(add_ch1_pred_node){
        #adding child 1 predictive node
        tree[this_node,1] <<- node_i
        prediction = numeric_mode_r(bagged_outcome[ch1_sample])
        tree[node_i,] <<- c(NA, NA, NA, NA, NA, NA, NA, prediction, TRUE, ch1_sample_size, TRUE)
      } else {
        #create decision rule -> call recursive function
        tree[this_node,1] <<- node_i
        build_class_nodes_r(TRUE)
      }
      
      #child 2
      node_i <<- node_i + 1
      sample_set[node_i,ch2_sample] <<- 1
      if(add_ch2_pred_node){
        #adding child 1 predictive node
        tree[this_node,2] <<- node_i
        prediction = numeric_mode_r(bagged_outcome[ch2_sample])
        tree[node_i,] <<- c(NA, NA, NA, NA, NA, NA, NA, prediction, TRUE, ch2_sample_size, FALSE)
      } else {
        #create decision rule -> call recursive function
        tree[this_node,2] <<- node_i
        build_class_nodes_r(FALSE)
      }
    }
    #end of function
  }
  
  build_class_nodes_r(TRUE)
  
  tree = tree[c(1:node_i),]
  
  tree[,1:2] = tree[,1:2]-1
  tree[,5:8] = round(tree[,5:8], digits = 4)
  
  
  return(tree)
}

