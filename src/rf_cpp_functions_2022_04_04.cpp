#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
using namespace Rcpp;

////////////////////////////////////////////////////////////////////////////////
//                            HELPER FUNCTIONS                                //
////////////////////////////////////////////////////////////////////////////////

// [[Rcpp::export]]
void set_seed_cpp(unsigned int seed) {
  Rcpp::Environment base_env("package:base");
  Rcpp::Function set_seed_r = base_env["set.seed"];
  set_seed_r(seed);
}

// [[Rcpp::export]]
arma::vec find_midpoints_cpp(const arma::vec x){
  //This function is important as it produces midpoints of *unique *sorted values of the vector
  arma::vec copy = unique(x);
  int n_unique_elements = copy.size();
  arma::vec midpoints(n_unique_elements-1);
  for(int i = 0; i<(n_unique_elements-1);i++){
    midpoints[i] = (copy[i]+copy[i+1])/2;
  }
  return midpoints;
}

// [[Rcpp::export]]
double decision_cpp(DataFrame tree, int node, NumericVector observation){
  node = node-1;
  NumericVector feature_splits = tree["feat_index"];
  int feature = feature_splits[node]-1;
  NumericVector split_values = tree["split_value"];
  double split = split_values[node];
  NumericVector outcomes = tree["value"];
  double value = outcomes[node];
  
  NumericVector ch1 = tree["child1"];
  NumericVector ch2 = tree["child2"];
  
  if(!Rcpp::NumericVector::is_na(value)){
    return value;
  } else 
    if(observation[feature]<split){
      return decision_cpp(tree, ch1[node], observation);
    } else {
      return decision_cpp(tree, ch2[node], observation);
    }
}

////////////////////////////////////////////////////////////////////////////////
//                            Regression Trees                                //
////////////////////////////////////////////////////////////////////////////////

// [[Rcpp::export]]
double calculate_mse_cpp(const arma::vec& x){
  double mean_x = mean(x);
  arma::vec errors = x-mean_x;
  arma::vec squareError = pow(errors, 2);
  return mean(squareError);
}

// [[Rcpp::export]]
NumericVector make_reg_decision_rule_cpp(bool ss, const arma::vec& bag_out, const arma::mat& bag_feat, int n_feats, int seed, int min_node_size){
  
  Vector<INTSXP> feats_to_try;
  if(ss){
    set_seed_cpp(seed+1000);
    feats_to_try = Rcpp::sample(bag_feat.n_cols, n_feats, false, R_NilValue, true);
  } else {
    feats_to_try = Rcpp::sample(bag_feat.n_cols, n_feats, false, R_NilValue, true);
  }
  
  Rcpp::List splits_for_all_features(n_feats);
  LogicalVector is_valid_feature = rep(false, n_feats);
  
  for(int index = 0; index<feats_to_try.size(); index++){
    arma::vec feature =bag_feat.col(feats_to_try[index]-1);
    arma::vec splits = find_midpoints_cpp(feature);
    
    arma::vec unique_elements = unique(feature);
    arma::uvec counts = hist(feature, unique_elements);
    //counts = counts.elem(find(counts>0));
    
    int bin = counts(0);
    
    while(bin<min_node_size){
      splits = splits.tail(splits.size()-1);
      counts = counts.tail(counts.size()-1);
      bin = bin + counts(0);
    }
    
    if(splits.size()>0){
      bin = counts(counts.size()-1);
      
      while(bin<min_node_size){
        splits = splits.head(splits.size()-1);
        counts = counts.head(counts.size()-1);
        bin = bin + counts(counts.size()-1);
      }
    }
    
    if(splits.size()>0){
      is_valid_feature[index] = true;
      splits_for_all_features[index] = splits;
    }
  }
  
  splits_for_all_features = splits_for_all_features[is_valid_feature];
  feats_to_try = feats_to_try[is_valid_feature];
  
  //This boolean will tell us if there is a valid split OR if it will become a predictive node
  bool hasValidFeature = splits_for_all_features.size()>0;
  
  //A matrix to store impurity data for each valid split
  //Columns are -> feature(0), split(1), weighted impurity(2), left impurity(3), right impurity(4)
  NumericMatrix results(feats_to_try.size(),5);
  
  if(hasValidFeature){
    for(int f = 0; f<feats_to_try.size(); f++){
      /////////////////////////////
      // findBestSplitContinuous //
      /////////////////////////////
      arma::vec valid_splits = splits_for_all_features[f];
      arma::vec feature = bag_feat.col(feats_to_try[f]-1);
      //A matrix to store impurity data for each valid split
      //Columns are -> split(0), weighted impurity(1), left impurity(2), right impurity(3)
      NumericMatrix split_data(valid_splits.size(),5);
      split_data(_,0) = rep(feats_to_try[f], valid_splits.size());
      
      
      int n_valid_splits = valid_splits.size();
      //Calculate impurity for each valid split
      for(int s = 0; s<n_valid_splits; s++){
        double split = valid_splits[s];
        //This splits the data into two groups based on the split
        arma::vec group1 = feature.elem(find(feature<=split));
        arma::vec group2 = feature.elem(find(feature>split));
        double n1 = group1.size();
        double n2 = group2.size();
        
        split_data(s,1) = split;
        
        //left impurity or impurity 1
        split_data(s,3) = calculate_mse_cpp(bag_out.elem(find(feature<=split)));
        
        //right impurity or impurity 2
        split_data(s,4) = calculate_mse_cpp(bag_out.elem(find(feature>split)));
        
        //weighted impurity
        split_data(s,2) = (n1/feature.size())*split_data(s,3)+(n2/feature.size())*split_data(s,4);

      }

      //return the row of the split that minimizes weighted impurity
      results(f,_) = split_data(which_min(split_data(_,2)),_);
    }  
  }
  
  NumericVector best;
  
  if(hasValidFeature){
    best = results(which_min(results(_,2)),_);
  } else {
    best = {NA_REAL, NA_REAL, NA_REAL, NA_REAL, NA_REAL};
  }
  
  return best;
}

// [[Rcpp::export]]
void build_reg_nodes_cpp(NumericMatrix tree_data, arma::umat& sample_set_data, arma::uword& node_i, const arma::vec& outcome_data, const arma::mat& feats_data, int n_feats_to_consider, int mns, bool seed, double isCh1) {
  
  arma::uword this_node = node_i;
  arma::uvec set = find(sample_set_data.row(node_i)==1);
  double node_size = set.size();
  NumericVector split_info = make_reg_decision_rule_cpp(seed, outcome_data.elem(set), feats_data.rows(set), n_feats_to_consider, node_i+1, mns);
  
  unsigned int sf = abs(round(split_info(0)));
  arma::uvec split_feature = {sf};
  
  double split_value = split_info(1);
  
  //if no valid decision rule
  if(all(is_na(split_info))){
    double prediction = mean(outcome_data.elem(find(sample_set_data.row(node_i)==1)));
    
    //add predictive node
    //"ch1", "ch2", "feat_index", "split_value", "weighted_impurity", "impurity_1", "impurity_2", "prediction_value", "is_predictive", "node_size", "isCh1"}
    NumericVector node = {NA_REAL, NA_REAL, NA_REAL, NA_REAL, NA_REAL, NA_REAL, NA_REAL, prediction, true, node_size, isCh1};
    tree_data(node_i,_) = node;
    
  } else {
    //add decision node
    //"ch1", "ch2", "feat_index", "split_value", "weighted_impurity", "impurity_1", "impurity_2", "prediction_value", "is_predictive", "node_size"}
    NumericVector node = {NA_REAL, NA_REAL, split_info(0), split_info(1), split_info(2), split_info(3), split_info(4), NA_REAL, false, node_size, isCh1};
    tree_data(node_i,_) = node;
    
    arma::uvec ch1_sample = set.elem(find(feats_data.submat(set,split_feature-1)<=split_value));
    arma::uvec ch2_sample = set.elem(find(feats_data.submat(set,split_feature-1)> split_value));
    
    double ch1_sample_size = ch1_sample.size();
    double ch2_sample_size = ch2_sample.size();
    
    bool add_ch1_pred_node = (ch1_sample_size<2*mns) | (split_info(3)==0);
    bool add_ch2_pred_node = (ch2_sample_size<2*mns) | (split_info(4)==0);
    //_____________________________________________
    //check if we should add predictive child nodes
    //_____________________________________________
    node_i++;
    arma::uvec node_identifier = {node_i};
    sample_set_data.submat(node_identifier,ch1_sample).fill(1);
    if(add_ch1_pred_node){
      //adding ch1 predictive node
      tree_data(this_node,0) = node_i;
      double prediction = mean(outcome_data.elem(ch1_sample));
      NumericVector node = {NA_REAL, NA_REAL, NA_REAL, NA_REAL, NA_REAL, NA_REAL, NA_REAL, prediction, true, ch1_sample_size, 1.0};
      tree_data(node_i,_) = node;
      
    } else {
      //create decision rule -> call recursive function
      tree_data(this_node,0) = node_i;
      build_reg_nodes_cpp(tree_data, sample_set_data, node_i, outcome_data, feats_data, n_feats_to_consider, mns, seed, 1.0);
    }
    
    node_i++; 
    node_identifier = {node_i};
    sample_set_data.submat(node_identifier,ch2_sample).fill(1);
    if(add_ch2_pred_node){
      //adding ch2 predictive node
      tree_data(this_node,1) = node_i;
      double prediction = mean(outcome_data.elem(ch2_sample));
      NumericVector node = {NA_REAL, NA_REAL, NA_REAL, NA_REAL, NA_REAL, NA_REAL, NA_REAL, prediction, true, ch2_sample_size, 0.0};
      tree_data(node_i,_) = node;
    } else {
      //create decision rule -> call recursive function
      tree_data(this_node,1) = node_i;
      build_reg_nodes_cpp(tree_data, sample_set_data, node_i, outcome_data, feats_data, n_feats_to_consider, mns, seed, 0.0);
    }
  }
}

// [[Rcpp::export]]
NumericMatrix generate_reg_tree_cpp(const arma::vec& bagged_outcome, const arma::mat& bagged_feats, const int n_feats, const int min_node_size, const bool setseed){

  //initialize some integer values
  int sample_size = bagged_outcome.size();
  int num_nodes = 2*sample_size+1;
  
  //matrix to indicate sample sets
  arma::umat sample_set(num_nodes, sample_size, arma::fill::zeros);
  sample_set.row(0).fill(1);
  
  //So we use an armadillo matrix which i
  NumericMatrix tree(num_nodes, 11);
  CharacterVector tree_names = {"ch1", "ch2", "feat_index", "split_value", "weighted_impurity", "impurity_1", "impurity_2", "prediction_value", "is_predictive", "node_size", "isCh1"};
  colnames(tree) = tree_names;
  
  arma::uword node_index = 0;

  build_reg_nodes_cpp(tree, sample_set, node_index, bagged_outcome, bagged_feats, n_feats, min_node_size, setseed, 1.0);
  
  tree = tree(Range(0,node_index),_);
  tree(_,4) = round(tree(_,4), 4);
  tree(_,5) = round(tree(_,5), 4);
  tree(_,6) = round(tree(_,6), 4);
  tree(_,7) = round(tree(_,7), 4);
  return tree;
}

// // [[Rcpp::export]]
// Rcpp::List generate_reg_forest_cpp(const arma::vec& outcome, const arma::mat& features, int mtry, int ntrees, int min_node_size, bool setseed){
//   Rcpp::List forest(ntrees);
//   
//   int sample_size = features.n_rows;
//   Vector<INTSXP> bag(sample_size);
//   
//   for(int i=0; i<ntrees;i++){
//     
//     if(setseed){
//       set_seed_cpp(i+1);
//       bag = Rcpp::sample(sample_size, sample_size, true, R_NilValue, false);
//     } else{
//       bag = Rcpp::sample(sample_size, sample_size, true, R_NilValue, false);
//     }   
//     
//     arma::uvec bagged_sample= as<arma::uvec>(wrap(bag));
// 
//     arma::vec bag_outcome = outcome.elem(bagged_sample);
//     arma::mat bag_feats = features.rows(bagged_sample);
//     
//     forest[i] = generate_reg_tree_cpp(bag_outcome, bag_feats, mtry, min_node_size = min_node_size, setseed = setseed);
//   
//   }
//   return forest;
// }
  
////////////////////////////////////////////////////////////////////////////////
//                        Prediction Functions                                //
////////////////////////////////////////////////////////////////////////////////  
  

// [[Rcpp::export]]
NumericVector predict_reg_cpp(List forest, NumericMatrix train){
  NumericVector preds = rep(NA_REAL, train.nrow());
  for(int r=0; r<train.nrow();r++){
    NumericVector outcomes = rep(NA_REAL, forest.size());
    NumericVector obs = train(r,_);
    for(int j = 0; j<forest.size(); j++){
      outcomes[j] = decision_cpp(Rcpp::as<Rcpp::DataFrame>(forest[j]), 1, obs);
    }
    preds[r] = mean(outcomes);
  }
  return preds;
}


////////////////////////////////////////////////////////////////////////////////
//                        Classification Trees                                //
////////////////////////////////////////////////////////////////////////////////

// [[Rcpp::export]]
double calculate_gini_cpp(const arma::vec& x){
  int n = x.size();
  double p1 = sum(x)/n;
  double p2 = 1-p1;
  double gini_impurity = 1 - (pow(p1, 2) + pow(p2, 2));
  return gini_impurity;
}

// [[Rcpp::export]]
double numeric_mode_cpp(const arma::vec& x){
  arma::vec unique_elements = unique(x);
  arma::uvec counts = hist(x, unique_elements);
  double mode = unique_elements(counts.index_max());
  return mode;
}


// [[Rcpp::export]]
NumericVector make_class_decision_rule_cpp(bool ss, const arma::vec& bag_out, const arma::mat& bag_feat, int n_feats, int seed, int min_node_size){
  
  Vector<INTSXP> feats_to_try;
  if(ss){
    set_seed_cpp(seed+1000);
    feats_to_try = Rcpp::sample(bag_feat.n_cols, n_feats, false, R_NilValue, true);
  } else {
    feats_to_try = Rcpp::sample(bag_feat.n_cols, n_feats, false, R_NilValue, true);
  }
  
  Rcpp::List splits_for_all_features(n_feats);
  LogicalVector is_valid_feature = rep(false, n_feats);
  
  for(int index = 0; index<feats_to_try.size(); index++){
    arma::vec feature =bag_feat.col(feats_to_try[index]-1);
    arma::vec splits = find_midpoints_cpp(feature);
    
    arma::vec unique_elements = unique(feature);
    arma::uvec counts = hist(feature, unique_elements);
    //counts = counts.elem(find(counts>0));
    
    int bin = counts(0);
    
    while(bin<min_node_size){
      splits = splits.tail(splits.size()-1);
      counts = counts.tail(counts.size()-1);
      bin = bin + counts(0);
    }
    
    if(splits.size()>0){
      bin = counts(counts.size()-1);
      
      while(bin<min_node_size){
        splits = splits.head(splits.size()-1);
        counts = counts.head(counts.size()-1);
        bin = bin + counts(counts.size()-1);
      }
    }
    
    if(splits.size()>0){
      is_valid_feature[index] = true;
      splits_for_all_features[index] = splits;
    }
  }
  
  splits_for_all_features = splits_for_all_features[is_valid_feature];
  feats_to_try = feats_to_try[is_valid_feature];
  
  //This boolean will tell us if there is a valid split OR if it will become a predictive node
  bool hasValidFeature = splits_for_all_features.size()>0;
  
  //A matrix to store impurity data for each valid split
  //Columns are -> feature(0), split(1), weighted impurity(2), left impurity(3), right impurity(4)
  NumericMatrix results(feats_to_try.size(),5);
  
  if(hasValidFeature){
    for(int f = 0; f<feats_to_try.size(); f++){
      /////////////////////////////
      // findBestSplitContinuous //
      /////////////////////////////
      arma::vec valid_splits = splits_for_all_features[f];
      arma::vec feature = bag_feat.col(feats_to_try[f]-1);
      //A matrix to store impurity data for each valid split
      //Columns are -> split(0), weighted impurity(1), left impurity(2), right impurity(3)
      NumericMatrix split_data(valid_splits.size(),5);
      split_data(_,0) = rep(feats_to_try[f], valid_splits.size());
      
      
      int n_valid_splits = valid_splits.size();
      //Calculate impurity for each valid split
      for(int s = 0; s<n_valid_splits; s++){
        double split = valid_splits[s];
        //This splits the data into two groups based on the split
        arma::vec group1 = feature.elem(find(feature<=split));
        arma::vec group2 = feature.elem(find(feature>split));
        double n1 = group1.size();
        double n2 = group2.size();
        
        split_data(s,1) = split;
        
        //left impurity or impurity 1
        split_data(s,3) = calculate_gini_cpp(bag_out.elem(find(feature<=split)));
        
        //right impurity or impurity 2
        split_data(s,4) = calculate_gini_cpp(bag_out.elem(find(feature>split)));
        
        //weighted impurity
        split_data(s,2) = (n1/feature.size())*split_data(s,3)+(n2/feature.size())*split_data(s,4);
        
      }
      
      //return the row of the split that minimizes weighted impurity
      results(f,_) = split_data(which_min(split_data(_,2)),_);
    }  
  }
  
  NumericVector best;
  
  if(hasValidFeature){
    best = results(which_min(results(_,2)),_);
  } else {
    best = {NA_REAL, NA_REAL, NA_REAL, NA_REAL, NA_REAL};
  }
  
  return best;
}

// [[Rcpp::export]]
void build_class_nodes_cpp(NumericMatrix tree_data, arma::umat& sample_set_data, arma::uword& node_i, const arma::vec& outcome_data, const arma::mat& feats_data, int n_feats_to_consider, int mns, bool seed, double isCh1) {
  
  arma::uword this_node = node_i;
  arma::uvec set = find(sample_set_data.row(node_i)==1);
  double node_size = set.size();
  NumericVector split_info = make_class_decision_rule_cpp(seed, outcome_data.elem(set), feats_data.rows(set), n_feats_to_consider, node_i+1, mns);
  
  unsigned int sf = abs(round(split_info(0)));
  arma::uvec split_feature = {sf};
  
  double split_value = split_info(1);
  
  //if no valid decision rule
  if(all(is_na(split_info))){
    double prediction = numeric_mode_cpp(outcome_data.elem(find(sample_set_data.row(node_i)==1)));
    
    //add predictive node
    //"ch1", "ch2", "feat_index", "split_value", "weighted_impurity", "impurity_1", "impurity_2", "prediction_value", "is_predictive", "node_size", "isCh1"}
    NumericVector node = {NA_REAL, NA_REAL, NA_REAL, NA_REAL, NA_REAL, NA_REAL, NA_REAL, prediction, true, node_size, isCh1};
    tree_data(node_i,_) = node;
    
  } else {
    //add decision node
    //"ch1", "ch2", "feat_index", "split_value", "weighted_impurity", "impurity_1", "impurity_2", "prediction_value", "is_predictive", "node_size"}
    NumericVector node = {NA_REAL, NA_REAL, split_info(0), split_info(1), split_info(2), split_info(3), split_info(4), NA_REAL, false, node_size, isCh1};
    tree_data(node_i,_) = node;
    
    arma::uvec ch1_sample = set.elem(find(feats_data.submat(set,split_feature-1)<=split_value));
    arma::uvec ch2_sample = set.elem(find(feats_data.submat(set,split_feature-1)> split_value));
    
    double ch1_sample_size = ch1_sample.size();
    double ch2_sample_size = ch2_sample.size();
    
    bool add_ch1_pred_node = (ch1_sample_size<2*mns) | (split_info(3)==0);
    bool add_ch2_pred_node = (ch2_sample_size<2*mns) | (split_info(4)==0);
    //_____________________________________________
    //create children
    //_____________________________________________
    node_i++;
    arma::uvec node_identifier = {node_i};
    sample_set_data.submat(node_identifier,ch1_sample).fill(1);
    if(add_ch1_pred_node){
      //adding ch1 predictive node
      tree_data(this_node,0) = node_i;
      double prediction = numeric_mode_cpp(outcome_data.elem(ch1_sample));
      NumericVector node = {NA_REAL, NA_REAL, NA_REAL, NA_REAL, NA_REAL, NA_REAL, NA_REAL, prediction, true, ch1_sample_size, 1.0};
      tree_data(node_i,_) = node;
      
    } else {
      //create decision rule -> call recursive function
      tree_data(this_node,0) = node_i;
      build_class_nodes_cpp(tree_data, sample_set_data, node_i, outcome_data, feats_data, n_feats_to_consider, mns, seed, 1.0);
    }
    
    node_i++; 
    node_identifier = {node_i};
    sample_set_data.submat(node_identifier,ch2_sample).fill(1);
    if(add_ch2_pred_node){
      //adding ch2 predictive node
      tree_data(this_node,1) = node_i;
      double prediction = numeric_mode_cpp(outcome_data.elem(ch2_sample));
      NumericVector node = {NA_REAL, NA_REAL, NA_REAL, NA_REAL, NA_REAL, NA_REAL, NA_REAL, prediction, true, ch2_sample_size, 0.0};
      tree_data(node_i,_) = node;
    } else {
      //create decision rule -> call recursive function
      tree_data(this_node,1) = node_i;
      build_class_nodes_cpp(tree_data, sample_set_data, node_i, outcome_data, feats_data, n_feats_to_consider, mns, seed, 0.0);
    }
  }
}

// [[Rcpp::export]]
NumericMatrix generate_class_tree_cpp(const arma::vec& bagged_outcome, const arma::mat& bagged_feats, const int n_feats, const int min_node_size, const bool setseed){
  
  //initialize some integer values
  int sample_size = bagged_outcome.size();
  int num_nodes = 2*sample_size+1;
  
  //matrix to indicate sample sets
  arma::umat sample_set(num_nodes, sample_size, arma::fill::zeros);
  sample_set.row(0).fill(1);
  
  //So we use an armadillo matrix which i
  NumericMatrix tree(num_nodes, 11);
  CharacterVector tree_names = {"ch1", "ch2", "feat_index", "split_value", "weighted_impurity", "impurity_1", "impurity_2", "prediction_value", "is_predictive", "node_size", "isCh1"};
  colnames(tree) = tree_names;
  
  arma::uword node_index = 0;
  
  build_class_nodes_cpp(tree, sample_set, node_index, bagged_outcome, bagged_feats, n_feats, min_node_size, setseed, 1.0);
  
  tree = tree(Range(0,node_index),_);
  
  return tree;
}

// [[Rcpp::export]]
Rcpp::List generate_class_forest_cpp(const arma::vec& outcome, const arma::mat& features, int mtry, int ntrees, int min_node_size, bool setseed){
  Rcpp::List forest(ntrees);
  
  int sample_size = features.n_rows;
  Vector<INTSXP> bag(sample_size);
  
  for(int i=0; i<ntrees;i++){
    
    if(setseed){
      set_seed_cpp(i+1);
      bag = Rcpp::sample(sample_size, sample_size, true, R_NilValue, false);
    } else{
      bag = Rcpp::sample(sample_size, sample_size, true, R_NilValue, false);
    }   
    
    arma::uvec bagged_sample= as<arma::uvec>(wrap(bag));
    
    arma::vec bag_outcome = outcome.elem(bagged_sample);
    arma::mat bag_feats = features.rows(bagged_sample);
    
    forest[i] = generate_class_tree_cpp(bag_outcome, bag_feats, mtry, min_node_size = min_node_size, setseed = setseed);
    
  }
  return forest;
}

////////////////////////////////////////////////////////////////////////////////
//                        Prediction Functions                                //
////////////////////////////////////////////////////////////////////////////////  


// [[Rcpp::export]]
NumericVector predict_class_cpp(List forest, NumericMatrix train){
  NumericVector preds = rep(NA_REAL, train.nrow());
  for(int r=0; r<train.nrow();r++){
    arma::vec outcomes = rep(NA_REAL, forest.size());
    NumericVector obs = train(r,_);
    for(int j = 0; j<forest.size(); j++){
      outcomes(j) = decision_cpp(Rcpp::as<Rcpp::DataFrame>(forest[j]), 1, obs);
    }
    preds[r] = numeric_mode_cpp(outcomes);
  }
  return preds;
}
