#######################################################################################################
################################      CALCULATE ERRORS       ##########################################
#######################################################################################################

calculate_reg_error = function(forest, outcome, features, training, test){
  predictions = predict_reg_cpp(forest, features)
  
  #Error 1: Resubstitution Estimation
  e1 = mean((outcome[training]-predictions[training])^2)
  
  #Error 2: Test Sample Estimation
  e2 = mean((outcome[test]-predictions[test])^2)
  
  return(c(e1,e2))
}

calculate_class_error = function(forest, outcome, features, training, test){
  predictions = predict_class_cpp(forest, features)
  
  #Error 1: Resubstitution Estimation
  accuracy1 = outcome[training]==predictions[training]
  e1 = sum(accuracy1)/length(accuracy1)
  
  #Error 2: Test Sample Estimation
  accuracy2 = outcome[test]==predictions[test]
  e2 = sum(accuracy2)/length(accuracy2)
  
  return(c(e1,e2))
}