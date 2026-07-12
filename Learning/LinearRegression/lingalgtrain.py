import numpy as np
import pandas as pd
from commonmath import normalize_col, normalize_data, calculate_MSE
from linalgmath import calculate_optimal_weights, calculate_gradient

# obtain optimal model weights
def math_train(features:pd.DataFrame, labels:pd.Series):
    features, labels, feature_mins, feature_maxes, label_min, label_max = normalize_data(features, labels)
    
    features_matrix = features.to_numpy()
    labels_matrix = labels.to_numpy()
    
    weights = calculate_optimal_weights(features_matrix, labels_matrix)
    
    return weights, feature_mins, feature_maxes, label_min, label_max

# obtain good model weights using gradient descent
def grad_train(features:pd.DataFrame, labels:pd.Series):
    features, labels, feature_mins, feature_maxes, label_min, label_max = normalize_data(features, labels)
    
    features_matrix = features.to_numpy()
    labels_vector = labels.to_numpy()
    
    weights = np.zeros(features_matrix.shape[1])
    
    err = calculate_MSE(weights, features, labels)
    
    while err > 1e-2:
        weights -= calculate_gradient(features_matrix, weights, labels_vector)
        err = calculate_MSE(weights, features, labels)
    
    return weights, feature_mins, feature_maxes, label_min, label_max