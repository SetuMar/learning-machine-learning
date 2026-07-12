import numpy as np
import pandas as pd
from calctrain import normalize_col
from linalgmath import calculate_optimal_weights, calculate_gradient
from calcmathlibs import calculate_MSE

def math_train(features:pd.DataFrame, labels:pd.Series):
    feature_mins = {}
    feature_maxes = {}

    for c in features.columns:
        try:
            mn = min(features[c])
            mx = max(features[c])
            features[c] = features[c].apply(normalize_col, mx=mx, mn=mn)
            feature_mins[c] = mn
            feature_maxes[c] = mx
        except:
            pass

    label_min = min(labels)
    label_max = max(labels)
    labels = labels.apply(normalize_col, mx=label_max, mn=label_min)
    
    features_matrix = features.to_numpy()
    labels_matrix = labels.to_numpy()
    
    weights = calculate_optimal_weights(features_matrix, labels_matrix)
    
    return weights, feature_mins, feature_maxes, label_min, label_max

def grad_train(features:pd.DataFrame, labels:pd.Series):
    feature_mins = {}
    feature_maxes = {}

    for c in features.columns:
        try:
            mn = min(features[c])
            mx = max(features[c])
            features[c] = features[c].apply(normalize_col, mx=mx, mn=mn)
            feature_mins[c] = mn
            feature_maxes[c] = mx
        except:
            pass

    label_min = min(labels)
    label_max = max(labels)
    labels = labels.apply(normalize_col, mx=label_max, mn=label_min)
    
    features_matrix = features.to_numpy()
    labels_vector = labels.to_numpy()
    
    weights = np.zeros(features_matrix.shape[1])
    
    err = calculate_MSE(weights, features, labels)
    
    while err > 1e-2:
        weights -= calculate_gradient(features_matrix, weights, labels_vector)
        err = calculate_MSE(weights, features, labels)
    
    return weights, feature_mins, feature_maxes, label_min, label_max