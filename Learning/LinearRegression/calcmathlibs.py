import pandas as pd
import sympy as sp

def predict(feature_weights:list, features:pd.Series):
    pred_val = 0
    
    f = list(features)
        
    # add feature_weight * feature value
    for i, w in enumerate(feature_weights):
        pred_val += w * f[i]
        
    return pred_val

def calculate_MSE(feature_weights, features, labels):
    mse = 0
    for i, l in enumerate(labels):
        mse += (predict(feature_weights, features.iloc[i]) - l) ** 2
    
    mse /= len(labels)
    
    return mse

def generate_loss_function(weight_symbols:list, features:pd.DataFrame, labels:pd.Series):
    loss_func = 0
    
    for i, l in enumerate(labels):
        # weight * features summation
        curr_row_features_matrix = sp.Matrix(features.iloc[i])
        sympy_feature_weights = sp.Matrix(weight_symbols)
        
        dotted = curr_row_features_matrix.dot(sympy_feature_weights)
        
        loss_func += (dotted - l) ** 2
    
    # divide by N
    loss_func /= len(labels)
    return loss_func

def calculate_gradient(func, symbols):
    # calculate gradient
    gradient = [sp.diff(func, w) for w in symbols]
    
    return gradient

def generate_weight_symbols(num_elements:int):
    return sp.symbols([f'w{n}' for n in range(num_elements)])

def calculate_R2(feature_weights, features, labels):
    # mean of actual labels — used as the "dumbest possible model" baseline
    label_mean = sum(labels) / len(labels)
 
    ss_res = 0  # sum of squared residuals (your model's errors)
    ss_tot = 0  # total sum of squares (errors if you just predicted the mean)
 
    for i in range(len(labels)):
        l = labels.iloc[i] if hasattr(labels, 'iloc') else labels[i]
        pred = predict(feature_weights, features.iloc[i])
        ss_res += (l - pred) ** 2
        ss_tot += (l - label_mean) ** 2
 
    return 1 - (ss_res / ss_tot)