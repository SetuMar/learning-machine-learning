import pandas as pd
import sympy as sp

# creates the loss function from weight symbols, features, and labels
def generate_loss_function(weight_symbols:list, features:pd.DataFrame, labels:pd.Series):
    loss_func = 0
    
    # create a Sum of Squared Errors
    for i, l in enumerate(labels):
        # weight * features summation
        curr_row_features_matrix = sp.Matrix(features.iloc[i])
        sympy_feature_weights = sp.Matrix(weight_symbols)
        
        dotted = curr_row_features_matrix.dot(sympy_feature_weights)
        
        loss_func += (dotted - l) ** 2
    
    # divide by N (Calculate mean of Squared Errors)
    loss_func /= len(labels)
    return loss_func

# calculate gradient
def calculate_gradient(func, symbols):
    return [sp.diff(func, w) for w in symbols]

# generate all of the weight symbols
def generate_weight_symbols(num_elements:int):
    return sp.symbols([f'w{n}' for n in range(num_elements)])