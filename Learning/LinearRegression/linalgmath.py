import numpy as np

def calculate_optimal_weights(features_matrix, labels_vector):
    return np.linalg.inv(features_matrix.transpose() @ features_matrix) @ (features_matrix.transpose() @ labels_vector)

def calculate_gradient(features_matrix:np.matrix, weights, labels_vector):
    return (1 / features_matrix.shape[0]) * (features_matrix.T @ (features_matrix @ weights - labels_vector))