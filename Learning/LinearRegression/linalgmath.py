import numpy as np

def calculate_optimal_weights(features_matrix, labels_matrix):
    return np.linalg.inv(features_matrix.transpose() @ features_matrix) @ (features_matrix.transpose() @ labels_matrix)