import pandas as pd
from calcmath import *
from commonmath import calculate_MSE, normalize_data

def train(train_features, train_labels):
    features = train_features
    labels = train_labels

    step_size = 0.01

    # convert to pandas DF and Series
    features = pd.DataFrame(features)

    labels = pd.Series(labels)

    features, labels, feature_mins, feature_maxes, label_min, label_max = normalize_data(features, labels)

    # number of weights
    num_weights = features.shape[1]

    # weight of each feature
    feature_weights = [0 for i in range(num_weights)]

    # used in equation calculations
    weight_symbols = generate_weight_symbols(num_weights)

    # current model error
    err = calculate_MSE(feature_weights, features, labels)

    # fine-tune model error
    while err > 1e-3:
        # generate loss function
        loss_func = generate_loss_function(weight_symbols, features, labels)

        # calculate gradient
        grad = calculate_gradient(loss_func, weight_symbols)
        grad_values = dict(zip(weight_symbols, feature_weights))

        # modify feature weights using the gradient
        weight_changes = [g.subs(grad_values) for g in grad]
        feature_weights = [f - step_size * weight_changes[i] for i, f in enumerate(feature_weights)]

        err = calculate_MSE(feature_weights, features, labels)

    return feature_weights, feature_mins, feature_maxes, label_min, label_max