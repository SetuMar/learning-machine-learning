import pandas as pd
from calcmathlibs import generate_weight_symbols, generate_loss_function, calculate_gradient, calculate_MSE

def normalize_col(x, mx, mn):    
    return (x - mn) / (mx - mn)

def train(train_features, train_labels):

    features = train_features
    labels = train_labels

    step_size = 0.01

    # convert to pandas DF and Series and add a y intercept column
    features = pd.DataFrame(features)
    features['y_int'] = 1

    labels = pd.Series(labels)

    # store the min/max used for each column so test data can be
    # normalized the same way later
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

    # number of weights
    num_weights = features.shape[1]

    # weight of each feature
    feature_weights = [0 for i in range(num_weights)]

    # used in equation calculations
    weight_symbols = generate_weight_symbols(num_weights)

    # training loop here and below:

    err = calculate_MSE(feature_weights, features, labels)

    while err > 1e-2:
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