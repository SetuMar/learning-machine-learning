import pandas as pd

# normalize as a value between 0 and 1
def normalize_col(x, mx, mn):    
    return (x - mn) / (mx - mn)

# given the weights and features, predict each value
def predict(feature_weights:list, features:pd.Series):
    pred_val = 0
    
    f = list(features)
        
    # add feature_weight * feature value
    for i, w in enumerate(feature_weights):
        pred_val += w * f[i]
        
    return pred_val

# calculate the error between each feature and the label
def calculate_MSE(feature_weights, features, labels):
    mse = 0
    for i, l in enumerate(labels):
        mse += (predict(feature_weights, features.iloc[i]) - l) ** 2
    
    mse /= len(labels)
    
    return mse

def calculate_R2(feature_weights, features, labels):
    # mean of actual labels - "dumbest possible model"
    label_mean = sum(labels) / len(labels)
 
     # sum of squared residuals
    ss_res = 0
     # total sum of squares
    ss_tot = 0
    
    # calculate r^2
    for i in range(len(labels)):
        l = labels.iloc[i] if hasattr(labels, 'iloc') else labels[i]
        pred = predict(feature_weights, features.iloc[i])
        ss_res += (l - pred) ** 2
        ss_tot += (l - label_mean) ** 2
 
    return 1 - (ss_res / ss_tot)

def normalize_data(features: pd.DataFrame, labels: pd.Series):
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

    return features, labels, feature_mins, feature_maxes, label_min, label_max
