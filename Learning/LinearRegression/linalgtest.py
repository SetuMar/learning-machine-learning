import pandas as pd
from sklearn.model_selection import train_test_split
from calcmathlibs import calculate_MSE, calculate_R2
from lingalgtrain import math_train

all_data = pd.read_csv("data/mediumhousing.csv")

# convert all columns to numbers
for c in all_data.columns: all_data[c] = pd.to_numeric(all_data[c], errors='coerce')

X = all_data.drop('price', axis=1)
X['y_int'] = 1

y = all_data['price']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

weights, feature_mins, feature_maxes, label_min, label_max = math_train(X_train, y_train)

for c in feature_mins:  # only normalize columns that were normalized during training
    X_test[c] = (X_test[c] - feature_mins[c]) / (feature_maxes[c] - feature_mins[c])

y_test_norm = (y_test - label_min) / (label_max - label_min)

print(calculate_MSE(weights, X_test, y_test_norm))

mse = calculate_MSE(weights, X_test, y_test_norm)
r2 = calculate_R2(weights, X_test, y_test_norm)

print(f"MSE: {mse}")
print(f"R^2: {r2}")

print(f"Final Weights: {weights}")