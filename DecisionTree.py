import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from utils import *

RANDOM_STATE = 55

# Load the dataset using pandas
df = pd.read_csv("data.csv")

print(df.head())
df = df.drop('I', axis=1)
df = df.drop('Gravity', axis=1)

## Removing our target variable
var = [x for x in df.columns if x not in ['Class']]

#normalize the dataset
scaler = StandardScaler()
df[var] = scaler.fit_transform(df[var])
# Split the dataset into training and testing sets
print(df.head())

X_train, X_test, y_train, y_test = train_test_split(df[var], df[['Class']], train_size = 0.6, random_state = RANDOM_STATE)

print(f'train samples: {len(X_train)}\ntest samples: {len(X_test)}')

# print("first 5 samples of the train dataset")
# print(X_train[:5])
# print(y_train[:5])
# print("first 5 samples of the test dataset")
# print(X_test[:5])
# print(y_test[:5])

min_samples_split_list = [2, 10, 30, 50, 100, 200, 300] ## If the number is an integer, then it is the actual quantity of samples,
max_depth_list = [1,2, 3, 4, 8, 16, 32, 64, None] # None means that there is no depth limit.

# accuracy_list_train = []
# accuracy_list_test = []


# def entropy(p):
#     if p == 0 or p == 1:
#         return 0
#     else:
#         return -p * np.log2(p) - (1- p)*np.log2(1 - p)
    
# def find_best_splits(X, var, num_thresholds=4):
#     """
#     For each column in var, test various thresholds (evenly spaced between min and max of the column)
#     and return a dictionary with (feature, threshold): (left_indices, right_indices).
#     Assumes X is standardized (mean 0, std 1).
#     """
#     splits = {}
#     for col in var:
#         col_values = X[col].values
#         min_val, max_val = col_values.min(), col_values.max()
#         thresholds = np.linspace(min_val, max_val, num=num_thresholds, endpoint=False)[1:]  # skip min
#         for threshold in thresholds:
#             left_indices = [i for i, x in enumerate(col_values) if x <= threshold]
#             right_indices = [i for i, x in enumerate(col_values) if x > threshold]
#             splits[(col, threshold)] = (left_indices, right_indices)
#     return splits

# thresholds = find_best_splits(X_train, var)
# print(thresholds)

# def split_indices(X_column, threshold):
#     """
#     Splits indices of X_column based on a continuous threshold.
#     Returns indices for left (<= threshold) and right (> threshold) splits.
#     """
#     left_indices = [i for i, x in enumerate(X_column) if x <= threshold]
#     right_indices = [i for i, x in enumerate(X_column) if x > threshold]
#     return left_indices, right_indices

accuracy_list_train = []
accuracy_list_test = []
for min_samples_split in min_samples_split_list:
    # You can fit the model at the same time you define it, because the fit function returns the fitted estimator.
    model = DecisionTreeClassifier(min_samples_split = min_samples_split,
                                   random_state = RANDOM_STATE).fit(X_train,y_train) 
    predictions_train = model.predict(X_train) ## The predicted values for the train dataset
    predictions_test = model.predict(X_test) ## The predicted values for the test dataset
    accuracy_train = accuracy_score(predictions_train,y_train)
    accuracy_test = accuracy_score(predictions_test,y_test)
    accuracy_list_train.append(accuracy_train)
    accuracy_list_test.append(accuracy_test)

plt.title('Train x Test metrics')
plt.xlabel('min_samples_split')
plt.ylabel('accuracy')
plt.xticks(ticks = range(len(min_samples_split_list )),labels=min_samples_split_list)
plt.plot(accuracy_list_train)
plt.plot(accuracy_list_test)
plt.legend(['Train','Test'])