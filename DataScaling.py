from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import pandas as pd

cancer = load_breast_cancer()

X_train, X_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, random_state=1)


df = pd.DataFrame(cancer.data)

df.to_csv("/home/axis/Desktop/cancer_data.csv", index=False)


scalar = MinMaxScaler().fit(X_train)

# Transform training data
X_train_scaled = scalar.transform(X_train)

df_scaled = pd.DataFrame(X_train_scaled)

print("per-feature minimum before scaling:\n {}".format(X_train.min(axis=0)))
print("per-feature maximum before scaling:\n {}".format(X_train.max(axis=0)))
print("per-feature minimum after scaling:\n {}".format(
    X_train_scaled.min(axis=0)))
print("per-feature maximum after scaling:\n {}".format(
    X_train_scaled.max(axis=0)))


df_scaled.to_csv("/home/axis/Desktop/cancer_data_scaled.csv", index=False)


# transform test data
X_test_scaled = scalar.transform(X_test)

# print test data properties after scaling
print("per-feature minimum after scaling:\n{}".format(X_test_scaled.min(axis=0)))
print("per-feature maximum after scaling:\n{}".format(X_test_scaled.max(axis=0)))


print("Training: Minimum fetaure value is ", X_train.min())
print("Test Set: Minimum fetaure value is ", X_test.min())


print("Training: Maximum fetaure value is ", X_train.max())
print("Test Set: Maximum fetaure value is ", X_test.max())

print("DONE..")
