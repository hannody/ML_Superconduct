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

# Transform data
X_train_scaled = scalar.transform(X_train)

df_scaled = pd.DataFrame(X_train_scaled)

df_scaled.to_csv("/home/axis/Desktop/cancer_data_scaled.csv", index=False)

print("DONE..")
