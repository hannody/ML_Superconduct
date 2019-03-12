import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import sys
import seaborn as sns

if sys.platform == 'darwin':
    train_path = "/Users/mqa/Desktop/Dev/ML/introduction_to_ml_with_python/Data sets/superconduct/train.csv"
else:
    train_path = "/home/axis/Desktop/ml_work/Data sets/superconduct/train.csv"

X = pd.read_csv(train_path, index_col=0)

y = X["critical_temp"]

X = X.drop(["critical_temp"], axis=1)

print("X_train shape = ", X.shape)

print('y_train shape = ', y.shape)


# The data need to be normalized

print("Done")
