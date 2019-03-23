from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import sys
import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

print(tf.__version__)


if sys.platform == 'darwin':
    train_path = "/Users/mqa/Desktop/Dev/ML/introduction_to_ml_with_python/Data sets/superconduct/train.csv"
else:
    train_path = "/home/axis/Desktop/ml_work/Data sets/superconduct/train.csv"

dataset = pd.DataFrame(pd.read_csv(train_path, na_values="?",
                                   comment='\t', sep=" ", skipinitialspace=True))

train_dataset = dataset.sample(frac=0.9, random_state=0)
test_dataset = dataset.drop(train_dataset.index)

print(train_dataset.shape)
print(test_dataset.shape)

print(train_dataset.head())
