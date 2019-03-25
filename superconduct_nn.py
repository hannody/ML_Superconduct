from keras import layers
from keras import models
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import sys
import numpy as np
import datetime

print("started at:", datetime.datetime.now())

if sys.platform == 'darwin':
    train_path = "/Users/mqa/Desktop/Dev/ML/introduction_to_ml_with_python/Data sets/superconduct/train.csv"
else:
    train_path = "/home/axis/Desktop/ml_work/Data sets/superconduct/train.csv"

dataset = pd.read_csv(train_path, sep=" ", delimiter=',')

print(dataset.shape)


Y = dataset.pop('critical_temp').values

Y = np.reshape(Y, (Y.shape[0], 1))
print(Y.shape)

print(type(Y))

X = dataset.values

print(X.shape, type(X))


train_data, test_data, train_targets, test_targets = train_test_split(
    X, Y, random_state=42, test_size=0.15)


print(train_targets.shape)
print(test_targets.shape)


print(type(train_data), train_data.shape)
print(type(test_data), test_data.shape)


rb_scaler = RobustScaler()

train_data = rb_scaler.fit_transform(train_data)
test_data = rb_scaler.transform(test_data)


def build_model():  # 5.6
    model = models.Sequential()
    model.add(layers.Dense(256, activation='relu',
                           input_shape=(train_data.shape[1],)))
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(1))
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model


def build_model2():  # 5.369
    model = models.Sequential()
    model.add(layers.Dense(256, activation='relu',
                           input_shape=(train_data.shape[1],)))
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(1))
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model


model = build_model()
model.fit(train_data, train_targets,
          epochs=80, batch_size=16, verbose=0)
test_mse_score, test_mae_score = model.evaluate(test_data, test_targets)

print(test_mae_score)
print(test_mse_score)
print("DONE", "Finished On:", datetime.datetime.now())
