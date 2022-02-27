import numpy as np
import pandas as pd
import plotly.express as px

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import tensorflow as tf

from tensorflow_addons.metrics import RSquare

data = pd.read_csv(r'C:\Users\emmaa\Documents\food.csv')
data.isna().sum()
data = data.drop('type_sports', axis=1)

numeric_nulls = [column for column in data.columns if data.dtypes[column] != 'object' and data.isna().sum()[column] != 0]
for column in numeric_nulls:
   data[column] = data[column].fillna(data[column].mean())
   {column: list(data[column].unique()) for column in data.columns if data.isna().sum()[column] > 0}

nonnumeric_nulls = [column for column in data.columns if data.dtypes[column] == 'object' and data.isna().sum()[column] != 0]
nonnumeric_nulls.remove('gpa')

data = data.drop(nonnumeric_nulls, axis=1)
y = data.loc[:, 'gpa']
X = data.drop('gpa', axis=1)

scaler = StandardScaler()

X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=22)

inputs = tf.keras.Input(shape=(48,))
x = tf.keras.layers.Dense(256, activation='relu')(inputs)
x = tf.keras.layers.Dense(512, activation='relu')(x)
x = tf.keras.layers.Dense(512, activation='relu')(x)
x = tf.keras.layers.Dense(256, activation='relu')(x)
outputs = tf.keras.layers.Dense(1, activation='linear')(x)

model = tf.keras.Model(inputs, outputs)


model.compile(
    optimizer='adam',
    loss='mse'
)