import pandas as pd
import sklearn as sk
import numpy as np
from sklearn import ensemble, model_selection, metrics, neural_network, preprocessing
from tensorflow import keras

X = pd.read_csv('~/data/xTrainFinal.csv').set_index('sample_id')
Y = pd.read_csv('data/y_train.csv').set_index('sample_id')
normalizerX = preprocessing.MinMaxScaler()
normalizerY = preprocessing.MinMaxScaler()
X = normalizerX.fit_transform(X)
Y = normalizerY.fit_transform(Y)
xTrain, xTest, yTrain, yTest = model_selection.train_test_split(X, Y, test_size=.1)
yTest = normalizerY.inverse_transform(yTest)
model = keras.models.Sequential([
    keras.layers.Dense(200, activation='sigmoid'),
    keras.layers.Dense(200, activation='sigmoid'),
    keras.layers.Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='mse')
model.fit(xTrain, yTrain, epochs=10)
print(model.summary())
pred = model.predict(xTest)
print(int(np.sqrt(metrics.mean_squared_error(yTest, normalizerY.inverse_transform(pred))) * 10000))