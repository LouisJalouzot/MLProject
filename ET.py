import pandas as pd
import sklearn as sk
import numpy as np
from sklearn import ensemble, model_selection, metrics, neural_network, preprocessing

X = pd.read_csv('~/data/MLProject/xTrainFinal.csv').set_index('sample_id')
Y = pd.read_csv('~/data/MLProject/y_train.csv').set_index('sample_id')
normalizerX = preprocessing.MinMaxScaler()
normalizerY = preprocessing.MinMaxScaler()
X = normalizerX.fit_transform(X)
Y = normalizerY.fit_transform(Y)
xTrain, xTest, yTrain, yTest = model_selection.train_test_split(X, Y, test_size=.1)
reg = neural_network.MLPRegressor(verbose=True, hidden_layer_sizes=(75, 50, 10), tol=1e-5, learning_rate_init=5e-4)
# # reg.fit(X, Y)
reg.fit(xTrain, yTrain)
# # sampleID = xTest['sample_id']
# # res = pd.DataFrame(res, index=sampleID)
# # print(res)
# # pd.DataFrame(res).to_csv("data/HGB.csv")
pred = reg.predict(xTest)
pred = normalizerY.inverse_transform(pred.reshape(-1, 1))
yTest = normalizerY.inverse_transform(yTest)
print(int(np.sqrt(metrics.mean_squared_error(yTest, pred)) * 10000))